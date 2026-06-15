from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are OpenHands agent, a helpful AI coding assistant."},
        {"role": "user", "content": prompt},
    ]


def _completion_tokens(completion: dict[str, Any]) -> int:
    return int((completion.get("usage") or {}).get("completion_tokens") or 0)


class MockNanoRolloutHandler(BaseHTTPRequestHandler):
    reward: float = 1.0
    reward_mode: str = "fixed"
    force_static: bool = False
    vary_prompt: bool = False
    multi_turn: bool = False

    def log_message(self, fmt: str, *args) -> None:
        print(f"[mock-nanorollout] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "healthy"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/run":
            self._send_json(404, {"error": "not found"})
            return

        started = time.time()
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        if self.multi_turn:
            self._handle_multi_turn_run(body, started)
            return

        sample_index = self._sample_index(body)
        extra_args = body.get("extra_args") or {}
        prompt = str(extra_args.get("prompt") or body.get("instance_id") or "Return PASS.")
        if self.vary_prompt and sample_index is not None:
            prompt = f"{prompt}\n\nTrial index: {sample_index}."
        messages = _build_messages(prompt)
        model_text = "PASS"
        model_query_time = 0.0
        reward = self._reward_for_sample(sample_index)

        base_url = (body.get("base_url") or "").rstrip("/")
        if base_url and not self.force_static:
            sampling_params = body.get("sampling_params") or {}
            chat_payload = {
                "model": body.get("model_name") or "local-model",
                "messages": messages,
                "temperature": sampling_params.get("temperature", 0.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "max_tokens": sampling_params.get("max_tokens", sampling_params.get("max_new_tokens", 8)),
            }
            headers = {}
            if body.get("api_key"):
                headers["Authorization"] = f"Bearer {body['api_key']}"
            try:
                query_start = time.time()
                completion = _post_json(f"{base_url}/chat/completions", chat_payload, headers)
                model_query_time = time.time() - query_start
                model_msg = completion.get("choices", [{}])[0].get("message", {})
                if model_msg.get("content"):
                    model_text = str(model_msg["content"])
                messages.append({"role": "assistant", "content": model_text})
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError) as exc:
                messages.append({"role": "assistant", "content": "PASS"})
                self._send_json(
                    200,
                    {
                        "reward": 0.0,
                        "messages": messages,
                        "exit_status": "error",
                        "agent_metrics": {
                            "turns": 1,
                            "tool_calls": 0,
                            "model_query_time_sum": model_query_time,
                            "total_time": time.time() - started,
                            "agent_run_time": time.time() - started,
                        },
                        "metadata": {"error": str(exc), "mock": True},
                        "tools": None,
                    },
                )
                return
        else:
            messages.append({"role": "assistant", "content": model_text})

        self._send_json(
            200,
            {
                "reward": reward,
                "messages": messages,
                "exit_status": "finished",
                "agent_metrics": {
                    "turns": 1,
                    "tool_calls": 0,
                    "model_query_time_sum": model_query_time,
                    "env_execution_time_sum": 0.0,
                    "eval_time": 0.0,
                    "agent_run_time": time.time() - started,
                    "total_time": time.time() - started,
                },
                "metadata": {
                    "mock": True,
                    "model_text": model_text,
                    "mock_reward_mode": self.reward_mode,
                    "mock_sample_index": sample_index,
                },
                "tools": None,
            },
        )

    def _handle_multi_turn_run(self, body: dict[str, Any], started: float) -> None:
        """One toy agent run: starts from extra_args.initial_messages if given
        (else the task prompt), chats through the supplied model endpoint
        (a fake observation between turns) until the conversation reaches
        extra_args.token_budget tokens (checked against the usage stats of
        each completion). It then performs one compaction turn and returns the
        conversation to the orchestrator."""
        extra_args = body.get("extra_args") or {}
        sample_index = self._sample_index(body)
        token_budget = int(extra_args.get("token_budget") or 4096)
        max_output_tokens = int(extra_args.get("max_output_tokens") or 10**12)

        initial_messages = extra_args.get("initial_messages")
        if initial_messages:
            messages = [dict(m) for m in initial_messages]
        else:
            prompt = str(extra_args.get("prompt") or body.get("instance_id") or "Solve the task.")
            if self.vary_prompt and sample_index is not None:
                prompt = f"{prompt}\n\nTrial index: {sample_index}."
            messages = _build_messages(prompt)

        base_url = (body.get("base_url") or "").rstrip("/")
        sampling_params = body.get("sampling_params") or {}
        headers = {}
        if body.get("api_key"):
            headers["Authorization"] = f"Bearer {body['api_key']}"

        model_query_time = 0.0
        turns = 0
        context_tokens = 0
        completion_tokens = 0

        def chat_payload(max_tokens: int) -> dict[str, Any]:
            return {
                "model": body.get("model_name") or "local-model",
                "messages": messages,
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "max_tokens": max(1, max_tokens),
            }

        def request_completion(max_tokens: int) -> tuple[dict[str, Any], str] | None:
            nonlocal model_query_time
            try:
                query_start = time.time()
                completion = _post_json(f"{base_url}/chat/completions", chat_payload(max_tokens), headers)
                model_query_time += time.time() - query_start
                content = completion.get("choices", [{}])[0].get("message", {}).get("content") or "..."
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
                self._send_json(
                    200,
                    {
                        "reward": 0.0,
                        "messages": messages,
                        "exit_status": "error",
                        "agent_metrics": {
                            "turns": turns,
                            "tool_calls": 0,
                            "completion_tokens": completion_tokens,
                            "total_time": time.time() - started,
                        },
                        "metadata": {"error": str(exc), "mock": True, "multi_turn": True},
                        "tools": None,
                    },
                )
                return None
            return completion, str(content)

        def remaining_turn_tokens() -> int:
            requested_tokens = int(sampling_params.get("max_tokens", sampling_params.get("max_new_tokens", 32)))
            return min(requested_tokens, max_output_tokens - completion_tokens)

        while context_tokens < token_budget and completion_tokens < max_output_tokens:
            result = request_completion(remaining_turn_tokens())
            if result is None:
                return
            completion, content = result
            context_tokens = int(completion.get("usage", {}).get("total_tokens") or token_budget)
            completion_tokens += _completion_tokens(completion)
            messages.append({"role": "assistant", "content": str(content)})
            turns += 1
            if context_tokens < token_budget:
                messages.append({"role": "user", "content": f"Observation {turns}: environment output. Continue."})

        if context_tokens >= token_budget and completion_tokens < max_output_tokens:
            messages.append({"role": "user", "content": "Summarize the current state for the next rollout."})
            result = request_completion(remaining_turn_tokens())
            if result is None:
                return
            completion, content = result
            completion_tokens += _completion_tokens(completion)
            messages.append({"role": "assistant", "content": str(content)})
            turns += 1

        self._send_json(
            200,
            {
                "reward": self._reward_for_sample(sample_index),
                "messages": messages,
                "exit_status": "finished",
                "agent_metrics": {
                    "turns": turns,
                    "tool_calls": 0,
                    "context_tokens": context_tokens,
                    "completion_tokens": completion_tokens,
                    "model_query_time_sum": model_query_time,
                    "agent_run_time": time.time() - started,
                    "total_time": time.time() - started,
                },
                "metadata": {
                    "mock": True,
                    "multi_turn": True,
                    "mock_sample_index": sample_index,
                },
                "tools": None,
            },
        )

    @classmethod
    def _sample_index(cls, body: dict[str, Any]) -> int | None:
        """Episode index from the orchestrator task id (…-ep{k} or …-ep{k}-seg{j}),
        so alternating rewards vary per episode while segments of one episode
        share theirs. Falls back to a bare trailing integer (legacy ids)."""
        api_key = body.get("api_key")
        if not isinstance(api_key, str):
            return None
        match = re.search(r"-ep(\d+)(?:-seg\d+)?$", api_key)
        if match:
            return int(match.group(1))
        try:
            return int(api_key.rsplit("-", 1)[-1])
        except ValueError:
            return None

    @classmethod
    def _reward_for_sample(cls, sample_index: int | None) -> float:
        if cls.reward_mode == "alternating" and sample_index is not None:
            return cls.reward if sample_index % 2 == 0 else 0.0
        return cls.reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic NanoRollout-compatible smoke server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=("fixed", "alternating"), default="fixed")
    parser.add_argument("--vary-prompt", action="store_true", help="Append the sample index to the prompt.")
    parser.add_argument("--static", action="store_true", help="Do not call the supplied model endpoint.")
    parser.add_argument("--multi-turn", action="store_true", help="Simulate a multi-turn agent with a turn budget.")
    args = parser.parse_args()

    MockNanoRolloutHandler.reward = args.reward
    MockNanoRolloutHandler.reward_mode = args.reward_mode
    MockNanoRolloutHandler.force_static = args.static
    MockNanoRolloutHandler.vary_prompt = args.vary_prompt
    MockNanoRolloutHandler.multi_turn = args.multi_turn
    server = ThreadingHTTPServer((args.host, args.port), MockNanoRolloutHandler)
    print(
        "mock NanoRollout server listening on "
        f"http://{args.host}:{args.port} reward_mode={args.reward_mode} vary_prompt={args.vary_prompt} "
        f"multi_turn={args.multi_turn}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
