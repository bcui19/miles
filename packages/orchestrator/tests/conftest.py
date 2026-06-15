"""Shared no-GPU stack for orchestrator end-to-end tests.

Topology (docs/orchestrator-design.md):

    orchestrator (under test)
      -> stub NanoRollout /run  (acts as the agent: calls the proxy's
                                 chat/completions with the run's api_key)
      -> real TITO proxy        (stub tokenizer, stub sglang behind it,
                                 real /v1/finalize artifact writing)
      -> real Miles train buffer (real generate_rollout/waiters)
      <- fake train loop thread  (pop -> advantages -> training_started ->
                                  weights_synced)

Requires PYTHONPATH to include packages/miles and the miles tito_proxy test
dir (for _stub_heavy_deps and stub_stack).
"""

import re
import socket
import threading
from argparse import Namespace

import _stub_heavy_deps  # noqa: F401  (must precede miles imports on dev machines)
import httpx
import pytest
from fastapi import FastAPI, Request
from stub_stack import StubTokenizer, make_stub_sglang, serve

from miles.rollout.nanorollout.proxy.tito_server import TITOProxy
from miles.rollout.train_service import train_buffer
from miles.rollout.train_service.advantages import post_process_rewards


def make_stub_nanorollout():
    """Stub NanoRollout: one 'agent turn' through the inference endpoint it
    was handed, deterministic reward from the task id's trailing digit
    (even -> 1.0, odd -> 0.0) so groups have within-group reward variance."""
    app = FastAPI()

    async def chat(base_url, task_id, messages, max_tokens):
        async with httpx.AsyncClient(timeout=30.0) as client:
            completion = await client.post(
                f"{base_url}/chat/completions",
                json={"messages": messages, "max_tokens": max(1, max_tokens)},
                headers={"Authorization": f"Bearer {task_id}"},
            )
        completion.raise_for_status()
        return completion.json()

    @app.post("/run")
    async def run(request: Request):
        payload = await request.json()
        task_id = payload["api_key"]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": payload["extra_args"]["prompt"]},
        ]
        initial = payload["extra_args"].get("initial_messages")
        if initial:
            messages = initial
        metadata = {}
        completion_tokens = 0
        turns = 0
        if payload["base_url"]:
            max_output_tokens = int(payload["extra_args"].get("max_output_tokens") or 10**9)
            completion = await chat(payload["base_url"], task_id, messages, max_output_tokens)
            assistant = completion["choices"][0]["message"]
            messages = [*messages, assistant]
            completion_tokens += int(completion.get("usage", {}).get("completion_tokens") or 0)
            turns += 1
            token_budget = payload["extra_args"].get("token_budget")
            total_tokens = int(completion.get("usage", {}).get("total_tokens") or 0)
            if (
                token_budget is not None
                and total_tokens >= int(token_budget)
                and completion_tokens < max_output_tokens
            ):
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": "Summarize the current state for the next rollout.",
                    },
                ]
                completion = await chat(payload["base_url"], task_id, messages, max_output_tokens - completion_tokens)
                messages = [*messages, completion["choices"][0]["message"]]
                completion_tokens += int(completion.get("usage", {}).get("completion_tokens") or 0)
                turns += 1
        else:
            # agent pointed at a plain (non-proxy) endpoint in this stub: no capture
            assistant = {"role": "assistant", "content": "offline"}
            messages = [*messages, assistant]
            turns = 1
        episode_part = re.sub(r"-seg\d+$", "", task_id)  # reward keyed by episode, not segment
        reward = 1.0 if int(episode_part[-1]) % 2 == 0 else 0.0
        return {
            "messages": messages,
            "reward": reward,
            "exit_status": "finished",
            "agent_metrics": {"turns": turns, "completion_tokens": completion_tokens},
            "metadata": metadata,
        }

    return app


class FakeTrainLoop:
    """Drives the buffer exactly like train.py: generate -> training_started
    -> (advantages, as RolloutManager would) -> weights_synced."""

    def __init__(self, args, steps):
        self.args = args
        self.steps = steps
        self.batches = []
        self.advantages = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for rollout_id in range(self.steps):
            output = train_buffer.generate_rollout(self.args, rollout_id, data_source=None)
            train_buffer.training_started(rollout_id)
            flat = [s for group in output.samples for s in group]
            self.batches.append(output)
            self.advantages.append(post_process_rewards(self.args, flat))
            train_buffer.weights_synced(rollout_id)

    def start(self):
        self.thread.start()

    def join(self, timeout=60):
        self.thread.join(timeout=timeout)
        assert not self.thread.is_alive(), "fake train loop did not finish"


@pytest.fixture(scope="session")
def stack(tmp_path_factory):
    artifact_root = tmp_path_factory.mktemp("artifacts")
    sglang_url, _ = serve(make_stub_sglang())
    proxy = TITOProxy(
        sglang_base_url=sglang_url,
        tokenizer=StubTokenizer(),
        args=Namespace(sglang_tool_call_parser=None, sglang_router_ip="127.0.0.1", use_rollout_routing_replay=False),
        artifact_root=str(artifact_root),
    )
    nanorollout_url, _ = serve(make_stub_nanorollout())

    with socket.socket() as sock:
        sock.bind(("", 0))
        buffer_port = sock.getsockname()[1]
    buffer_args = Namespace(
        rollout_batch_size=2,
        train_buffer_port=buffer_port,
        train_buffer_host="127.0.0.1",
        train_buffer_pad_multiple=1,
        grpo_std_normalization=False,
    )
    train_buffer._get_server(buffer_args)

    return Namespace(
        proxy=proxy,
        inference_url=proxy.base_url,
        nanorollout_url=nanorollout_url,
        buffer_url=f"http://127.0.0.1:{buffer_port}",
        buffer_args=buffer_args,
        artifact_root=str(artifact_root),
    )
