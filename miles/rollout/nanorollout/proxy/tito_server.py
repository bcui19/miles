from __future__ import annotations

import asyncio
import base64
import logging
import socket
import threading

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .tito_converter import ChatConverter
from .tito_state import TaskState, copy_message, derive_assistant_affixes, hash_message, stable_message_json

logger = logging.getLogger(__name__)


def _openai_error(message: str, error_type: str, status_code: int) -> JSONResponse:
    return JSONResponse(content={"error": {"message": message, "type": error_type}}, status_code=status_code)


class _HistoryMutatedError(Exception):
    def __init__(self, index: int):
        self.index = index


def _short_json(message: dict | None, limit: int = 4000) -> str:
    """For debug logging: a recorded/current message for diagnostics without huge log lines."""
    if message is None:
        return "<missing>"
    text = stable_message_json(message)
    if len(text) > limit:
        return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
    return text


def _diff_message_summary(recorded: dict | None, current: dict | None) -> str:
    """For debug logging: summarize why two message dicts differ when their TITO history hashes mismatch."""
    if recorded is None or current is None:
        return "recorded/current message missing"
    recorded_keys = set(recorded)
    current_keys = set(current)
    parts: list[str] = []
    if recorded_keys != current_keys:
        parts.append(
            f"keys recorded_only={sorted(recorded_keys - current_keys)} current_only={sorted(current_keys - recorded_keys)}"
        )
    for key in sorted(recorded_keys & current_keys):
        if recorded[key] != current[key]:
            parts.append(f"{key}: recorded={repr(recorded[key])[:500]} current={repr(current[key])[:500]}")
    return "; ".join(parts) or "hash mismatch with no top-level difference"


class TITOProxy:
    """Local OpenAI-compatible proxy that records exact SGLang token output.

    Captured trajectories leave the proxy only as on-disk artifacts: the
    orchestrator's rollout helper calls ``POST /v1/finalize`` and the
    consolidated token data is written under ``artifact_root``
    (docs/orchestrator-design.md). ``port=None`` binds an ephemeral
    port (tests); deployments pass a fixed port via the launcher.
    """

    def __init__(self, sglang_base_url: str, tokenizer, args, artifact_root: str, port: int | None = None):
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.args = args
        self.artifact_root = artifact_root
        raw_max_model_len = getattr(args, "max_model_len", None)
        self.max_model_len: int | None = int(raw_max_model_len) if raw_max_model_len is not None else None
        self.tasks: dict[str, TaskState] = {}
        self.converters: dict[str, ChatConverter] = {}

        if getattr(args, "use_rollout_routing_replay", False) and (
            getattr(args, "num_layers", None) is None or getattr(args, "moe_router_topk", None) is None
        ):
            raise ValueError("rollout routing replay requires --num-layers and --moe-router-topk")

        self.assistant_prefix_ids, self.assistant_suffix_ids = derive_assistant_affixes(tokenizer)
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None)
        self.tool_call_parser = getattr(args, "sglang_tool_call_parser", None)
        self.host_ip = getattr(args, "sglang_router_ip", "127.0.0.1")
        self.port = port if port is not None else self._find_free_port()
        self._start_server()

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host_ip}:{self.port}/v1"

    def _start_server(self) -> None:
        app = FastAPI()
        app.post("/v1/chat/completions")(self.handle_chat_completion)
        app.post("/v1/finalize")(self.handle_finalize)
        app.post("/v1/discard")(self.handle_discard)
        app.get("/health")(self.handle_health)
        thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": self.port, "log_level": "warning"},
            daemon=True,
        )
        thread.start()
        logger.info("TITOProxy server started at %s", self.base_url)

    async def handle_chat_completion(self, request: Request) -> JSONResponse:
        request_data = await request.json()
        messages = request_data.get("messages", [])
        tools = request_data.get("tools")
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not api_key:
            return _openai_error("Authorization bearer token (task id) is required", "authentication_error", 401)

        # The TITO protocol expects one sequential request stream per task id;
        # concurrent requests with the same id would race on the captured token history.
        is_new_task = api_key not in self.tasks
        if is_new_task:
            state = TaskState(
                assistant_prefix_ids=self.assistant_prefix_ids,
                assistant_suffix_ids=self.assistant_suffix_ids,
            )
            converter = ChatConverter(
                tokenizer=self.tokenizer,
                tool_call_parser=self.tool_call_parser,
                reasoning_parser=self.reasoning_parser,
            )
        else:
            state = self.tasks[api_key]
            converter = self.converters[api_key]
        pre_msg_length = state.get_num_recorded_messages()

        def _prepare_tokenization() -> tuple[list, list[str], list[dict], list[int]]:
            # TITO: generation input is the append-only token stream — new
            # (non-assistant) messages are tokenized once; generated tokens are
            # reused verbatim, never re-encoded. History is immutable: clients
            # must echo recorded messages verbatim, verified by hash.
            if len(messages) < pre_msg_length:
                raise _HistoryMutatedError(len(messages))
            for index, (recorded, msg) in enumerate(zip(state.message_hashes, messages[:pre_msg_length], strict=True)):
                if recorded != hash_message(msg):
                    raise _HistoryMutatedError(index)
            new_message_items = converter.tokenize_new_messages(messages, pre_msg_length, tools=tools)
            new_hashes = [hash_message(m) for m in messages[pre_msg_length:]]
            new_records = [copy_message(m) for m in messages[pre_msg_length:]]
            input_ids = state.get_input_ids()
            for item in new_message_items:
                input_ids.extend(item.token_ids)
            input_ids += self.assistant_prefix_ids
            return new_message_items, new_hashes, new_records, input_ids

        try:
            new_message_items, new_hashes, new_records, input_ids = await asyncio.to_thread(_prepare_tokenization)
        except _HistoryMutatedError as exc:
            recorded_message = state.message_records[exc.index] if exc.index < len(state.message_records) else None
            current_message = messages[exc.index] if exc.index < len(messages) else None
            logger.error("History mutated for %s at message index %d — TITO history is immutable.", api_key, exc.index)
            logger.error(
                "History mutation detail for %s index %d: %s",
                api_key,
                exc.index,
                _diff_message_summary(recorded_message, current_message),
            )
            logger.error("History mutation recorded[%d]=%s", exc.index, _short_json(recorded_message))
            logger.error("History mutation current[%d]=%s", exc.index, _short_json(current_message))
            return _openai_error(
                f"recorded history mutated at message index {exc.index}; TITO history is immutable",
                "history_mutated",
                409,
            )
        sampling_params = {
            key: value
            for key, value in request_data.items()
            if key in ("temperature", "top_p", "top_k", "stop", "no_stop_trim")
        }
        requested_max_new_tokens = int(request_data.get("max_tokens", request_data.get("max_new_tokens", 4096)))
        max_new_tokens = requested_max_new_tokens
        if self.max_model_len is not None:
            # Cut max_new_tokens to the remaining context; otherwise SGLang will
            # reject the request. This can silently lower the client's requested cap.
            context_remaining = self.max_model_len - len(input_ids)
            if context_remaining <= 0:
                logger.warning(
                    "TITO context exhausted for %s: input_tokens=%d max_model_len=%d",
                    api_key,
                    len(input_ids),
                    self.max_model_len,
                )
                return _openai_error(
                    (
                        f"Requested input has {len(input_ids)} tokens, which exceeds the configured "
                        f"model context length of {self.max_model_len} tokens."
                    ),
                    "context_length_exceeded",
                    400,
                )
            max_new_tokens = min(max_new_tokens, context_remaining)
            if max_new_tokens < requested_max_new_tokens:
                logger.info(
                    "TITO clipped max_new_tokens for %s: requested=%d effective=%d input_tokens=%d "
                    "context_remaining=%d max_model_len=%d",
                    api_key,
                    requested_max_new_tokens,
                    max_new_tokens,
                    len(input_ids),
                    context_remaining,
                    self.max_model_len,
                )
        sampling_params["max_new_tokens"] = max_new_tokens
        sampling_params["skip_special_tokens"] = False

        generate_payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "return_routed_experts": getattr(self.args, "use_rollout_routing_replay", False),
            "routed_experts_start_len": state.routed_experts_len,
        }

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(f"{self.sglang_base_url}/generate", json=generate_payload)
            output = resp.json()

        if "text" not in output:
            error_msg = output.get("error", output)
            logger.warning("SGLang generation failed for %s: %s", api_key, error_msg)
            return _openai_error(
                str(error_msg), "invalid_request_error", resp.status_code if resp.status_code >= 400 else 400
            )

        meta_info = output.get("meta_info", {})
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        output_ids = [item[1] for item in output_token_logprobs]
        output_logprobs = [item[0] for item in output_token_logprobs]

        if generate_payload["return_routed_experts"] and "routed_experts" not in meta_info:
            raise RuntimeError(
                f"routing replay enabled but SGLang returned no routed_experts for {api_key} "
                f"(meta_keys={sorted(meta_info.keys())})"
            )
        routed_experts = None
        if experts_b64 := meta_info.get("routed_experts"):
            # incremental R3 chunk (random_async protocol): one row per token
            # processed since routed_experts_start_len — the re-prefilled tail
            # plus the new outputs, all but the last token
            rows = len(input_ids) + len(output_ids) - 1 - state.routed_experts_len
            raw = np.frombuffer(base64.b64decode(experts_b64.encode("ascii")), dtype=np.int32)
            routed_experts = raw.reshape(rows, self.args.num_layers, self.args.moe_router_topk)

        response_payload = converter.build_response(request_data, output, output["text"])
        assistant_message = response_payload["choices"][0]["message"]
        if is_new_task:
            # Publish new task state only after generation succeeds, so failed
            # first requests cannot leave partially captured history behind.
            self.tasks[api_key] = state
            self.converters[api_key] = converter
        state.add_message_items(new_message_items)
        state.message_hashes.extend(new_hashes)
        state.message_records.extend(new_records)
        state.add_response(
            output_ids,
            output_logprobs,
            routed_experts,
            msg_hash=hash_message(assistant_message),
            msg_record=assistant_message,
        )
        usage = response_payload.setdefault("usage", {})
        usage["prompt_tokens"] = len(input_ids)
        usage["completion_tokens"] = len(output_ids)
        usage["total_tokens"] = len(input_ids) + len(output_ids)
        return JSONResponse(content=response_payload)

    async def handle_health(self) -> JSONResponse:
        return JSONResponse(content={"ok": True, "inference_url": self.base_url})

    async def handle_finalize(self, request: Request) -> JSONResponse:
        """Consolidate one trajectory and write it to disk; returns the
        artifact path. Called by the orchestrator's rollout helper after the
        NanoRollout ``/run`` response returned. Control plane only: the
        response carries a path string, never token data."""
        request_data = await request.json()
        task_id = request_data.get("task_id")
        if not task_id:
            return _openai_error("task_id is required", "invalid_request_error", 400)

        result = self._consume_task(task_id)
        if result is None:
            return _openai_error(f"no capture for task_id {task_id}", "not_found", 404)

        from miles.rollout.train_service.artifacts import write_artifact

        path = await asyncio.to_thread(
            write_artifact, self.artifact_root, task_id, result, request_data.get("metadata")
        )
        return JSONResponse(content={"path": path, "response_length": result["response_length"]})

    async def handle_discard(self, request: Request) -> JSONResponse:
        """Drop one captured trajectory without writing a train artifact."""
        request_data = await request.json()
        task_id = request_data.get("task_id")
        if not task_id:
            return _openai_error("task_id is required", "invalid_request_error", 400)
        existed = self.tasks.pop(task_id, None) is not None
        self.converters.pop(task_id, None)
        return JSONResponse(content={"ok": True, "discarded": existed})

    def _consume_task(self, task_id: str) -> dict | None:
        """Pop and consolidate one trajectory's capture state. Internal to
        finalize — token data leaves the proxy only as a disk artifact."""
        state = self.tasks.pop(task_id, None)
        self.converters.pop(task_id, None)
        if state is None:
            logger.warning("No TITO state found for %s", task_id)
            return None
        return state.finalize()
