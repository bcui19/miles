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
from .tito_state import TaskState, derive_assistant_affixes, hash_message

logger = logging.getLogger(__name__)


class _HistoryMutatedError(Exception):
    def __init__(self, index: int):
        self.index = index


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
        self.tasks: dict[str, TaskState] = {}
        self.converters: dict[str, ChatConverter] = {}

        if getattr(args, "use_rollout_routing_replay", False) and (
            getattr(args, "num_layers", None) is None or getattr(args, "moe_router_topk", None) is None
        ):
            raise ValueError("rollout routing replay requires --num-layers and --moe-router-topk")

        self.assistant_prefix_ids, self.assistant_suffix_ids = derive_assistant_affixes(tokenizer)
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
            return JSONResponse(content={"error": "Authorization bearer token (task id) is required"}, status_code=401)

        if api_key not in self.tasks:
            self.tasks[api_key] = TaskState(
                assistant_prefix_ids=self.assistant_prefix_ids,
                assistant_suffix_ids=self.assistant_suffix_ids,
            )
            self.converters[api_key] = ChatConverter(
                tokenizer=self.tokenizer,
                tool_call_parser=self.tool_call_parser,
            )

        state = self.tasks[api_key]
        converter = self.converters[api_key]
        pre_msg_length = state.get_num_recorded_messages()

        def _tokenize() -> list[int]:
            # TITO: generation input is the append-only token stream — new
            # (non-assistant) messages are tokenized once; generated tokens are
            # reused verbatim, never re-encoded. History is immutable: clients
            # must echo recorded messages verbatim, verified by hash.
            if len(messages) < pre_msg_length:
                raise _HistoryMutatedError(len(messages))
            for index, (recorded, msg) in enumerate(zip(state.message_hashes, messages[:pre_msg_length], strict=True)):
                if recorded != hash_message(msg):
                    raise _HistoryMutatedError(index)
            state.add_message_items(converter.tokenize_new_messages(messages, pre_msg_length, tools=tools))
            state.message_hashes.extend(hash_message(m) for m in messages[pre_msg_length:])
            return state.get_input_ids() + self.assistant_prefix_ids

        try:
            input_ids = await asyncio.to_thread(_tokenize)
        except _HistoryMutatedError as exc:
            logger.error("History mutated for %s at message index %d — TITO history is immutable.", api_key, exc.index)
            return JSONResponse(
                content={"error": f"recorded history mutated at message index {exc.index}; TITO history is immutable"},
                status_code=409,
            )
        sampling_params = {
            key: value for key, value in request_data.items() if key in ("temperature", "top_p", "top_k", "stop")
        }
        max_new_tokens = request_data.get("max_tokens", request_data.get("max_new_tokens", 4096))
        generate_payload = {
            "input_ids": input_ids,
            "sampling_params": {
                **sampling_params,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": False,
            },
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
            return JSONResponse(
                content={"error": {"message": str(error_msg), "type": "invalid_request_error"}},
                status_code=resp.status_code if resp.status_code >= 400 else 400,
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
        state.add_response(output_ids, output_logprobs, routed_experts, msg_hash=hash_message(assistant_message))
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
            return JSONResponse(content={"error": "task_id is required"}, status_code=400)

        result = self._consume_task(task_id)
        if result is None:
            return JSONResponse(content={"error": f"no capture for task_id {task_id}"}, status_code=404)

        from miles.rollout.train_service.artifacts import write_artifact

        path = await asyncio.to_thread(
            write_artifact, self.artifact_root, task_id, result, request_data.get("metadata")
        )
        return JSONResponse(content={"path": path, "response_length": result["response_length"]})

    def _consume_task(self, task_id: str) -> dict | None:
        """Pop and consolidate one trajectory's capture state. Internal to
        finalize — token data leaves the proxy only as a disk artifact."""
        state = self.tasks.pop(task_id, None)
        self.converters.pop(task_id, None)
        if state is None:
            logger.warning("No TITO state found for %s", task_id)
            return None
        return state.finalize()
