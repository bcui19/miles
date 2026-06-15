"""Shared no-GPU test stack for the orchestrator-driven rollout buffer:
a tiktoken-backed ChatML tokenizer, an in-thread uvicorn runner, and a stub
SGLang ``/generate``. Used by the miles tito_proxy tests and the orchestrator
end-to-end tests."""

from __future__ import annotations

import base64
import json
import threading
import time

import httpx
import numpy as np
import tiktoken
import uvicorn
from fastapi import FastAPI, Request

_BASE_ENC = tiktoken.get_encoding("cl100k_base")
_ENC = tiktoken.Encoding(
    name="cl100k_chatml",
    pat_str=_BASE_ENC._pat_str,
    mergeable_ranks=_BASE_ENC._mergeable_ranks,
    special_tokens={**_BASE_ENC._special_tokens, "<|im_start|>": 100300, "<|im_end|>": 100301},
)


class StubTokenizer:
    """ChatML over a real BPE vocab (tiktoken cl100k): content tokens merge
    non-trivially — unlike a char-level stub, where every render is trivially
    additive — and im_start/im_end are the atomic specials real chat templates
    rely on as merge barriers. Assistant content renders verbatim (keep-think),
    so the template is TITO-certifiable."""

    def _template(self, messages, add_generation_prompt, tools=None):
        parts = []
        for i, m in enumerate(messages):
            content = m.get("content") or ""
            if i == 0 and tools:
                content += "\n\ntools:\n" + json.dumps(tools)
            parts.append(f"<|im_start|>{m.get('role', 'user')}\n{content}<|im_end|>\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, return_dict=False
    ):
        text = self._template(messages, add_generation_prompt, tools)
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return _ENC.encode(text, allowed_special="all")

    def decode(self, token_ids):
        return _ENC.decode(token_ids)


def serve(app):
    """Run a FastAPI app on an ephemeral port in a daemon thread; returns
    (base_url, server)."""

    class _Server(uvicorn.Server):
        def install_signal_handlers(self):
            pass

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = _Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "stub server failed to start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", server


def make_stub_sglang(completion_text="ok", num_layers=2, topk=3):
    app = FastAPI()
    tok = StubTokenizer()

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        output_ids = tok.encode(completion_text)
        meta = {
            "prompt_tokens": len(body["input_ids"]),
            "completion_tokens": len(output_ids),
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[-0.5, tid] for tid in output_ids],
        }
        if body.get("return_routed_experts"):
            # incremental R3 (random_async protocol): rows for every token
            # since start_len, all but the last; row value = global position
            start = body["routed_experts_start_len"]
            rows = len(body["input_ids"]) + len(output_ids) - 1 - start
            chunk = np.repeat(np.arange(start, start + rows, dtype=np.int32), num_layers * topk)
            meta["routed_experts"] = base64.b64encode(chunk.tobytes()).decode("ascii")
        return {"text": completion_text, "meta_info": meta}

    return app


def wait_healthy(base_url):
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url.removesuffix('/v1')}/health", timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.02)
    raise AssertionError("server never became healthy")
