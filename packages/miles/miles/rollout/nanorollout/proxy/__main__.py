"""Standalone TITO proxy launcher.

Runs the proxy as its own process (docs/orchestrator-design.md: services
start before the orchestrator, outside it): a fixed-port inference +
finalize + artifact-writing service.

    python -m miles.rollout.nanorollout.proxy \
        --sglang-url http://127.0.0.1:30000 \
        --hf-checkpoint /path/to/model \
        --port 11100 \
        --artifact-root /shared/runs/run-a/rollout/tito_proxy
"""

from __future__ import annotations

import argparse
import logging
import threading
from argparse import Namespace

from .tito_server import TITOProxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone TITO proxy")
    parser.add_argument("--sglang-url", type=str, required=True, help="SGLang router base URL, e.g. http://host:port")
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="Tokenizer/model path or HF id")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host advertised in base_url")
    parser.add_argument("--artifact-root", type=str, required=True)
    parser.add_argument(
        "--chat-template-path",
        type=str,
        default=None,
        help="Override chat template (e.g. a keep-think/verbatim variant so re-rendered history matches generated tokens).",
    )
    parser.add_argument("--tool-call-parser", type=str, default=None)
    parser.add_argument("--use-rollout-routing-replay", action="store_true", default=False)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--moe-router-topk", type=int, default=None)
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from miles.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(cli.hf_checkpoint, chat_template_path=cli.chat_template_path, trust_remote_code=True)

    proxy_args = Namespace(
        sglang_tool_call_parser=cli.tool_call_parser,
        sglang_router_ip=cli.host,
        use_rollout_routing_replay=cli.use_rollout_routing_replay,
        num_layers=cli.num_layers,
        moe_router_topk=cli.moe_router_topk,
    )
    proxy = TITOProxy(
        sglang_base_url=cli.sglang_url,
        tokenizer=tokenizer,
        args=proxy_args,
        port=cli.port,
        artifact_root=cli.artifact_root,
    )
    print(f"TITO proxy serving at {proxy.base_url} (artifacts -> {cli.artifact_root})", flush=True)
    threading.Event().wait()  # uvicorn runs in a daemon thread; keep the process alive


if __name__ == "__main__":
    main()
