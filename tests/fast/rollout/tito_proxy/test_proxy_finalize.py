"""End-to-end proxy test without GPUs: a stub tokenizer and a stub SGLang
``/generate`` server, the real TITOProxy in between. Agent traffic goes
through ``/v1/chat/completions``; ``/v1/finalize`` consolidates and writes
the artifact, which must round-trip through the artifact loader."""

from argparse import Namespace

import httpx
import pytest
from fastapi import FastAPI, Request
from stub_stack import StubTokenizer
from stub_stack import make_stub_sglang as _make_stub_sglang
from stub_stack import serve as _serve
from stub_stack import wait_healthy as _wait_healthy

from miles.rollout.nanorollout.proxy.tito_converter import ChatConverter
from miles.rollout.nanorollout.proxy.tito_server import TITOProxy
from miles.rollout.nanorollout.proxy.tito_state import MessageItem, TaskState, derive_assistant_affixes
from miles.rollout.train_service.artifacts import load_artifact


@pytest.fixture(scope="module")
def proxy_env(tmp_path_factory):
    artifact_root = tmp_path_factory.mktemp("artifacts")
    sglang_url, sglang_server = _serve(_make_stub_sglang())
    args = Namespace(
        sglang_reasoning_parser=None,
        sglang_tool_call_parser=None,
        sglang_router_ip="127.0.0.1",
        use_rollout_routing_replay=False,
    )
    proxy = TITOProxy(
        sglang_base_url=sglang_url,
        tokenizer=StubTokenizer(),
        args=args,
        artifact_root=str(artifact_root),
    )
    _wait_healthy(proxy.base_url)
    yield proxy, str(artifact_root)
    sglang_server.should_exit = True


def _chat_response(proxy, task_id, messages):
    response = httpx.post(
        f"{proxy.base_url}/chat/completions",
        json={"messages": messages, "max_tokens": 32},
        headers={"Authorization": f"Bearer {task_id}"},
        timeout=30.0,
    )
    return response


def _chat(proxy, task_id, messages):
    response = _chat_response(proxy, task_id, messages)
    assert response.status_code == 200, response.text
    return response.json()


def _finalize(proxy, task_id, metadata=None):
    return httpx.post(
        f"{proxy.base_url.removesuffix('/v1')}/v1/finalize",
        json={"task_id": task_id, "metadata": metadata or {}},
        timeout=30.0,
    )


def _discard(proxy, task_id):
    return httpx.post(
        f"{proxy.base_url.removesuffix('/v1')}/v1/discard",
        json={"task_id": task_id},
        timeout=30.0,
    )


def test_proxy_enforces_context_window_without_mutating_rejected_requests(tmp_path):
    requests = []
    app = FastAPI()
    tok = StubTokenizer()

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        requests.append(body)
        if body["sampling_params"]["max_new_tokens"] == 3:
            return {"error": "synthetic generation failure"}
        output_ids = tok.encode("ok")
        return {
            "text": "ok",
            "meta_info": {
                "prompt_tokens": len(body["input_ids"]),
                "completion_tokens": len(output_ids),
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, tid] for tid in output_ids],
            },
        }

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the thing"}]
    input_len = len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    sglang_url, sglang_server = _serve(app)

    def make_proxy(max_model_len: int, artifact_name: str):
        proxy = TITOProxy(
            sglang_base_url=sglang_url,
            tokenizer=tok,
            args=Namespace(
                sglang_tool_call_parser=None,
                sglang_reasoning_parser=None,
                sglang_router_ip="127.0.0.1",
                max_model_len=max_model_len,
                use_rollout_routing_replay=False,
            ),
            artifact_root=str(tmp_path / artifact_name),
        )
        _wait_healthy(proxy.base_url)
        return proxy

    clipped_proxy = make_proxy(input_len + 7, "clip")
    _chat(clipped_proxy, "task-context-clip", messages)
    assert requests[0]["sampling_params"]["max_new_tokens"] == 7

    exhausted_proxy = make_proxy(input_len - 1, "exhausted")
    request_count = len(requests)

    first = _chat_response(exhausted_proxy, "task-context-exhausted", messages)
    assert first.status_code == 400
    assert first.json()["error"]["type"] == "context_length_exceeded"

    modified_messages = [messages[0], {"role": "user", "content": "do the other thing"}]
    second = _chat_response(exhausted_proxy, "task-context-exhausted", modified_messages)
    assert second.status_code == 400
    assert second.json()["error"]["type"] == "context_length_exceeded"
    assert len(requests) == request_count

    sglang_error_proxy = make_proxy(input_len + 3, "sglang-error")
    failed = _chat_response(sglang_error_proxy, "task-sglang-error", messages)
    assert failed.status_code == 400
    assert failed.json()["error"]["type"] == "invalid_request_error"
    assert _finalize(sglang_error_proxy, "task-sglang-error").status_code == 404
    sglang_server.should_exit = True


def test_proxy_returns_exact_openai_usage_counts(tmp_path):
    requests = []
    app = FastAPI()
    tok = StubTokenizer()

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        requests.append(body)
        output_ids = tok.encode("ok")
        return {
            "text": "ok",
            "meta_info": {
                "prompt_tokens": len(body["input_ids"]),
                "completion_tokens": len(output_ids),
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, tid] for tid in output_ids],
            },
        }

    sglang_url, sglang_server = _serve(app)
    proxy = TITOProxy(
        sglang_base_url=sglang_url,
        tokenizer=tok,
        args=Namespace(
            sglang_tool_call_parser=None,
            sglang_reasoning_parser=None,
            sglang_router_ip="127.0.0.1",
            max_model_len=None,
            use_rollout_routing_replay=False,
        ),
        artifact_root=str(tmp_path),
    )
    _wait_healthy(proxy.base_url)

    task_id = "task-response-clip"
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "turn one"}]
    first_response = _chat(proxy, task_id, messages)
    first = first_response["choices"][0]["message"]["content"]
    messages += [{"role": "assistant", "content": first}, {"role": "user", "content": "turn two"}]
    second_response = _chat(proxy, task_id, messages)

    first_usage = first_response["usage"]
    assert first_usage["prompt_tokens"] == len(requests[0]["input_ids"])
    assert first_usage["completion_tokens"] == len(tok.encode("ok"))
    assert first_usage["total_tokens"] == first_usage["prompt_tokens"] + first_usage["completion_tokens"]
    assert set(first_usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}

    second_usage = second_response["usage"]
    assert requests[1]["sampling_params"]["max_new_tokens"] == 32
    assert second_usage["prompt_tokens"] == len(requests[1]["input_ids"])
    assert second_usage["completion_tokens"] == len(tok.encode("ok"))
    assert second_usage["total_tokens"] == second_usage["prompt_tokens"] + second_usage["completion_tokens"]
    assert set(second_usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    sglang_server.should_exit = True


def test_single_turn_finalize_round_trip(proxy_env):
    proxy, _ = proxy_env
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
    ]
    completion = _chat(proxy, "task-single", messages)
    assert completion["choices"][0]["message"]["content"] == "ok"

    finalize = _finalize(proxy, "task-single", metadata={"exit_status": "finished"})
    assert finalize.status_code == 200, finalize.text
    path = finalize.json()["path"]

    data = load_artifact(path)
    # response = assistant prefix + "ok" + suffix; only the content tokens train
    tok = StubTokenizer()
    prefix_ids, suffix_ids = derive_assistant_affixes(tok)
    n_content = len(tok.encode("ok"))
    assert data["response_length"] == len(prefix_ids) + n_content + len(suffix_ids)
    assert len(data["loss_mask"]) == data["response_length"]
    assert sum(data["loss_mask"]) == n_content
    assert len(data["tokens"]) > data["response_length"]  # prompt included
    content_logprobs = [lp for lp, m in zip(data["rollout_log_probs"], data["loss_mask"], strict=True) if m]
    assert content_logprobs == pytest.approx([-0.5] * n_content)
    assert data["sidecar_metadata"] == {"exit_status": "finished"}


def test_multi_turn_accumulates_in_one_artifact(proxy_env):
    proxy, _ = proxy_env
    task_id = "task-multi"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn one"},
    ]
    first = _chat(proxy, task_id, messages)["choices"][0]["message"]["content"]
    messages += [
        {"role": "assistant", "content": first},
        {"role": "user", "content": "turn two"},
    ]
    _chat(proxy, task_id, messages)

    finalize = _finalize(proxy, task_id)
    assert finalize.status_code == 200, finalize.text
    data = load_artifact(finalize.json()["path"])
    n_content = len(StubTokenizer().encode("ok"))
    assert sum(data["loss_mask"]) == 2 * n_content  # two assistant turns x "ok"
    assert len(data["loss_mask"]) == data["response_length"]


def test_add_response_strips_generated_suffix_stop_token():
    state = TaskState(assistant_prefix_ids=[10, 11], assistant_suffix_ids=[99, 100])
    state.add_message_items([MessageItem(index=0, role="context", token_ids=[7, 8])])

    state.add_response([1, 2, 99], [-0.1, -0.2, -0.3])
    data = state.finalize()

    assert data["tokens"] == [7, 8, 10, 11, 1, 2, 99, 100]
    assert data["response_length"] == 6
    assert len(data["tokens"]) > data["response_length"]
    assert data["loss_mask"] == [0, 0, 1, 1, 1, 0]
    assert data["rollout_log_probs"] == [0.0, 0.0, -0.1, -0.2, -0.3, 0.0]


def test_build_response_uses_sglang_tool_call_parser():
    pytest.importorskip("sglang.srt.function_call.function_call_parser")
    pytest.importorskip("sglang.srt.entrypoints.openai.protocol")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_year",
                "description": "Get current year",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    converter = ChatConverter(StubTokenizer(), tool_call_parser="qwen25")

    response = converter.build_response(
        {"model": "model", "tools": tools},
        {"meta_info": {"prompt_tokens": 3, "completion_tokens": 4, "finish_reason": {"type": "stop"}}},
        'Let me check.\n<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>',
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Let me check."
    assert len(choice["message"]["tool_calls"]) == 1
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"] == {"name": "get_year", "arguments": "{}"}


def test_build_response_uses_sglang_reasoning_parser():
    pytest.importorskip("sglang.srt.parser.reasoning_parser")
    converter = ChatConverter(StubTokenizer(), reasoning_parser="qwen3")

    response = converter.build_response(
        {"model": "model"},
        {"meta_info": {"prompt_tokens": 3, "completion_tokens": 4, "finish_reason": {"type": "stop"}}},
        "<think>\nreasoning\n</think>\n\nanswer",
    )

    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "\nreasoning\n"
    assert message["content"] == "answer"


def test_routing_replay_covers_full_sequence(tmp_path):
    """R3 chunks (one per generation, rows position-indexed by the stub)
    assemble into experts covering every token: env/affix tokens get real
    rows from the next turn's prefill; only the trailing suffix is -1."""
    import numpy as np

    sglang_url, sglang_server = _serve(_make_stub_sglang())
    args = Namespace(
        sglang_reasoning_parser=None,
        sglang_tool_call_parser=None,
        sglang_router_ip="127.0.0.1",
        use_rollout_routing_replay=True,
        num_layers=2,
        moe_router_topk=3,
    )
    proxy = TITOProxy(sglang_base_url=sglang_url, tokenizer=StubTokenizer(), args=args, artifact_root=str(tmp_path))
    _wait_healthy(proxy.base_url)

    task_id = "task-r3"
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    first = _chat(proxy, task_id, messages)["choices"][0]["message"]["content"]
    messages += [{"role": "assistant", "content": first}, {"role": "user", "content": "again"}]
    _chat(proxy, task_id, messages)

    data = load_artifact(_finalize(proxy, task_id).json()["path"])
    experts = data["rollout_routed_experts"]
    total = len(data["tokens"])
    suffix_len = len(derive_assistant_affixes(StubTokenizer())[1])
    assert experts.shape == (total - 1, 2, 3)
    covered = total - 1 - suffix_len
    assert (experts[:covered, 0, 0] == np.arange(covered)).all()
    assert (experts[covered:] == -1).all()
    sglang_server.should_exit = True


def test_finalize_is_consume_once(proxy_env):
    proxy, _ = proxy_env
    _chat(proxy, "task-once", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert _finalize(proxy, "task-once").status_code == 200
    assert _finalize(proxy, "task-once").status_code == 404


def test_discard_consumes_without_artifact(proxy_env):
    proxy, _ = proxy_env
    _chat(proxy, "task-discard", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    discard = _discard(proxy, "task-discard")
    assert discard.status_code == 200, discard.text
    assert discard.json() == {"ok": True, "discarded": True}
    assert _finalize(proxy, "task-discard").status_code == 404


def test_finalize_unknown_task_404(proxy_env):
    proxy, _ = proxy_env
    assert _finalize(proxy, "never-ran").status_code == 404
    health = httpx.get(f"{proxy.base_url.removesuffix('/v1')}/health", timeout=5.0).json()
    assert health["ok"] is True and health["inference_url"] == proxy.base_url
