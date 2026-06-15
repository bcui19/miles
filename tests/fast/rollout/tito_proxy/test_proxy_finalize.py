"""End-to-end proxy test without GPUs: a stub tokenizer and a stub SGLang
``/generate`` server, the real TITOProxy in between. Agent traffic goes
through ``/v1/chat/completions``; ``/v1/finalize`` consolidates and writes
the artifact, which must round-trip through the artifact loader."""

from argparse import Namespace

import httpx
import pytest
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


def _chat(proxy, task_id, messages):
    response = httpx.post(
        f"{proxy.base_url}/chat/completions",
        json={"messages": messages, "max_tokens": 32},
        headers={"Authorization": f"Bearer {task_id}"},
        timeout=30.0,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _finalize(proxy, task_id, metadata=None):
    return httpx.post(
        f"{proxy.base_url.removesuffix('/v1')}/v1/finalize",
        json={"task_id": task_id, "metadata": metadata or {}},
        timeout=30.0,
    )


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


def test_routing_replay_covers_full_sequence(tmp_path):
    """R3 chunks (one per generation, rows position-indexed by the stub)
    assemble into experts covering every token: env/affix tokens get real
    rows from the next turn's prefill; only the trailing suffix is -1."""
    import numpy as np

    sglang_url, sglang_server = _serve(_make_stub_sglang())
    args = Namespace(
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


def test_finalize_unknown_task_404(proxy_env):
    proxy, _ = proxy_env
    assert _finalize(proxy, "never-ran").status_code == 404
    health = httpx.get(f"{proxy.base_url.removesuffix('/v1')}/health", timeout=5.0).json()
    assert health["ok"] is True and health["inference_url"] == proxy.base_url
