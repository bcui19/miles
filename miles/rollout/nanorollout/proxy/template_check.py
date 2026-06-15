"""TITO template certification: the proxy's incremental assembly must equal
one-shot rendering (no history mutation like <think> stripping, tools block
rendered once). Certify any new model/template before a run:

    python -m miles.rollout.nanorollout.proxy.template_check \
        --hf-checkpoint Qwen/Qwen3-0.6B \
        --chat-template-path examples/nanorollout/qwen3_keep_think.jinja
"""

from __future__ import annotations

import argparse

from .tito_converter import ChatConverter
from .tito_state import derive_assistant_affixes

_THINK = "<think>\nLet me inspect foo.py first.\n</think>\n\n"

FIXTURE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]

# Full-range conversation: multi-turn, thinking on every generated assistant
# turn (incl. the last prefix one — the retract-thinking failure), tool-call
# markup emitted raw by the model, and a tail of one new env response: two
# consecutive tool results plus a user message.
FIXTURE_MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the bug in foo.py."},
    {
        "role": "assistant",
        "content": _THINK
        + 'I will read both files.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "foo.py"}}\n</tool_call>\n<tool_call>\n{"name": "read_file", "arguments": {"path": "bar.py"}}\n</tool_call>',
    },
    {"role": "tool", "content": "def foo(): return 1"},
    {"role": "tool", "content": "def bar(): return 2"},
    {"role": "user", "content": "Tests are still failing."},
    {"role": "assistant", "content": _THINK + "The bug is in bar.py, checking again."},
]
FIXTURE_TAIL = [
    {"role": "tool", "content": "def bar(): return 3"},
    {"role": "tool", "content": "extra observation"},
    {"role": "user", "content": "Continue."},
]


def verify_tito_template(tokenizer, tools: list[dict] | None = None) -> None:
    """Raise AssertionError if the template is not TITO-compatible. Simulates
    the proxy's loop: env messages via the converter, assistant turns as
    GENERATED (raw encode + add_response wrapping — never re-rendered)."""
    tools = FIXTURE_TOOLS if tools is None else tools
    full = FIXTURE_MESSAGES + FIXTURE_TAIL

    converter = ChatConverter(tokenizer=tokenizer)
    assistant_prefix_ids, assistant_suffix_ids = derive_assistant_affixes(tokenizer)

    chunked: list[int] = []
    index = 0
    while index < len(full):
        boundary = index
        while boundary < len(full) and full[boundary]["role"] != "assistant":
            boundary += 1
        for item in converter.tokenize_new_messages(full[:boundary], index, tools=tools):
            chunked += list(item.token_ids)
        index = boundary
        if index < len(full):  # generated assistant turn: raw tokens, wrapped
            raw = list(tokenizer.encode(full[index]["content"], add_special_tokens=False))
            chunked += assistant_prefix_ids + raw + assistant_suffix_ids
            index += 1
    chunked += assistant_prefix_ids

    oneshot = list(
        tokenizer.apply_chat_template(full, tokenize=True, return_dict=False, add_generation_prompt=True, tools=tools)
    )

    if chunked != oneshot:
        divergence = next(
            (i for i, (a, b) in enumerate(zip(chunked, oneshot, strict=False)) if a != b),
            min(len(chunked), len(oneshot)),
        )
        context_chunked = tokenizer.decode(chunked[max(0, divergence - 20) : divergence + 20])
        context_oneshot = tokenizer.decode(oneshot[max(0, divergence - 20) : divergence + 20])
        raise AssertionError(
            f"template is not TITO-compatible: chunked ({len(chunked)}) != one-shot ({len(oneshot)}) tokens, "
            f"divergence at {divergence}.\n  chunked : ...{context_chunked!r}...\n  one-shot: ...{context_oneshot!r}..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify a tokenizer/template as TITO-compatible")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--chat-template-path", default=None)
    cli = parser.parse_args()

    from miles.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(cli.hf_checkpoint, chat_template_path=cli.chat_template_path, trust_remote_code=True)
    verify_tito_template(tokenizer)
    print(f"OK: {cli.hf_checkpoint} (template={cli.chat_template_path or 'builtin'}) is TITO-compatible")


if __name__ == "__main__":
    main()
