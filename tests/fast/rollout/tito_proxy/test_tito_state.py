from __future__ import annotations

import json

from miles.rollout.nanorollout.proxy.tito_state import hash_message, stable_message_json


def test_assistant_reasoning_content_split_does_not_hash_like_verbatim_think_content() -> None:
    # SGLang may return parsed reasoning separately from visible content. The
    # hash must preserve that structure rather than normalizing it back into a
    # verbatim ``<think>`` prefix, otherwise TITO could accept a mutated history.
    recorded = {
        "role": "assistant",
        "content": "<think>\nreasoning\n</think>\n\n<function=search>\n<parameter=path>/testbed</parameter>\n</function>",
    }
    current = {
        "role": "assistant",
        "reasoning_content": "\nreasoning\n",
        "content": "\n\n<function=search>\n<parameter=path>/testbed</parameter>\n</function>",
    }

    assert stable_message_json(current) != stable_message_json(recorded)
    assert json.loads(stable_message_json(current)) == current
    assert hash_message(current) != hash_message(recorded)


def test_assistant_reasoning_content_still_detects_real_mutation() -> None:
    # This is the paired negative case: once reasoning is represented as its own
    # field, changing that field still has to change the message hash.
    recorded = {
        "role": "assistant",
        "content": "<think>\nreasoning\n</think>\n\n<function=search>\n<parameter=path>/testbed</parameter>\n</function>",
    }
    current = {
        "role": "assistant",
        "reasoning_content": "\nchanged\n",
        "content": "\n\n<function=search>\n<parameter=path>/testbed</parameter>\n</function>",
    }

    assert hash_message(current) != hash_message(recorded)
