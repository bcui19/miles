"""TITO template certification: chunked assembly must equal one-shot
rendering for the full-range fixture, and a history-mutating (think-stripping)
template must be rejected."""

import _stub_heavy_deps  # noqa: F401

import pytest
from stub_stack import StubTokenizer

from miles.rollout.nanorollout.proxy.template_check import verify_tito_template


def test_consistent_template_certifies():
    verify_tito_template(StubTokenizer())


def test_think_stripping_template_rejected():
    class ThinkStrippingTokenizer(StubTokenizer):
        def _template(self, messages, add_generation_prompt, tools=None):
            stripped = []
            for i, m in enumerate(messages):
                content = m.get("content") or ""
                if m.get("role") == "assistant" and i < len(messages) - 1 and "</think>" in content:
                    m = {**m, "content": content.split("</think>")[-1].lstrip("\n")}
                stripped.append(m)
            return super()._template(stripped, add_generation_prompt, tools)

    with pytest.raises(AssertionError, match="not TITO-compatible"):
        verify_tito_template(ThinkStrippingTokenizer())
