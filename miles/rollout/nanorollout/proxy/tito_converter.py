from __future__ import annotations

import json
import logging
import uuid

from .tito_state import MessageItem

logger = logging.getLogger(__name__)


_BASE = [{"role": "system", "content": "."}, {"role": "user", "content": "."}]


class ChatConverter:
    """The proxy's two message/token conversions: ``tokenize_new_messages``
    renders incoming env messages to exact template tokens;
    ``build_response`` wraps SGLang generate output into an OpenAI
    chat-completion reply."""

    def __init__(self, tokenizer, tool_call_parser: str | None = None):
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser

    def tokenize_new_messages(
        self,
        messages: list[dict],
        pre_msg_length: int,
        tools: list[dict] | None = None,
    ) -> list[MessageItem]:
        """Render the request's new messages as ONE block (the diff against a
        stub context), so template-level merging — e.g. consecutive tool
        messages into one user block — is preserved. Block boundaries are
        always preceded by a generated assistant turn, so this is exactly
        additive for the proxy flow; certify a template with
        ``template_check.verify_tito_template``. Assistant messages in the
        input (e.g. compacted context) render as history, loss-masked."""
        if pre_msg_length >= len(messages):
            return []

        def render(msgs: list[dict]) -> list[int]:
            if not msgs:
                return []
            normalized = [{**m, "content": m.get("content") or ""} for m in msgs]
            return list(
                self.tokenizer.apply_chat_template(
                    normalized, tokenize=True, return_dict=False, add_generation_prompt=False, tools=tools
                )
            )

        # continuation requests lack the real conversation head; a fixed stub
        # gives the template a stable, subtracted-out context
        base = [] if pre_msg_length == 0 else _BASE
        previous = render(base)
        current = render(base + messages[pre_msg_length:])
        if current[: len(previous)] != previous:
            raise ValueError(
                "tokenizer is not prefix-stable: render(base + tail) does not extend render(base), "
                "so the block diff would corrupt the token stream"
            )
        return [MessageItem(index=pre_msg_length, role="context", token_ids=current[len(previous) :])]

    def build_response(self, request_data: dict, generate_output: dict, text: str) -> dict:
        """Assemble the OpenAI chat-completion response; parses tool-call
        markup out of ``text`` when a parser is configured."""
        meta_info = generate_output.get("meta_info", {})
        prompt_tokens = meta_info.get("prompt_tokens", 0)
        completion_tokens = meta_info.get("completion_tokens", 0)
        finish_reason = "length" if meta_info.get("finish_reason", {}).get("type") == "length" else "stop"

        tool_calls_list = None
        tools = request_data.get("tools")
        if tools and self.tool_call_parser:
            try:
                from pydantic import TypeAdapter
                from sglang.srt.entrypoints.openai.protocol import Tool
                from sglang.srt.function_call.function_call_parser import FunctionCallParser

                parsed_tools = TypeAdapter(list[Tool]).validate_python(tools)
                parser = FunctionCallParser(parsed_tools, self.tool_call_parser)
                text, call_info_list = parser.parse_non_stream(text)
                if call_info_list:
                    finish_reason = "tool_calls"
                    tool_calls_list = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": ci.name,
                                "arguments": (
                                    ci.parameters
                                    if isinstance(ci.parameters, str)
                                    else json.dumps(ci.parameters) if ci.parameters else "{}"
                                ),
                            },
                        }
                        for ci in call_info_list
                    ]
            except Exception as exc:
                logger.warning("TITO tool-call parsing failed: %s", exc)

        message = {"role": "assistant", "content": text.strip() if text.strip() else None}
        if tool_calls_list:
            message["tool_calls"] = tool_calls_list

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": request_data.get("model", "tito-proxy"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
