from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def derive_assistant_affixes(tokenizer) -> tuple[list[int], list[int]]:
    """Derive the template's assistant turn prefix (generation prompt) and
    suffix (turn closing, e.g. ``<|im_end|>\\n``) from template arithmetic —
    no hardcoded chatml assumptions."""
    marker = "TITO_SUFFIX_MARKER"
    user = [{"role": "user", "content": "."}]
    gen = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    no_gen = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=False)
    prefix_ids = list(tokenizer.encode(gen[len(no_gen) :], add_special_tokens=False))
    full = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": marker}], tokenize=False, add_generation_prompt=False
    )
    suffix_text = full[full.index(marker, len(no_gen)) + len(marker) :]
    return prefix_ids, list(tokenizer.encode(suffix_text, add_special_tokens=False))


def hash_message(message: dict) -> str:
    """Stable hash of one chat message. TITO history is immutable: clients
    must echo recorded messages verbatim, and the proxy verifies these hashes
    on every request."""
    return hashlib.sha256(stable_message_json(message).encode("utf-8")).hexdigest()


def stable_message_json(message: dict) -> str:
    return json.dumps(message, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def copy_message(message: dict) -> dict:
    return json.loads(stable_message_json(message))


@dataclass
class MessageItem:
    index: int
    role: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    generated_token_count: int | None = None


class TaskState:
    """Per-task exact-token trajectory: env blocks and generated assistant
    turns, appended in order, never re-tokenized."""

    def __init__(
        self,
        assistant_prefix_ids: list[int] | None = None,
        assistant_suffix_ids: list[int] | None = None,
    ):
        self.messages: list[MessageItem] = []
        self.message_hashes: list[str] = []  # one per chat message, in order
        self.message_records: list[dict] = []  # canonical message snapshots for mismatch diagnostics
        self.assistant_prefix_ids = assistant_prefix_ids or []
        self.assistant_suffix_ids = assistant_suffix_ids or []
        # incremental R3 (rollout routing replay) assembly, random_async-style:
        # each generate returns rows for every token processed since
        # routed_experts_len; chunks concatenate into full-sequence coverage
        self.routed_experts_chunks: list[np.ndarray] = []
        self.routed_experts_len = 0

    def add_message_items(self, msg_items: list[MessageItem]) -> None:
        self.messages.extend(msg_items)

    def get_num_recorded_messages(self) -> int:
        return len(self.message_hashes)

    def add_response(
        self,
        token_ids: list[int],
        logprobs: list[float],
        routed_experts: np.ndarray | None = None,
        msg_hash: str | None = None,
        msg_record: dict | None = None,
    ) -> None:
        """Append one generated assistant turn, wrapped as prefix + tokens +
        suffix; logprobs are zero-padded over the affixes. ``routed_experts``
        is this generation's R3 chunk, position-indexed over the whole
        stream."""
        if routed_experts is not None:
            self.routed_experts_chunks.append(routed_experts)
            self.routed_experts_len += routed_experts.shape[0]

        generated_token_count = len(token_ids)
        suffix_ids = list(self.assistant_suffix_ids)
        for n in range(min(generated_token_count, len(suffix_ids)), 0, -1):
            if token_ids[-n:] == suffix_ids[:n]:
                suffix_ids = suffix_ids[n:]
                break

        wrapped_ids = list(self.assistant_prefix_ids) + token_ids + suffix_ids
        n_prefix = len(self.assistant_prefix_ids)
        n_suffix = len(suffix_ids)
        wrapped_logprobs = [0.0] * n_prefix + logprobs + [0.0] * n_suffix

        self.messages.append(
            MessageItem(
                index=len(self.messages),
                role="assistant",
                token_ids=wrapped_ids,
                logprobs=wrapped_logprobs,
                generated_token_count=generated_token_count,
            )
        )
        if msg_hash is not None:
            self.message_hashes.append(msg_hash)
        if msg_record is not None:
            self.message_records.append(copy_message(msg_record))

    def get_input_ids(self) -> list[int]:
        input_ids: list[int] = []
        for item in self.messages:
            input_ids.extend(item.token_ids)
        return input_ids

    def finalize(self) -> dict | None:
        """Consolidate into the train-ready dict (tokens, loss_mask over the
        response tail, logprobs, optional experts): loss is 1 only on
        generated assistant content, 0 on affixes and env tokens."""
        first_assistant_idx = None
        for i, item in enumerate(self.messages):
            if item.role == "assistant":
                first_assistant_idx = i
                break

        if first_assistant_idx is None:
            all_ids = self.get_input_ids()
            return {
                "tokens": all_ids,
                "loss_mask": [],
                "rollout_log_probs": [],
                "rollout_routed_experts": None,
                "response": "",
                "response_length": 0,
            }

        prompt_ids: list[int] = []
        for item in self.messages[:first_assistant_idx]:
            prompt_ids.extend(item.token_ids)

        response_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []

        for item in self.messages[first_assistant_idx:]:
            response_ids.extend(item.token_ids)

            if item.role == "assistant" and item.logprobs is not None:
                n_pre = len(self.assistant_prefix_ids)
                if item.generated_token_count is not None:
                    n_generated = item.generated_token_count
                    n_appended = max(0, len(item.token_ids) - n_pre - n_generated)
                    loss_mask.extend([0] * n_pre + [1] * n_generated + [0] * n_appended)
                else:
                    n_suf = len(self.assistant_suffix_ids)
                    n_content = max(0, len(item.token_ids) - n_pre - n_suf)
                    loss_mask.extend([0] * n_pre + [1] * n_content + [0] * n_suf)
                logprobs.extend(item.logprobs)
            else:
                loss_mask.extend([0] * len(item.token_ids))
                logprobs.extend([0.0] * len(item.token_ids))

        tokens = prompt_ids + response_ids

        # one row per token = its [num_layers, topk] expert ids; len(tokens)-1
        # rows total (the last token predicts nothing). Trailing tokens no
        # forward pass processed (the sampled stop token, client-appended
        # suffix) get miles' -1 no-routing-recorded sentinel; loss-masked.
        all_routed_experts = None
        if self.routed_experts_chunks:
            pad_rows = len(tokens) - 1 - self.routed_experts_len
            if not 0 <= pad_rows <= len(self.assistant_suffix_ids):
                raise ValueError(
                    f"routed experts coverage ({self.routed_experts_len}) inconsistent with "
                    f"tokens ({len(tokens)}): gap {pad_rows} exceeds suffix length "
                    f"({len(self.assistant_suffix_ids)})"
                )
            _, num_layers, topk = self.routed_experts_chunks[0].shape
            tail = np.full((pad_rows, num_layers, topk), -1, dtype=np.int32)
            all_routed_experts = np.concatenate([*self.routed_experts_chunks, tail], axis=0)
        logger.debug(
            "TITO finalize tokens=%d response_length=%d loss_tokens=%d",
            len(tokens),
            len(response_ids),
            sum(loss_mask),
        )
        return {
            "tokens": tokens,
            "loss_mask": loss_mask,
            "rollout_log_probs": logprobs,
            "rollout_routed_experts": all_routed_experts,
            "response": "",
            "response_length": len(response_ids),
        }
