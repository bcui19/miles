"""Episode-grouped GRPO advantages for the TITO train buffer
(docs/orchestrator-design.md: advantages are computed in Miles, from rewards).

Registered as the reward post-process hook:

    --custom-reward-post-process-path miles.rollout.train_service.advantages.post_process_rewards

Each ``episodes[]`` entry of a submitted group is one trajectory: its samples
(e.g. compaction segments) share one reward and receive one common advantage.
Advantages are centered (and optionally std-normalized) over the episodes
within each group. The optional ``tito_advantage_mode=leave_one_out`` uses a
leave-one-out baseline. Synthetic padding samples (metadata.padding) are
excluded and get advantage 0.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from miles.utils.metric_utils import compute_statistics

if TYPE_CHECKING:
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _stats(values: list[float]) -> dict[str, float]:
    return compute_statistics(values) | {"std": float(np.std(values))}


def _group_advantages(rewards: list[float], mode: str) -> list[float]:
    if len(rewards) == 1:
        return [0.0]
    if mode == "leave_one_out":
        if len(rewards) == 2:
            # Degenerate pair groups have only one held-out reward per sample,
            # so the leave-one-out variance estimate is undefined.
            return [0.0, 0.0]
        reward_sum = sum(rewards)
        return [reward - (reward_sum - reward) / (len(rewards) - 1) for reward in rewards]
    if mode != "grpo":
        raise ValueError(f"Unknown advantage mode: {mode!r}")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _normalize_group_advantages(rewards: list[float], advantages: list[float], mode: str, eps: float) -> list[float]:
    if len(rewards) <= 1:
        return advantages
    if mode == "leave_one_out":
        normalized = []
        for i, advantage in enumerate(advantages):
            std = _sample_std([reward for j, reward in enumerate(rewards) if j != i])
            normalized.append(advantage / (std + eps) if std > 0 else advantage)
        return normalized
    std = _sample_std(rewards)
    return [advantage / (std + eps) if std > 0 else advantage for advantage in advantages]


def post_process_rewards(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    raw_rewards = [float(s.reward or 0.0) for s in samples]

    def episode_key(sample: Sample) -> tuple | None:
        metadata = sample.metadata or {}
        if metadata.get("padding"):
            return None
        return (sample.group_index, metadata.get("episode_index", 0))

    episode_reward: dict[tuple, float] = {}
    group_episodes: dict[Any, list[tuple]] = {}
    for sample, reward in zip(samples, raw_rewards, strict=True):
        key = episode_key(sample)
        if key is None:
            continue
        if key not in episode_reward:
            group_episodes.setdefault(sample.group_index, []).append(key)
        episode_reward[key] = reward

    advantage_mode = getattr(args, "tito_advantage_mode", "grpo")
    std_normalize = getattr(args, "grpo_std_normalization", True)
    eps = 1e-6
    episode_advantage: dict[tuple, float] = {}
    for episode_keys in group_episodes.values():
        rewards = [episode_reward[k] for k in episode_keys]
        centered = _group_advantages(rewards, advantage_mode)
        if std_normalize and len(rewards) > 1:
            centered = _normalize_group_advantages(rewards, centered, advantage_mode, eps)
        for key, advantage in zip(episode_keys, centered, strict=True):
            episode_advantage[key] = advantage

    keys = [episode_key(s) for s in samples]
    advantages = [episode_advantage.get(key, 0.0) if key is not None else 0.0 for key in keys]
    if episode_reward:
        # per-batch distributions; a bad buffer slice surfaces here
        logger.info(
            "TITO advantages: episodes=%d reward %s advantage %s",
            len(episode_reward),
            _stats(list(episode_reward.values())),
            _stats(list(episode_advantage.values())),
        )
    return raw_rewards, advantages
