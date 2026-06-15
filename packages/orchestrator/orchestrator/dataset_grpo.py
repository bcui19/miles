"""Dataset-GRPO orchestrator: classic iid GRPO expressed as the top-level
program (docs/orchestrator-design.md).

Per training step it takes ``rollout_batch_size`` datapoints from a JSONL
dataset, runs ``group_size`` episodes per datapoint against NanoRollout with
asyncio-limited parallelism, and submits one group per datapoint to the Miles
train buffer. The name is deliberately Miles' own: the launcher wires one
value into both processes. All of a step's groups are submitted concurrently
— with blocking wait modes the orchestrator must keep ``rollout_batch_size``
submissions in flight, or the buffer deadlocks.

The per-trajectory behavior is a pluggable ``episode_fn``; the default
``single_run_episode`` (one run, one artifact) is the trivial policy that
replaces the old per-slot ``single_run.py``. The compaction orchestrator
plugs in its own multi-segment policy.

Dataset format: one JSON object per line, ``{"prompt": ..., "metadata": {...},
"extra_args": {...}}`` — everything beyond ``prompt`` is optional.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from orchestrator.runtime import MilesTrainer, RolloutClient, RolloutConfig

logger = logging.getLogger(__name__)

# episode_fn(rollout, datapoint, task_id) -> (reward, artifact_paths, episode_metadata)
EpisodeFn = Callable[[RolloutClient, dict, str], Awaitable[tuple[float, list[str], dict]]]


@dataclass
class DatasetGRPOConfig:
    dataset_path: str
    steps: int
    rollout_batch_size: int  # same value as Miles --rollout-batch-size (groups per training step)
    group_size: int  # episodes (trajectories) per group
    episode_concurrency: int = 8
    wait: str = "weights"  # "weights" = classic on-policy GRPO
    run_id: str = "dataset-grpo"


async def single_run_episode(rollout: RolloutClient, datapoint: dict, task_id: str) -> tuple[float, list[str], dict]:
    """Trivial episode policy: one NanoRollout run, one artifact."""
    episode = await rollout.run_and_finalize(datapoint, task_id)
    reward = episode.response.get("reward")
    if reward is None:
        return 0.0, [], {"exit_status": str(episode.response.get("exit_status", "unknown")), "error": "missing_reward"}
    paths = [episode.artifact_path] if episode.artifact_path else []
    return float(reward), paths, {"exit_status": str(episode.response.get("exit_status", "unknown"))}


def load_dataset(path: str | Path) -> list[dict]:
    datapoints = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                datapoints.append(json.loads(line))
    if not datapoints:
        raise ValueError(f"dataset {path} is empty")
    return datapoints


async def run_orchestrator(
    config: DatasetGRPOConfig,
    trainer: MilesTrainer,
    rollout: RolloutClient,
    episode_fn: EpisodeFn = single_run_episode,
) -> dict[str, Any]:
    dataset = itertools.cycle(enumerate(load_dataset(config.dataset_path)))
    group_counter = (
        itertools.count()
    )  # task ids need a never-repeating component; dp_index repeats when the dataset cycles
    semaphore = asyncio.Semaphore(config.episode_concurrency)
    stats = {"steps": 0, "groups_submitted": 0, "groups_discarded": 0, "episodes": 0, "episodes_dropped": 0}

    async def run_episode(datapoint: dict, task_id: str) -> tuple[float, list[str], dict]:
        async with semaphore:
            return await episode_fn(rollout, datapoint, task_id)

    async def run_group(step: int) -> dict:
        # draws datapoints until one yields a valid group, so a step can never
        # under-fill the Miles batch (which would deadlock blocking wait modes)
        while True:
            dp_index, datapoint = next(dataset)
            group_no = next(group_counter)
            task_ids = [f"{config.run_id}-g{group_no}-ep{k}" for k in range(config.group_size)]
            episodes = await asyncio.gather(*(run_episode(datapoint, task_id) for task_id in task_ids))
            manifest_episodes = []
            invalid_episodes = 0
            for reward, paths, metadata in episodes:
                stats["episodes"] += 1
                if not paths:
                    stats["episodes_dropped"] += 1
                    invalid_episodes += 1
                    continue
                manifest_episodes.append(trainer.episode(reward, paths, metadata=metadata))
            if invalid_episodes:
                stats["groups_discarded"] += 1
                logger.warning(
                    "%d/%d episodes for datapoint %d failed (no artifacts); discarding group, drawing a replacement",
                    invalid_episodes,
                    config.group_size,
                    dp_index,
                )
                continue
            group = trainer.group(manifest_episodes, metadata={"datapoint_index": dp_index, "step": step})
            ack = await trainer.train(group, wait=config.wait)
            stats["groups_submitted"] += 1
            return ack

    for step in range(config.steps):
        # all of the step's groups in flight together — required for blocking wait modes
        acks = await asyncio.gather(*(run_group(step) for _ in range(config.rollout_batch_size)))
        stats["steps"] += 1
        logger.info("step %d done: %s", step, [a and a.get("status", "enqueued") for a in acks])

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-GRPO orchestrator")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--miles-url", required=True, help="Miles train buffer URL")
    parser.add_argument("--nanorollout-url", required=True)
    parser.add_argument("--inference-url", required=True, help="TITO proxy base_url (.../v1)")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--rollout-batch-size", type=int, required=True, help="same value as Miles --rollout-batch-size"
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--episode-concurrency", type=int, default=8)
    parser.add_argument("--wait", choices=("none", "train", "weights"), default="weights")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--run-id", default="dataset-grpo")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = DatasetGRPOConfig(
        dataset_path=cli.dataset,
        steps=cli.steps,
        rollout_batch_size=cli.rollout_batch_size,
        group_size=cli.group_size,
        episode_concurrency=cli.episode_concurrency,
        wait=cli.wait,
        run_id=cli.run_id,
    )
    trainer = MilesTrainer(cli.miles_url)
    rollout = RolloutClient(
        RolloutConfig(
            nanorollout_url=cli.nanorollout_url,
            inference_url=cli.inference_url,
            model_name=cli.model_name,
            run_name=cli.run_id,
        )
    )

    async def _run():
        try:
            stats = await run_orchestrator(config, trainer, rollout)
            logger.info("orchestrator finished: %s", stats)
        finally:
            await trainer.aclose()
            await rollout.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
