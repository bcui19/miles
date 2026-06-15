"""Compaction episode policy for the dataset-GRPO orchestrator.

NanoRollout owns each segment's agent loop. When it reaches the configured
token budget, it performs one final compaction turn and returns. The
orchestrator finalizes that segment, then starts the next segment from the
compaction response.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass

from orchestrator.dataset_grpo import DatasetGRPOConfig, run_orchestrator
from orchestrator.runtime import MilesTrainer, RolloutClient, RolloutConfig

logger = logging.getLogger(__name__)


@dataclass
class CompactionPolicy:
    """Async callable matching dataset_grpo's EpisodeFn."""

    token_budget: int = 2048
    max_output_tokens: int = 8192

    async def __call__(self, rollout: RolloutClient, datapoint: dict, task_id: str) -> tuple[float, list[str], dict]:
        messages = None
        reward = 0.0
        paths: list[str] = []
        exit_status = "unknown"
        output_tokens = 0
        segment_idx = 0

        while output_tokens < self.max_output_tokens:
            segment_task_id = f"{task_id}-seg{segment_idx}"
            response = await rollout.run(
                datapoint,
                segment_task_id,
                rollout_tag=str(segment_idx),
                initial_messages=messages,
                extra_args={
                    "token_budget": self.token_budget,
                    "max_output_tokens": self.max_output_tokens - output_tokens,
                },
            )
            exit_status = str(response.get("exit_status", "unknown"))
            if exit_status == "error":
                logger.warning("episode %s: segment %d errored, dropping episode", task_id, segment_idx)
                return 0.0, [], {"exit_status": "error", "num_segments": len(paths)}

            path = await rollout.finalize(
                segment_task_id, metadata={"segment_index": segment_idx, "exit_status": exit_status}
            )
            if not path:
                logger.warning("episode %s: missing artifact for segment %s, dropping episode", task_id, segment_idx)
                return 0.0, [], {"exit_status": exit_status, "num_segments": len(paths)}
            paths.append(path)

            segment_reward = response.get("reward")
            if segment_reward is None:
                logger.warning("episode %s: segment %d returned no reward", task_id, segment_idx)
                return 0.0, [], {"exit_status": exit_status, "num_segments": len(paths), "error": "missing_reward"}
            reward = float(segment_reward)
            segment_output_tokens = _completion_tokens(response)
            if segment_output_tokens is None or segment_output_tokens <= 0:
                logger.warning("episode %s: segment %d returned no completion token count", task_id, segment_idx)
                return 0.0, [], {"exit_status": exit_status, "num_segments": len(paths)}
            output_tokens += segment_output_tokens
            if output_tokens >= self.max_output_tokens:
                break
            messages = _next_segment_messages(response.get("messages") or [])
            if messages is None:
                logger.warning("episode %s: segment %d returned no compaction response", task_id, segment_idx)
                return 0.0, [], {"exit_status": exit_status, "num_segments": len(paths)}
            segment_idx += 1

        return reward, paths, {"exit_status": exit_status, "num_segments": len(paths), "output_tokens": output_tokens}


def _completion_tokens(response: dict) -> int | None:
    agent_metrics = response.get("agent_metrics") or {}
    value = agent_metrics.get("completion_tokens")
    return int(value) if value is not None else None


def _next_segment_messages(messages: list[dict]) -> list[dict] | None:
    if not messages:
        return None
    system = messages[0] if messages[0].get("role") == "system" else {"role": "system", "content": ""}
    compacted = messages[-1]
    return [
        system,
        compacted,
        {"role": "user", "content": "Continue the task."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compaction GRPO orchestrator")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--miles-url", required=True)
    parser.add_argument("--nanorollout-url", required=True)
    parser.add_argument("--inference-url", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--rollout-batch-size", type=int, required=True, help="same value as Miles --rollout-batch-size"
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--episode-concurrency", type=int, default=8)
    parser.add_argument("--wait", choices=("none", "train", "weights"), default="weights")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--run-id", default="compaction-grpo")
    parser.add_argument("--compaction-token-budget", type=int, default=2048)
    parser.add_argument("--compaction-max-output-tokens", type=int, default=8192)
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
    policy = CompactionPolicy(
        token_budget=cli.compaction_token_budget,
        max_output_tokens=cli.compaction_max_output_tokens,
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
            stats = await run_orchestrator(config, trainer, rollout, episode_fn=policy)
            logger.info("orchestrator finished: %s", stats)
        finally:
            await trainer.aclose()
            await rollout.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
