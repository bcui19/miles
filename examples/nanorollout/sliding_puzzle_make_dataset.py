"""Generate deterministic datapoints for the NanoRollout sliding-puzzle GRPO example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write sliding-puzzle NanoRollout JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=3840, help="120 steps x 32 prompts, matching NeMo docs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--shuffle-moves", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=30)
    parser.add_argument("--max-rollout-turns", type=int, default=50)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for idx in range(args.length):
            dataset_index = args.start_index + idx
            row = {
                "prompt": "",
                "metadata": {
                    "task": "games",
                    "agent": "sliding-puzzle",
                    "runner": "sliding-puzzle",
                    "env_type": "local",
                    "max_iterations": args.max_rollout_turns,
                },
                "extra_args": {
                    "dataset_index": dataset_index,
                    "puzzle_seed": args.seed + dataset_index,
                    "size": args.size,
                    "shuffle_moves": args.shuffle_moves,
                    "max_moves": args.max_moves,
                    "max_rollout_turns": args.max_rollout_turns,
                },
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
