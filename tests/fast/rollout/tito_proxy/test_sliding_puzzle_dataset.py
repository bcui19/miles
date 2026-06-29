from __future__ import annotations

import json

from examples.nanorollout.sliding_puzzle_make_dataset import main


def test_sliding_puzzle_dataset_rows(tmp_path, monkeypatch) -> None:
    """Generate default train rows with stable metadata and sequential puzzle seeds."""
    output = tmp_path / "puzzles.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sliding_puzzle_make_dataset",
            "--output",
            str(output),
            "--length",
            "2",
            "--seed",
            "100",
        ],
    )

    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["metadata"] == {
        "task": "games",
        "agent": "sliding-puzzle",
        "runner": "sliding-puzzle",
        "env_type": "local",
        "max_iterations": 50,
    }
    assert rows[0]["extra_args"]["puzzle_seed"] == 100
    assert rows[1]["extra_args"]["puzzle_seed"] == 101
    assert rows[0]["extra_args"]["size"] == 5
    assert rows[0]["extra_args"]["shuffle_moves"] == 10
    assert rows[0]["extra_args"]["max_moves"] == 30


def test_sliding_puzzle_dataset_start_index_offsets_ids_and_seeds(tmp_path, monkeypatch) -> None:
    """Build an eval shard with non-overlapping deterministic puzzle IDs.

    ``--start-index`` should offset both ``dataset_index`` and ``puzzle_seed``
    so train/eval rows do not reuse puzzles. ``--max-rollout-turns`` must also
    stay mirrored in metadata and extra_args because NanoRollout reads the latter
    while dashboards/review tooling inspect the former.
    """
    output = tmp_path / "puzzles_eval.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sliding_puzzle_make_dataset",
            "--output",
            str(output),
            "--length",
            "2",
            "--seed",
            "100",
            "--start-index",
            "3840",
            "--max-rollout-turns",
            "40",
        ],
    )

    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["extra_args"]["dataset_index"] == 3840
    assert rows[0]["extra_args"]["puzzle_seed"] == 3940
    assert rows[0]["extra_args"]["max_rollout_turns"] == 40
    assert rows[0]["metadata"]["max_iterations"] == 40
    assert rows[1]["extra_args"]["dataset_index"] == 3841
    assert rows[1]["extra_args"]["puzzle_seed"] == 3941
