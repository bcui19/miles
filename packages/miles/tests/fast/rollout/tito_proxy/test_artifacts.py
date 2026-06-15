import numpy as np
import pytest

from miles.rollout.train_service.artifacts import load_artifact, validate_artifact, write_artifact


def _finalize_dict(response_length=4, with_experts=False):
    return {
        "tokens": [1, 2, 3, 10, 11, 12, 13],
        "loss_mask": [0, 1, 1, 0][:response_length],
        "rollout_log_probs": [0.0, -0.5, -0.25, 0.0][:response_length],
        "rollout_routed_experts": np.ones((6, 2, 3), dtype=np.int32) if with_experts else None,  # total_len - 1 rows
        "response": "hi",
        "response_length": response_length,
    }


def test_write_load_round_trip(tmp_path):
    data = _finalize_dict()
    path = write_artifact(tmp_path, "task-1", data, metadata={"exit_status": "finished"})
    assert path.endswith(".npz")
    validate_artifact(path)

    loaded = load_artifact(path)
    assert loaded["tokens"] == data["tokens"]
    assert loaded["loss_mask"] == data["loss_mask"]
    assert loaded["rollout_log_probs"] == pytest.approx(data["rollout_log_probs"])
    assert loaded["response_length"] == 4
    assert loaded["rollout_routed_experts"] is None
    assert loaded["response"] == "hi"
    assert loaded["sidecar_metadata"] == {"exit_status": "finished"}


def test_routed_experts_round_trip(tmp_path):
    data = _finalize_dict(with_experts=True)
    path = write_artifact(tmp_path, "task-2", data)
    loaded = load_artifact(path)
    assert loaded["rollout_routed_experts"].shape == (6, 2, 3)
    assert loaded["rollout_routed_experts"].dtype == np.int32


def test_missing_artifact_fails_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_artifact(tmp_path / "nope.npz")


def test_inconsistent_lengths_rejected_on_write(tmp_path):
    with pytest.raises(ValueError, match="inconsistent"):
        write_artifact(tmp_path, "task-3", {**_finalize_dict(), "loss_mask": [0, 1]})


def test_inconsistent_lengths_rejected_on_load(tmp_path):
    path = tmp_path / "corrupt.npz"  # written by a buggy/old producer, bypassing write_artifact
    good = _finalize_dict()
    np.savez(
        path,
        schema_version=np.int64(1),
        tokens=np.asarray(good["tokens"], np.int32),
        loss_mask=np.asarray([0, 1], np.int32),
        rollout_log_probs=np.asarray(good["rollout_log_probs"], np.float32),
        response_length=np.int64(4),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        load_artifact(path)


def test_colliding_task_ids_rejected(tmp_path):
    write_artifact(tmp_path, "task.4", _finalize_dict())
    with pytest.raises(FileExistsError):
        write_artifact(tmp_path, "task/4", _finalize_dict())  # same sanitized stem


def test_task_id_sanitized(tmp_path):
    path = write_artifact(tmp_path, "tito-r0/i:7 seg0", _finalize_dict())
    assert "/" not in path[len(str(tmp_path)) + 1 :]
    validate_artifact(path)
