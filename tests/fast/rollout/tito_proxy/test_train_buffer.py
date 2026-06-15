"""TITO train buffer tests: FIFO group buffering, wait-mode acks via the
lifecycle hooks, artifact loading into Samples, padding, and advantages —
no GPU, no ray; a fake trainer drives pop/training_started/weights_synced."""

import socket
import threading
import time
from argparse import Namespace

import _stub_heavy_deps  # noqa: F401  (must precede miles.rollout imports on dev machines)
import httpx
import pytest

from miles.rollout.train_service import train_buffer
from miles.rollout.train_service.advantages import post_process_rewards
from miles.rollout.train_service.artifacts import write_artifact
from miles.utils.types import Sample


def _free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


PORT = _free_port()


def _args(**overrides):
    base = dict(
        rollout_batch_size=2,
        train_buffer_port=PORT,
        train_buffer_host="127.0.0.1",
        train_buffer_pad_multiple=1,
        grpo_std_normalization=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _artifact(tmp_path, task_id, response_length=3, reward_logprob=-0.5):
    data = {
        "tokens": list(range(5 + response_length)),
        "loss_mask": [1] * response_length,
        "rollout_log_probs": [reward_logprob] * response_length,
        "rollout_routed_experts": None,
        "response": "r",
        "response_length": response_length,
    }
    return write_artifact(tmp_path, task_id, data)


def _manifest(paths_by_episode, rewards, metadata=None):
    return {
        "metadata": metadata or {},
        "episodes": [
            {"reward": reward, "artifacts": [{"path": p} for p in paths]}
            for paths, reward in zip(paths_by_episode, rewards, strict=True)
        ],
    }


def _submit(manifest, wait="none"):
    return httpx.post(
        f"http://127.0.0.1:{PORT}/v1/train",
        json={"wait": wait, "group": manifest},
        timeout=None,
    )


def _buffer_groups():
    return httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5.0).json()["buffer_groups"]


@pytest.fixture(scope="module", autouse=True)
def server():
    train_buffer._get_server(_args())
    yield


def test_none_mode_round_trip(tmp_path):
    # group 0: two episodes, the first with two segments (compaction-style)
    g0 = _manifest(
        [
            [_artifact(tmp_path, "g0-e0-s0"), _artifact(tmp_path, "g0-e0-s1", response_length=2)],
            [_artifact(tmp_path, "g0-e1-s0")],
        ],
        rewards=[1.0, 0.0],
        metadata={"datapoint_id": "17"},
    )
    g1 = _manifest(
        [[_artifact(tmp_path, "g1-e0-s0")], [_artifact(tmp_path, "g1-e1-s0", response_length=0)]], rewards=[0.5, 0.5]
    )

    assert _submit(g0).json()["ok"] is True
    assert _submit(g1).json()["ok"] is True

    output = train_buffer.generate_rollout(_args(), rollout_id=0, data_source=None)
    assert len(output.samples) == 2
    first, second = output.samples

    assert len(first) == 3  # 2 segments + 1
    assert {s.group_index for s in first} == {first[0].group_index}
    assert [s.metadata["episode_index"] for s in first] == [0, 0, 1]
    assert [s.reward for s in first] == [1.0, 1.0, 0.0]  # episode reward stamped on all its samples
    assert first[0].metadata["group_metadata"] == {"datapoint_id": "17"}
    assert first[0].rollout_log_probs == pytest.approx([-0.5] * 3)

    assert second[0].group_index == first[0].group_index + 1
    assert second[1].response_length == 0 and second[1].remove_sample  # empty sample masked out
    assert output.metrics["tito_buffer/samples"] == 5.0


def test_padding_to_multiple(tmp_path):
    g0 = _manifest([[_artifact(tmp_path, "p0")]], rewards=[1.0])
    g1 = _manifest([[_artifact(tmp_path, "p1"), _artifact(tmp_path, "p2")]], rewards=[0.0])
    _submit(g0)
    _submit(g1)

    output = train_buffer.generate_rollout(_args(train_buffer_pad_multiple=4), rollout_id=1, data_source=None)
    flat = [s for group in output.samples for s in group]
    assert len(flat) == 4  # 3 real + 1 padding
    padding = [s for s in flat if (s.metadata or {}).get("padding")]
    assert len(padding) == 1
    assert padding[0].remove_sample and sum(padding[0].loss_mask) == 0 and padding[0].response_length == 1
    assert output.metrics["tito_buffer/padding"] == 1.0


def test_generate_rollout_padding_excluded_from_advantages(tmp_path):
    g0 = _manifest([[_artifact(tmp_path, "pa0")], [_artifact(tmp_path, "pa1")]], rewards=[1.0, 0.0])
    g1 = _manifest([[_artifact(tmp_path, "pa2")]], rewards=[0.5])
    _submit(g0)
    _submit(g1)

    args = _args(train_buffer_pad_multiple=4, grpo_std_normalization=False)
    output = train_buffer.generate_rollout(args, rollout_id=2, data_source=None)
    flat = [sample for group in output.samples for sample in group]

    padding = [sample for sample in flat if (sample.metadata or {}).get("padding")]
    real = [sample for sample in flat if not (sample.metadata or {}).get("padding")]
    assert len(real) == 3
    assert len(padding) == 1
    real_group_ids = {sample.group_index for sample in real}
    assert len(real_group_ids) == 2
    assert -1 not in real_group_ids
    assert {sample.group_index for sample in padding} == {-1}
    assert all(sample.remove_sample and sum(sample.loss_mask) == 0 for sample in padding)

    raw, adv = post_process_rewards(args, flat)
    assert raw == [1.0, 0.0, 0.5, 0.0]
    assert adv[:2] == pytest.approx([0.5, -0.5])
    assert adv[2:] == [0.0, 0.0]


def test_wait_modes_ack_on_lifecycle_hooks(tmp_path):
    results = {}

    def submit_blocking(name, manifest, wait):
        results[name] = _submit(manifest, wait=wait).json()

    threads = [
        threading.Thread(
            target=submit_blocking,
            args=(f"train-{i}", _manifest([[_artifact(tmp_path, f"w{i}")]], rewards=[1.0]), "train"),
            daemon=True,
        )
        for i in range(2)
    ]
    weights_thread = threading.Thread(
        target=submit_blocking,
        args=("weights-0", _manifest([[_artifact(tmp_path, "w-weights")]], rewards=[1.0]), "weights"),
        daemon=True,
    )
    for t in threads:
        t.start()
    deadline = time.time() + 5
    while _buffer_groups() < 2 and time.time() < deadline:
        time.sleep(0.02)
    weights_thread.start()
    while _buffer_groups() < 3 and time.time() < deadline:
        time.sleep(0.02)

    # batch of 2 pops the two wait="train" groups (FIFO); the weights group stays queued
    train_buffer.generate_rollout(_args(), rollout_id=10, data_source=None)
    assert not results, "no acks before training_started"
    train_buffer.training_started(10)
    for t in threads:
        t.join(timeout=5)
    assert results["train-0"]["status"] == "training" and results["train-1"]["status"] == "training"
    assert "weights-0" not in results, "queued group must not be woken by an earlier batch's lifecycle"

    # next batch consumes the weights group; ack only on weights_synced
    _submit(_manifest([[_artifact(tmp_path, "w-fill")]], rewards=[0.0]))
    train_buffer.generate_rollout(_args(), rollout_id=11, data_source=None)
    train_buffer.training_started(11)
    time.sleep(0.1)
    assert "weights-0" not in results
    train_buffer.weights_synced(11)
    weights_thread.join(timeout=5)
    assert results["weights-0"]["status"] == "trained" and results["weights-0"]["rollout_id"] == 11


def test_invalid_manifests_rejected(tmp_path):
    assert _submit({"episodes": []}).status_code == 400
    assert (
        _submit({"episodes": [{"reward": 1.0, "artifacts": [{"path": str(tmp_path / "missing.npz")}]}]}).status_code
        == 400
    )
    assert _submit({"episodes": [{"artifacts": [{"path": _artifact(tmp_path, "no-reward")}]}]}).status_code == 400
    bad_wait = httpx.post(f"http://127.0.0.1:{PORT}/v1/train", json={"wait": "later", "group": {}}, timeout=5.0)
    assert bad_wait.status_code == 400
    assert _buffer_groups() == 0, "rejected manifests must not be buffered"


def test_eval_disabled():
    with pytest.raises(RuntimeError, match="eval"):
        train_buffer.generate_rollout(_args(), rollout_id=0, data_source=None, evaluation=True)


def _sample(group_index, episode_index, reward, padding=False):
    return Sample(
        group_index=group_index,
        reward=reward,
        metadata={"episode_index": episode_index, **({"padding": True} if padding else {})},
    )


def test_episode_grouped_advantages():
    samples = [
        _sample(0, 0, 1.0),
        _sample(0, 0, 1.0),  # same episode: same advantage
        _sample(0, 1, 0.0),
        _sample(1, 0, 0.5),  # singleton group: centered to 0
        _sample(-1, 0, 0.0, padding=True),
    ]
    raw, adv = post_process_rewards(Namespace(grpo_std_normalization=False), samples)
    assert raw == [1.0, 1.0, 0.0, 0.5, 0.0]
    assert adv[0] == adv[1] == pytest.approx(0.5)
    assert adv[2] == pytest.approx(-0.5)
    assert adv[3] == pytest.approx(0.0)
    assert adv[4] == 0.0

    _, adv_norm = post_process_rewards(Namespace(grpo_std_normalization=True), samples)
    assert adv_norm[0] == pytest.approx(0.5 / (0.7071067 + 1e-6), rel=1e-3)
