"""End-to-end: dataset-GRPO orchestrator -> stub NanoRollout -> real TITO
proxy (capture + finalize + artifacts) -> real Miles train buffer -> fake
train loop. Two training steps, two groups per step, sync (wait=weights)."""

import json
from argparse import Namespace

import pytest
from conftest import FakeTrainLoop
from orchestrator.dataset_grpo import DatasetGRPOConfig, run_orchestrator
from orchestrator.runtime import MilesTrainer, RolloutClient, RolloutConfig
from stub_stack import StubTokenizer

from miles.rollout.train_service.artifacts import write_artifact

N_CONTENT = len(StubTokenizer().encode("ok"))  # trainable tokens per stub assistant turn
STEPS = 2
ROLLOUT_BATCH_SIZE = 2  # same value the buffer runs with
GROUP_SIZE = 4


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset.jsonl"
    rows = [{"prompt": f"solve problem {i}", "metadata": {"instance_id": f"dp-{i}"}} for i in range(3)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


async def test_sync_grpo_two_steps(stack, dataset):
    config = DatasetGRPOConfig(
        dataset_path=dataset,
        steps=STEPS,
        rollout_batch_size=ROLLOUT_BATCH_SIZE,
        group_size=GROUP_SIZE,
        episode_concurrency=4,
        wait="weights",
        run_id="e2e",
    )
    trainer = MilesTrainer(stack.buffer_url)
    rollout = RolloutClient(
        RolloutConfig(nanorollout_url=stack.nanorollout_url, inference_url=stack.inference_url, model_name="stub")
    )

    train_loop = FakeTrainLoop(stack.buffer_args, steps=STEPS)
    train_loop.start()
    try:
        stats = await run_orchestrator(config, trainer, rollout)
    finally:
        await trainer.aclose()
        await rollout.aclose()
    train_loop.join()

    # orchestrator-side accounting
    assert stats == {
        "steps": STEPS,
        "groups_submitted": STEPS * ROLLOUT_BATCH_SIZE,
        "groups_discarded": 0,
        "episodes": STEPS * ROLLOUT_BATCH_SIZE * GROUP_SIZE,
        "episodes_dropped": 0,
    }

    # trainer-side: each batch holds whole groups with trainable samples
    assert len(train_loop.batches) == STEPS
    for output in train_loop.batches:
        assert len(output.samples) == ROLLOUT_BATCH_SIZE
        for group in output.samples:
            assert len(group) == GROUP_SIZE
            assert {s.group_index for s in group} == {group[0].group_index}
            for sample in group:
                assert sample.response_length > 0
                assert sum(sample.loss_mask) == N_CONTENT  # only the "ok" content tokens train
                assert len(sample.tokens) > sample.response_length
                assert sample.reward in (0.0, 1.0)

    # advantage math: task ids alternate reward 1/0 within each group ->
    # centered advantages +-0.5, equal within episode (single-sample episodes)
    for raw, advantages in train_loop.advantages:
        assert sorted(set(raw)) == [0.0, 1.0]
        assert sorted(set(round(a, 6) for a in advantages)) == [-0.5, 0.5]


async def test_partial_group_failure_discards_whole_group(stack, dataset, tmp_path):
    calls = []

    async def episode_fn(rollout, datapoint, task_id):
        del rollout, datapoint
        calls.append(task_id)
        if "-g0-ep1" in task_id:
            return 0.0, [], {"exit_status": "error"}
        reward = 1.0 if task_id.endswith("ep0") else 0.0
        artifact = write_artifact(
            tmp_path,
            task_id,
            {
                "tokens": [101, 202],
                "loss_mask": [1],
                "rollout_log_probs": [-0.5],
                "rollout_routed_experts": None,
                "response": "ok",
                "response_length": 1,
            },
        )
        return reward, [artifact], {"exit_status": "finished"}

    config = DatasetGRPOConfig(
        dataset_path=dataset,
        steps=1,
        rollout_batch_size=1,
        group_size=2,
        episode_concurrency=2,
        wait="weights",
        run_id="partial",
    )
    trainer = MilesTrainer(stack.buffer_url)
    rollout = RolloutClient(
        RolloutConfig(nanorollout_url=stack.nanorollout_url, inference_url=stack.inference_url, model_name="stub")
    )
    buffer_args = Namespace(**{**vars(stack.buffer_args), "rollout_batch_size": 1})
    train_loop = FakeTrainLoop(buffer_args, steps=1)
    train_loop.start()

    try:
        stats = await run_orchestrator(config, trainer, rollout, episode_fn=episode_fn)
    finally:
        await trainer.aclose()
        await rollout.aclose()
    train_loop.join()

    assert stats == {
        "steps": 1,
        "groups_submitted": 1,
        "groups_discarded": 1,
        "episodes": 4,
        "episodes_dropped": 1,
    }
    assert calls == ["partial-g0-ep0", "partial-g0-ep1", "partial-g1-ep0", "partial-g1-ep1"]
    assert len(train_loop.batches) == 1
    assert len(train_loop.batches[0].samples) == 1
    assert [sample.reward for sample in train_loop.batches[0].samples[0]] == [1.0, 0.0]
    assert sorted(round(advantage, 6) for advantage in train_loop.advantages[0][1]) == [-0.5, 0.5]


async def test_no_proxy_orchestrator_skips_finalize(stack, dataset):
    """OpenEvolve-style: no inference_url -> no finalize call, no artifacts,
    nothing submitted; NanoRollout untouched."""
    rollout = RolloutClient(RolloutConfig(nanorollout_url=stack.nanorollout_url, inference_url=None))
    episode = await rollout.run_and_finalize({"prompt": "p"}, task_id="noproxy-0")
    await rollout.aclose()
    assert episode.response["exit_status"] == "finished"
    assert episode.artifact_path is None


async def test_unknown_capture_yields_no_artifact(stack):
    """A task id the proxy never saw -> finalize 404 -> episode drops cleanly."""
    rollout = RolloutClient(RolloutConfig(nanorollout_url=stack.nanorollout_url, inference_url=stack.inference_url))
    path = await rollout.finalize("never-seen-1", metadata={})
    await rollout.aclose()
    assert path is None
