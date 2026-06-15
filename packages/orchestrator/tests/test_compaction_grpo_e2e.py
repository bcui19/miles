"""End-to-end compaction: each segment runs in NanoRollout until the token
budget triggers one compaction turn. The orchestrator finalizes that segment,
starts the next segment from the compaction response, and submits all segment
artifacts as one episode sharing the final reward."""

import json

import pytest
from conftest import FakeTrainLoop
from orchestrator.compaction_grpo import CompactionPolicy
from orchestrator.dataset_grpo import DatasetGRPOConfig, run_orchestrator
from orchestrator.runtime import MilesTrainer, RolloutClient, RolloutConfig
from stub_stack import StubTokenizer

STEPS = 1
N_CONTENT = len(StubTokenizer().encode("ok"))  # trainable tokens per stub assistant turn
TURNS_PER_SEGMENT = 2  # normal turn + compaction turn
ROLLOUT_BATCH_SIZE = 2
GROUP_SIZE = 2
EXPECTED_SEGMENTS_FROM_OUTPUT_BUDGET = 3
MAX_OUTPUT_TOKENS = EXPECTED_SEGMENTS_FROM_OUTPUT_BUDGET * TURNS_PER_SEGMENT * N_CONTENT


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset.jsonl"
    rows = [{"prompt": f"long task {i}", "metadata": {"instance_id": f"cdp-{i}"}} for i in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


async def test_compaction_episode_multi_segment(stack, dataset):
    config = DatasetGRPOConfig(
        dataset_path=dataset,
        steps=STEPS,
        rollout_batch_size=ROLLOUT_BATCH_SIZE,
        group_size=GROUP_SIZE,
        episode_concurrency=4,
        wait="weights",
        run_id="c-e2e",
    )
    policy = CompactionPolicy(token_budget=1, max_output_tokens=MAX_OUTPUT_TOKENS)
    trainer = MilesTrainer(stack.buffer_url)
    rollout = RolloutClient(
        RolloutConfig(nanorollout_url=stack.nanorollout_url, inference_url=stack.inference_url, model_name="stub")
    )

    train_loop = FakeTrainLoop(stack.buffer_args, steps=STEPS)
    train_loop.start()
    try:
        stats = await run_orchestrator(config, trainer, rollout, episode_fn=policy)
    finally:
        await trainer.aclose()
        await rollout.aclose()
    train_loop.join()

    assert stats["groups_submitted"] == ROLLOUT_BATCH_SIZE
    assert stats["episodes"] == ROLLOUT_BATCH_SIZE * GROUP_SIZE
    assert stats["episodes_dropped"] == 0

    (output,) = train_loop.batches
    assert len(output.samples) == ROLLOUT_BATCH_SIZE
    for group in output.samples:
        assert len(group) == GROUP_SIZE * EXPECTED_SEGMENTS_FROM_OUTPUT_BUDGET  # every segment its own sample
        episodes = {}
        for sample in group:
            episodes.setdefault(sample.metadata["episode_index"], []).append(sample)
        assert len(episodes) == GROUP_SIZE
        for ep_samples in episodes.values():
            assert len(ep_samples) == EXPECTED_SEGMENTS_FROM_OUTPUT_BUDGET
            assert [s.metadata["segment_index"] for s in ep_samples] == [0, 1, 2]
            assert len({s.reward for s in ep_samples}) == 1  # one reward per episode
            for sample in ep_samples:
                assert sample.response_length > 0 and sum(sample.loss_mask) == TURNS_PER_SEGMENT * N_CONTENT

    # one advantage per episode, broadcast to all its segments
    ((raw, advantages),) = train_loop.advantages
    flat = [s for group in output.samples for s in group]
    by_episode = {}
    for sample, advantage in zip(flat, advantages):
        key = (sample.group_index, sample.metadata["episode_index"])
        by_episode.setdefault(key, set()).add(round(advantage, 9))
    assert all(len(v) == 1 for v in by_episode.values()), "segments of an episode must share one advantage"
    # group rewards alternate 1/0 across the two episodes -> advantages +-0.5
    assert sorted(set(round(a, 6) for a in advantages)) == [-0.5, 0.5]


async def test_compaction_segments_get_compacted_context(stack):
    """Segment k>0 must start from the previous segment's compaction response."""
    from miles.rollout.train_service.artifacts import load_artifact

    rollout = RolloutClient(
        RolloutConfig(nanorollout_url=stack.nanorollout_url, inference_url=stack.inference_url, model_name="stub")
    )
    policy = CompactionPolicy(token_budget=1, max_output_tokens=2 * TURNS_PER_SEGMENT * N_CONTENT)
    reward, paths, metadata = await policy(
        rollout, {"prompt": "ctx test", "metadata": {"instance_id": "ctx-0"}}, "ctx-ep0"
    )
    await rollout.aclose()

    assert metadata["num_segments"] == 2 and metadata["output_tokens"] == 2 * TURNS_PER_SEGMENT * N_CONTENT
    assert len(paths) == 2
    seg1 = load_artifact(paths[1])
    prompt_text = StubTokenizer().decode(seg1["tokens"][: len(seg1["tokens"]) - seg1["response_length"]])
    assert "ok" in prompt_text
    assert "Continue the task." in prompt_text
    assert "Summarize the current state for the next rollout." not in prompt_text
