# Miles + NanoRollout Minimal Integration

This example is the smallest local integration of NanoRollout/OpenHands with Miles.

Miles still owns the trainer, local SGLang rollout engines, weight updates, token/logprob capture, and train-data conversion. NanoRollout owns the external agent/environment loop through `POST /run`.

For local smoke tests, `examples.nanorollout.mock_server` implements the same `/run` shape and simulates an OpenHands worker. It calls `miles.rollout.nanorollout.proxy.TITOProxy` through an OpenAI-compatible chat endpoint, and the proxy forwards to the local Miles SGLang router. No OpenAI cloud model is used.

Run the one-GPU smoke from the host:

```bash
bash packages/miles/scripts/run_miles_nanorollout_smoke.sh
```

For a real NanoRollout service, start `nro serve` and pass:

```bash
NANOROLLOUT_URL=http://host:11000 bash packages/miles/scripts/run_miles_nanorollout_smoke.sh
```

## NVIDIA Sliding Puzzle GRPO

The `games/sliding-puzzle` NanoRollout task adapts NVIDIA NeMo-RL's pinned
sliding puzzle GRPO example to the Miles/NanoRollout/TITO stack. It keeps the
public recipe settings:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- `rollout_batch_size`: 32 prompts per step
- `group_size`: 16 generations per prompt
- rollout turns: 50
- model context / max new tokens: 1024
- puzzle max size: 5, shuffle moves: 10, max moves: 30
- validation: 256 held-out generated puzzles every 10 completed train steps
- reward: solved puzzle = 1, otherwise 0

Generate deterministic prompt rows:

```bash
PYTHONPATH=packages/miles \
python -m examples.nanorollout.sliding_puzzle_make_dataset \
  --output /tmp/sliding_puzzle_3840.jsonl \
  --length 3840
```

Generate non-overlapping validation rows:

```bash
PYTHONPATH=packages/miles \
python -m examples.nanorollout.sliding_puzzle_make_dataset \
  --output /tmp/sliding_puzzle_eval_256.jsonl \
  --length 256 \
  --start-index 3840
```

One-node 8-GPU launch, defaulting to 4 Megatron actor GPUs and 4 SGLang rollout
GPUs. The YAML contains the experiment settings; `runners/grpo_runner.py`
provides the standard online-GRPO process defaults for Ray, NanoRollout, Miles,
TITO, and the orchestrator:

```bash
WANDB_API_KEY=... \
REPO=/root/tttt \
python3 runners/grpo_runner.py \
  packages/miles/examples/nanorollout/sliding_puzzle_grpo.yaml
```

W&B is enabled by default and `WANDB_API_KEY` is required. For local debugging
only, disable it explicitly:

```bash
python3 runners/grpo_runner.py \
  packages/miles/examples/nanorollout/sliding_puzzle_grpo.yaml \
  wandb.enabled=false
```

For debugging, run only the orchestrator against an already-running NanoRollout
service, TITO proxy, and Miles train buffer:

```bash
PYTHONPATH=packages/miles:packages/orchestrator \
python -m orchestrator.dataset_grpo \
  --dataset /tmp/sliding_puzzle_3840.jsonl \
  --eval-dataset /tmp/sliding_puzzle_eval_256.jsonl \
  --eval-inference-url http://127.0.0.1:11100/v1 \
  --eval-interval 10 \
  --eval-name sliding_puzzle \
  --miles-url http://127.0.0.1:11200 \
  --nanorollout-url http://127.0.0.1:11000 \
  --inference-url http://127.0.0.1:11100/v1 \
  --steps 120 \
  --rollout-batch-size 32 \
  --group-size 16 \
  --episode-concurrency 512 \
  --wait weights \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --run-id sliding-puzzle-grpo
```

For the TITO proxy, use `--max-model-len 1024`; Qwen2.5-Instruct does not need
a reasoning parser. Validation is scheduled by the orchestrator after
`wait=weights` train steps, uses the same TITO proxy to preserve the 1024 token
context behavior, and discards eval captures instead of writing train artifacts.
