#!/usr/bin/env bash
# Service-mode compaction smoke (docs/orchestrator-design.md), container-side.
#
# Topology (all on localhost):
#   mock NanoRollout (:11000, --multi-turn)  <- toy agent, fills token budget
#   miles train.py    (train buffer :11200)  <- 2 GPUs, colocated, sync mode
#   TITO proxy        (:11100)               <- captures tokens, writes artifacts
#   orchestrator.compaction_grpo             <- the boss; exits after --steps
#
# train.py exits cleanly because MILES_NUM_ROLLOUT == orchestrator --steps.
set -euo pipefail

RUN_ID="${1:?run id required}"
REPO="${REPO:-/root/tttt}"
MILES_SRC="${REPO}/packages/miles"
SHARED="/root/shared_data/${RUN_ID}"
MODEL_DIR="/root/models/Qwen3-0.6B"

export PYTHONPATH="${MILES_SRC}:${REPO}/packages/orchestrator:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# required for --rollout-function-path add_arguments discovery (train buffer flags)
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_ADDRESS=auto
export no_proxy="127.0.0.1,localhost"

GPUS="${MILES_GPUS:-2}"
NUM_ROLLOUT="${MILES_NUM_ROLLOUT:-3}"
ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-2}"
GROUP_SIZE="${MILES_GROUP_SIZE:-3}"
TOKEN_BUDGET="${MILES_COMPACTION_TOKEN_BUDGET:-512}"
MAX_OUTPUT_TOKENS="${MILES_COMPACTION_MAX_OUTPUT_TOKENS:-6}"
MAX_RESPONSE_LEN="${MILES_ROLLOUT_MAX_RESPONSE_LEN:-64}"
ROUTER_PORT=30000
NANOROLLOUT_PORT=11000
PROXY_PORT=11100
BUFFER_PORT=11200
# upper bound on samples per training step: each segment contributes at least
# one output token, so max output tokens is also a segment upper bound.
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * GROUP_SIZE * MAX_OUTPUT_TOKENS))

mkdir -p "${SHARED}/artifacts" "${SHARED}/checkpoints" "/dev/shm/${USER:-root}/cuda_cache"
export CUDA_CACHE_PATH="/dev/shm/${USER:-root}/cuda_cache"

command -v hf >/dev/null && HF=hf || HF=huggingface-cli
[[ -d "${MODEL_DIR}" ]] || ${HF} download Qwen/Qwen3-0.6B --local-dir "${MODEL_DIR}"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 sglang 2>/dev/null || true
  pkill -f "examples.nanorollout.mock_server" 2>/dev/null || true
  pkill -f "miles.rollout.nanorollout.proxy" 2>/dev/null || true
}
trap cleanup EXIT
cleanup; sleep 2

wait_health() { # url name
  for _ in $(seq 1 "${3:-120}"); do
    curl -sf "$1" >/dev/null 2>&1 && { echo "[smoke] $2 healthy"; return 0; }
    sleep 2
  done
  echo "[smoke] $2 never became healthy at $1" >&2; return 1
}

cd "${MILES_SRC}"

python3 -m examples.nanorollout.mock_server \
  --host 127.0.0.1 --port ${NANOROLLOUT_PORT} --multi-turn --reward-mode alternating \
  >"${SHARED}/mock_nanorollout.log" 2>&1 &
wait_health "http://127.0.0.1:${NANOROLLOUT_PORT}/health" "mock nanorollout" 30

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${GPUS}" --disable-usage-stats

python3 train.py \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${SHARED}/checkpoints" \
  --save-interval 1000 \
  --prompt-data "${MILES_SRC}/examples/nanorollout/data/local_smoke.jsonl" \
  --input-key prompt \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len "${MAX_RESPONSE_LEN}" \
  --rollout-temperature 1.0 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --use-dynamic-global-batch-size \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --lr-decay-iters "${NUM_ROLLOUT}" \
  --adam-beta1 0.9 --adam-beta2 0.98 \
  --advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 \
  --use-dynamic-batch-size --max-tokens-per-gpu 32768 \
  --ci-test \
  --ci-disable-kl-checker \
  --ci-disable-logprobs-checker \
  --rollout-num-gpus-per-engine 1 \
  --sglang-router-port ${ROUTER_PORT} \
  --sglang-chunked-prefill-size 4096 --sglang-mem-fraction-static 0.7 \
  --sglang-disable-cuda-graph \
  --sglang-disable-radix-cache \
  --train-backend fsdp --attn-implementation sdpa --gradient-checkpointing \
  --update-weight-buffer-size 536870912 \
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${GPUS}" --num-gpus-per-node "${GPUS}" \
  --colocate \
  --rollout-function-path miles.rollout.train_service.train_buffer.generate_rollout \
  --custom-reward-post-process-path miles.rollout.train_service.advantages.post_process_rewards \
  --train-buffer-port ${BUFFER_PORT} \
  --train-buffer-pad-multiple "${GPUS}" \
  >"${SHARED}/miles.log" 2>&1 &
TRAIN_PID=$!

wait_health "http://127.0.0.1:${BUFFER_PORT}/health" "train buffer" 300

# miles starts the router on the node IP (setting --sglang-router-ip would
# make it assume an external router and start none)
NODE_IP=$(hostname -I | awk '{print $1}')
python3 -m miles.rollout.nanorollout.proxy \
  --sglang-url "http://${NODE_IP}:${ROUTER_PORT}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --host 127.0.0.1 --port ${PROXY_PORT} \
  --artifact-root "${SHARED}/artifacts" \
  --chat-template-path "${MILES_SRC}/examples/nanorollout/qwen3_keep_think.jinja" \
  >"${SHARED}/proxy.log" 2>&1 &
wait_health "http://127.0.0.1:${PROXY_PORT}/health" "TITO proxy" 60

python3 -m orchestrator.compaction_grpo \
  --dataset "${MILES_SRC}/examples/nanorollout/data/local_smoke.jsonl" \
  --miles-url "http://127.0.0.1:${BUFFER_PORT}" \
  --nanorollout-url "http://127.0.0.1:${NANOROLLOUT_PORT}" \
  --inference-url "http://127.0.0.1:${PROXY_PORT}/v1" \
  --steps "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --group-size "${GROUP_SIZE}" \
  --episode-concurrency 6 \
  --wait weights \
  --model-name Qwen3-0.6B \
  --run-id "${RUN_ID}" \
  --compaction-token-budget "${TOKEN_BUDGET}" \
  --compaction-max-output-tokens "${MAX_OUTPUT_TOKENS}" \
  2>&1 | tee "${SHARED}/orchestrator.log"

echo "[smoke] orchestrator finished; waiting for train.py (pid ${TRAIN_PID})"
wait "${TRAIN_PID}"
status=$?
echo "[smoke] train.py exited with ${status}"
ls -la "${SHARED}/artifacts" | head -5
exit "${status}"
