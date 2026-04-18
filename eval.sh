#!/usr/bin/env bash
set -euo pipefail

# Evaluate Qwen 7B-Instruct on:
# - Coding (BigCodeBench remote)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/eval_qwen_7b_8b}"
LOG_ROOT="${ROOT_DIR}/logs"
RES_ROOT="${ROOT_DIR}/results"

MODEL_7B_ID="${MODEL_7B_ID:-Qwen/Qwen2.5-7B-Instruct}"

NUM_TASKS="${NUM_TASKS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SUBMIT_RETRIES="${SUBMIT_RETRIES:-12}"
SUBMIT_RETRY_DELAY_SEC="${SUBMIT_RETRY_DELAY_SEC:-90}"

mkdir -p "${LOG_ROOT}" "${RES_ROOT}"

eval_model() {
  local model_tag="$1"
  local model_id="$2"

  local model_log_root="${LOG_ROOT}/${model_tag}"
  local model_res_root="${RES_ROOT}/${model_tag}"

  mkdir -p \
    "${model_log_root}/coding" \
    "${model_res_root}/coding"

  echo "[${model_tag}] Coding eval (BigCodeBench)"
  if ! python -u eval_bigcodebench_remote.py \
    --model_id "${model_id}" \
    --num_tasks "${NUM_TASKS}" \
    --split instruct \
    --subset full \
    --seed 42 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --use_rest_split \
    --train_size 800 \
    --submit_retries "${SUBMIT_RETRIES}" \
    --submit_retry_delay_sec "${SUBMIT_RETRY_DELAY_SEC}" \
    --out_dir "${model_res_root}/coding/bcb_full" \
    --no-debug_first_sample \
    2>&1 | tee "${model_log_root}/coding/eval_bcb.log"; then
    echo "[WARN] Coding eval failed for ${model_tag}; continuing." | tee -a "${model_log_root}/coding/eval_bcb.log"
  fi

  echo "[${model_tag}] Done"
}

echo "[1/1] Evaluating ${MODEL_7B_ID} on BigCodeBench"
eval_model "qwen_7b_instruct" "${MODEL_7B_ID}"

echo "All evaluations completed. Artifacts under ${ROOT_DIR}"
