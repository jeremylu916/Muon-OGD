#!/usr/bin/env bash
set -euo pipefail

# Evaluate Qwen 7B-Instruct and Qwen 8B-Instruct on:
# - Coding (BigCodeBench remote)
# - Math (GSM8K)
# - Medical (medical verifier)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/eval_qwen_7b_8b}"
LOG_ROOT="${ROOT_DIR}/logs"
RES_ROOT="${ROOT_DIR}/results"

MODEL_7B_ID="${MODEL_7B_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_8B_ID="${MODEL_8B_ID:-Qwen/Qwen2.5-8B-Instruct}"

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
    "${model_log_root}/coding" "${model_log_root}/math" "${model_log_root}/medical" \
    "${model_res_root}/coding" "${model_res_root}/math" "${model_res_root}/medical"

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

  echo "[${model_tag}] Math eval (GSM8K)"
  python -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --num_examples 500 \
    --seed 42 \
    --out_file "${model_res_root}/math/gsm8k.json" \
    2>&1 | tee "${model_log_root}/math/eval_gsm8k.log"

  echo "[${model_tag}] Medical eval"
  python -u eval_medical.py \
    --model_id "${model_id}" \
    --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
    --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
    --dataset_config default \
    --split train \
    --num_examples 500 \
    --seed 42 \
    --progress_every 20 \
    --judge_final_answer_only \
    --out_file "${model_res_root}/medical/medical.json" \
    2>&1 | tee "${model_log_root}/medical/eval_medical.log"

  echo "[${model_tag}] Done"
}

echo "[1/2] Evaluating ${MODEL_7B_ID}"
eval_model "qwen_7b_instruct" "${MODEL_7B_ID}"

echo "[2/2] Evaluating ${MODEL_8B_ID}"
eval_model "qwen_8b_instruct" "${MODEL_8B_ID}"

echo "All evaluations completed. Artifacts under ${ROOT_DIR}"
