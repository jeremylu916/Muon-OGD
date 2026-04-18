#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-3B-Instruct runner with O-LoRA (parameter-efficient) updates:

USER_NAME="${USER_NAME:-blu7}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/nvme/bgeo/${USER_NAME}/muon_CL}"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/3B-instruct}"

NUM_REPEATS="${NUM_REPEATS:-2}"
BASE_SEED="${BASE_SEED:-42}"
NUM_TASKS="${NUM_TASKS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SUBMIT_RETRIES="${SUBMIT_RETRIES:-10}"
SUBMIT_RETRY_DELAY_SEC="${SUBMIT_RETRY_DELAY_SEC:-60}"
BCB_NO_SUBMIT="${BCB_NO_SUBMIT:-1}"
BCB_EVAL_ARGS=()
if [[ "${BCB_NO_SUBMIT}" == "1" ]]; then
  BCB_EVAL_ARGS+=(--no_submit)
fi

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/O-LoRA}"
mkdir -p "${RUN_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"

# O-LoRA hyperparameters
OLORA_R="${OLORA_R:-16}"
OLORA_ALPHA="${OLORA_ALPHA:-32}"
OLORA_DROPOUT="${OLORA_DROPOUT:-0.05}"
OLORA_TARGET_MODULES="${OLORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

O_LORA_ARGS=(
  --use_olora
  --olora_r "${OLORA_R}"
  --olora_alpha "${OLORA_ALPHA}"
  --olora_dropout "${OLORA_DROPOUT}"
  --olora_target_modules "${OLORA_TARGET_MODULES}"
)

preflight_olora_checks() {
  echo "[preflight] checking PEFT..."
  python -c 'import peft' || { echo "Install peft"; exit 1; }
  echo "[preflight] OK"
}

eval_stage() {
  local model_id="$1"
  local res_root="$2"
  local log_root="$3"
  local stage_tag="$4"
  local bcb_tag="$5"

  mkdir -p \
    "${res_root}/${stage_tag}/coding/${bcb_tag}" \
    "${res_root}/${stage_tag}/math" \
    "${res_root}/${stage_tag}/medical"

  python -u eval_bigcodebench_remote.py \
    --model_id "${model_id}" \
    "${BCB_EVAL_ARGS[@]}" \
    --out_dir "${res_root}/${stage_tag}/coding/${bcb_tag}"

  python -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --out_file "${res_root}/${stage_tag}/math/gsm8k_${stage_tag}.json"

  python -u eval_medical.py \
    --model_id "${model_id}" \
    --out_file "${res_root}/${stage_tag}/medical/medical_${stage_tag}.json"
}

run_olora() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="olora_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}"

  echo "[O-LoRA] Stage A"
  python train_coding_bigcodebench_sft.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --output_dir "${out_root}/coding_qwen3b_olora" \
    "${O_LORA_ARGS[@]}"

  eval_stage "${out_root}/coding_qwen3b_olora" "${res_root}" "" "stage_a" "bcb"

  echo "[O-LoRA] Stage B"
  python train_math_sft.py \
    --model_id "${out_root}/coding_qwen3b_olora" \
    --output_dir "${out_root}/math_qwen3b_olora" \
    "${O_LORA_ARGS[@]}"

  eval_stage "${out_root}/math_qwen3b_olora" "${res_root}" "" "stage_b" "bcb"

  echo "[O-LoRA] Stage C"
  python train_medical_sft.py \
    --model_id "${out_root}/math_qwen3b_olora" \
    --output_dir "${out_root}/medical_qwen3b_olora" \
    "${O_LORA_ARGS[@]}"

  eval_stage "${out_root}/medical_qwen3b_olora" "${res_root}" "" "stage_c" "bcb"
}

preflight_olora_checks

for i in $(seq 1 "${NUM_REPEATS}"); do
  run_olora "${i}" $((BASE_SEED+i))
done

echo "O-LoRA 3B runs completed."