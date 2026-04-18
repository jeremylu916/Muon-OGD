#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-3B-Instruct runner with Sculpting-Subspace training

USER_NAME="${USER_NAME:-blu7}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/nvme/bgeo/${USER_NAME}/muon_CL}"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/3B-instruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/subspace}"
mkdir -p "${RUN_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"

SUBSPACE_TOP_FRACTION="${SUBSPACE_TOP_FRACTION:-0.5}"
SUBSPACE_TARGET_MODULES="${SUBSPACE_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

SUBSPACE_ARGS=(
  --use_sculpt_subspace
  --subspace_top_fraction "${SUBSPACE_TOP_FRACTION}"
  --subspace_target_modules "${SUBSPACE_TARGET_MODULES}"
)

preflight_subspace_checks() {
  for f in train_coding_bigcodebench_sft.py train_math_sft.py train_medical_sft.py; do
    [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
  done
  echo "[preflight] OK"
}

eval_stage() {
  local model_id="$1"
  local res_root="$2"
  local log_root="$3"
  local stage_tag="$4"
  local bcb_tag="$5"

  mkdir -p "${res_root}/${stage_tag}/coding/${bcb_tag}" "${res_root}/${stage_tag}/math" "${res_root}/${stage_tag}/medical"

  "${PYTHON_BIN}" -u eval_bigcodebench_remote.py \
    --model_id "${model_id}" \
    "${BCB_EVAL_ARGS[@]}" \
    --out_dir "${res_root}/${stage_tag}/coding/${bcb_tag}"

  "${PYTHON_BIN}" -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --out_file "${res_root}/${stage_tag}/math/gsm8k_${stage_tag}.json"

  "${PYTHON_BIN}" -u eval_medical.py \
    --model_id "${model_id}" \
    --out_file "${res_root}/${stage_tag}/medical/medical_${stage_tag}.json"
}

run_subspace() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="subspace_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}"

  echo "[Subspace] Stage A"
  "${PYTHON_BIN}" train_coding_bigcodebench_sft.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --output_dir "${out_root}/coding_qwen3b_subspace" \
    "${SUBSPACE_ARGS[@]}"

  eval_stage "${out_root}/coding_qwen3b_subspace" "${res_root}" "" "stage_a" "bcb"

  echo "[Subspace] Stage B"
  "${PYTHON_BIN}" train_math_sft.py \
    --model_id "${out_root}/coding_qwen3b_subspace" \
    --output_dir "${out_root}/math_qwen3b_subspace" \
    "${SUBSPACE_ARGS[@]}"

  eval_stage "${out_root}/math_qwen3b_subspace" "${res_root}" "" "stage_b" "bcb"

  echo "[Subspace] Stage C"
  "${PYTHON_BIN}" train_medical_sft.py \
    --model_id "${out_root}/math_qwen3b_subspace" \
    --output_dir "${out_root}/medical_qwen3b_subspace" \
    "${SUBSPACE_ARGS[@]}"

  eval_stage "${out_root}/medical_qwen3b_subspace" "${res_root}" "" "stage_c" "bcb"
}

preflight_subspace_checks

for i in $(seq 1 "${NUM_REPEATS}"); do
  run_subspace "${i}" $((BASE_SEED+i))
done

echo "Subspace 3B runs completed."