#!/usr/bin/env bash
set -euo pipefail

# Unified runner for Qwen2.5-3B-Instruct:
# - AdamW: 2 runs
# - Eval after every stage
# - Final mean/std summary

USER_NAME="blu7"
PROJECT_ROOT="/work/nvme/bgeo/${USER_NAME}/muon_CL"
ROOT_DIR="${PROJECT_ROOT}/3B-instruct"

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

RUN_ROOT="${ROOT_DIR}/multi_runs"
mkdir -p "${RUN_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"

eval_stage() {
  local model_id="$1"
  local res_root="$2"
  local log_root="$3"
  local stage_tag="$4"
  local bcb_tag="$5"

  mkdir -p \
    "${res_root}/${stage_tag}/coding/${bcb_tag}" \
    "${res_root}/${stage_tag}/math" \
    "${res_root}/${stage_tag}/medical" \
    "${log_root}/${stage_tag}/coding" \
    "${log_root}/${stage_tag}/math" \
    "${log_root}/${stage_tag}/medical"

  python -u eval_bigcodebench_remote.py \
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
    "${BCB_EVAL_ARGS[@]}" \
    --out_dir "${res_root}/${stage_tag}/coding/${bcb_tag}" \
    --no-debug_first_sample \
    2>&1 | tee "${log_root}/${stage_tag}/coding/eval_bcb_${stage_tag}.log"

  python -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --num_examples 500 \
    --seed 42 \
    --out_file "${res_root}/${stage_tag}/math/gsm8k_${stage_tag}.json" \
    2>&1 | tee "${log_root}/${stage_tag}/math/eval_gsm8k_${stage_tag}.log"

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
    --out_file "${res_root}/${stage_tag}/medical/medical_${stage_tag}.json" \
    2>&1 | tee "${log_root}/${stage_tag}/medical/eval_medical_${stage_tag}.log"
}

run_adamw() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="adamw_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}" "${log_root}/stage_a/train" "${log_root}/stage_b/train" "${log_root}/stage_c/train"

  echo "[AdamW][Run ${run_idx}] Stage A train"
  python -u train_coding_bigcodebench_sft.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_qwen3b_instruct_adamw" \
    --max_length 2048 \
    --lr 5e-6 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --seed "${seed}" \
    2>&1 | tee "${log_root}/stage_a/train/train_coding_adamw.log"

  eval_stage "${out_root}/sft_seq2_coding_qwen3b_instruct_adamw" "${res_root}" "${log_root}" "stage_a" "bcb_hard_adamw"

  python -u train_math_sft.py \
    --model_id "${out_root}/sft_seq2_coding_qwen3b_instruct_adamw" \
    --output_dir "${out_root}/sft_seq2_math_qwen3b_instruct_adamw_from_coding" \
    --num_train_examples 2000 \
    --max_length 512 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 4 \
    --lr 2e-5 \
    --max_steps 1000 \
    --seed "${seed}" \
    2>&1 | tee "${log_root}/stage_b/train/train_math_adamw.log"

  eval_stage "${out_root}/sft_seq2_math_qwen3b_instruct_adamw_from_coding" "${res_root}" "${log_root}" "stage_b" "bcb_hard_adamw_from_coding_math"

  python -u train_medical_sft.py \
    --model_id "${out_root}/sft_seq2_math_qwen3b_instruct_adamw_from_coding" \
    --output_dir "${out_root}/sft_seq2_medical_qwen3b_instruct_adamw_from_coding_math" \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --lr 5e-6 \
    --seed "${seed}" \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_adamw.log"

  eval_stage "${out_root}/sft_seq2_medical_qwen3b_instruct_adamw_from_coding_math" "${res_root}" "${log_root}" "stage_c" "bcb_hard_adamw_final"
}

for i in $(seq 1 "${NUM_REPEATS}"); do
  seed=$((BASE_SEED + i - 1))
  run_adamw "${i}" "${seed}"
done

echo "All runs complete. Results under ${RUN_ROOT}"