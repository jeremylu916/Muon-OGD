#!/usr/bin/env bash
set -euo pipefail

# Rerun Stage C (medical train + stage_c eval only) with ablation modes.
# Usage examples:
#   bash rerun.sh --model 1p5b --run-idx 1 --stage3-mode muon_off
#   bash rerun.sh --model 1p5b --run-idx 1 --stage3-mode muon_on_low_eta
#   bash rerun.sh --model 1p5b --run-idx 1 --stage3-mode muon_on_low_eta_2048
#   bash rerun.sh --model 7b --run-idx 2 --base-seed 100
#   bash rerun.sh --model both --run-idx 1

USER_NAME="blu7"
PROJECT_ROOT="/work/nvme/bgeo/${USER_NAME}/muon_CL"

MODEL="both"             # 1p5b | 7b | both
RUN_IDX=1
BASE_SEED=42
STAGE3_MODE="muon_on_low_eta_2048"  # muon_off | muon_on_low_eta | muon_on_low_eta_2048
NUM_TASKS="${NUM_TASKS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SUBMIT_RETRIES="${SUBMIT_RETRIES:-12}"
SUBMIT_RETRY_DELAY_SEC="${SUBMIT_RETRY_DELAY_SEC:-90}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"; shift 2 ;;
    --run-idx)
      RUN_IDX="$2"; shift 2 ;;
    --base-seed)
      BASE_SEED="$2"; shift 2 ;;
    --stage3-mode)
      STAGE3_MODE="$2"; shift 2 ;;
    --help|-h)
      sed -n '1,40p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! "$MODEL" =~ ^(1p5b|7b|both)$ ]]; then
  echo "--model must be one of: 1p5b, 7b, both" >&2
  exit 1
fi

if [[ ! "$STAGE3_MODE" =~ ^(muon_off|muon_on_low_eta|muon_on_low_eta_2048)$ ]]; then
  echo "--stage3-mode must be one of: muon_off, muon_on_low_eta, muon_on_low_eta_2048" >&2
  exit 1
fi

seed=$((BASE_SEED + RUN_IDX - 1))

run_stage_c_eval() {
  local model_id="$1"
  local res_root="$2"
  local log_root="$3"
  local stage_tag="$4"

  mkdir -p \
    "${res_root}/${stage_tag}/coding/bcb_hard_muon_final" \
    "${res_root}/${stage_tag}/math" \
    "${res_root}/${stage_tag}/medical" \
    "${log_root}/${stage_tag}/coding" \
    "${log_root}/${stage_tag}/math" \
    "${log_root}/${stage_tag}/medical"

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
    --out_dir "${res_root}/${stage_tag}/coding/bcb_hard_muon_final" \
    --no-debug_first_sample \
    2>&1 | tee "${log_root}/${stage_tag}/coding/eval_bcb_stage_c.log"; then
    echo "[WARN] Coding eval failed at stage_c (bcb_hard_muon_final); continuing." \
      | tee -a "${log_root}/${stage_tag}/coding/eval_bcb_stage_c.log"
  fi

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

run_stage_c_1p5b() {
  local ROOT_DIR="${PROJECT_ROOT}/1.5B-instruct"
  local RUN_ROOT="${ROOT_DIR}/multi_runs"
  local run_tag="muon_run${RUN_IDX}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  local stage_tag="stage_c_${STAGE3_MODE}"

  mkdir -p "${out_root}" "${log_root}/${stage_tag}/train"

  echo "[1.5B][${STAGE3_MODE}][Run ${RUN_IDX}] Stage C train"

  if [[ "${STAGE3_MODE}" == "muon_off" ]]; then
    python -u train_medical_sft.py \
      --model_id "${out_root}/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding" \
      --output_dir "${out_root}/sft_seq2_medical_qwen1.5b_instruct_stagec_muon_off" \
      --max_length 2048 \
      --batch_size 1 \
      --grad_accum 8 \
      --epochs 1 \
      --lr 5e-6 \
      --max_steps 0 \
      --no-answer_after_cot_only \
      --seed "${seed}" \
      2>&1 | tee "${log_root}/${stage_tag}/train/train_medical_stagec_muon_off.log"
    local model_for_eval="${out_root}/sft_seq2_medical_qwen1.5b_instruct_stagec_muon_off"
  elif [[ "${STAGE3_MODE}" == "muon_on_low_eta" ]]; then
    python -u train_medical_muon_ogd.py \
      --model_id "${out_root}/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding" \
      --ci_model_id "${out_root}/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding" \
      --output_dir "${out_root}/sft_seq2_medical_qwen1.5b_instruct_stagec_muon_on_low_eta" \
      --max_length 1536 \
      --batch_size 1 \
      --grad_accum 8 \
      --epochs 1 \
      --lr 5e-6 \
      --max_steps 0 \
      --no-answer_after_cot_only \
      --seed "${seed}" \
      --muon_ogd \
      --muon_use_optimizer_class \
      --muon_k 3 \
      --muon_T 1 \
      --muon_eta 1e-5 \
      --muon_eta_dual 1e-5 \
      --muon_warm_start \
      --muon_layers gate_proj,up_proj,down_proj \
      --muon_momentum 0.95 \
      2>&1 | tee "${log_root}/${stage_tag}/train/train_medical_stagec_muon_on_low_eta.log"
    local model_for_eval="${out_root}/sft_seq2_medical_qwen1.5b_instruct_stagec_muon_on_low_eta"
  else
    python -u train_medical_muon_ogd.py \
      --model_id "${out_root}/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding" \
      --ci_model_id "${out_root}/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding" \
      --output_dir "${out_root}/sft_seq2_medical_qwen1.5b_instruct_stagec_muon_on_low_eta_2048" \
      --max_length 2048 \
      --batch_size 1 \
      --grad_accum 8 \
      --epochs 1 \
      --lr 5e-6 \
      --max_steps 0 \
      --no-answer_after_cot_only \
      --seed "${seed}" \
      --muon_ogd \
      --muon_use_optimizer_class \
      --muon_k 3 \
      --muon_T 1 \
      --muon_eta 1e-5 \
      --muon_eta_dual 1e-5 \
      --muon_warm_start \
      --muon_layers gate_proj,up_proj,down_proj \
      --muon_momentum 0.95 \
      2>&1 | tee "${log_root}/${stage_tag}/train/train_medical_stagec_muon_on_low_eta_2048.log"
    local model_for_eval="${out_root}/sft_seq2_medical_qwen1.5b_instruct_stagec_muon_on_low_eta_2048"
  fi

  echo "[1.5B][${STAGE3_MODE}][Run ${RUN_IDX}] Stage C eval"
  run_stage_c_eval \
    "${model_for_eval}" \
    "${res_root}" \
    "${log_root}" \
    "${stage_tag}"
}

run_stage_c_7b() {
  local ROOT_DIR="${PROJECT_ROOT}/7B-instruct"
  local RUN_ROOT="${ROOT_DIR}/multi_runs"
  local run_tag="muon_run${RUN_IDX}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  local stage_tag="stage_c_${STAGE3_MODE}"

  mkdir -p "${out_root}" "${log_root}/${stage_tag}/train"

  echo "[7B][${STAGE3_MODE}][Run ${RUN_IDX}] Stage C train"

  if [[ "${STAGE3_MODE}" == "muon_off" ]]; then
    python -u train_medical_sft.py \
      --model_id "${out_root}/sft_seq2_math_qwen7b_instruct_muon_ogd_from_coding" \
      --output_dir "${out_root}/sft_seq2_medical_qwen7b_instruct_stagec_muon_off" \
      --max_length 2048 \
      --batch_size 1 \
      --grad_accum 8 \
      --epochs 1 \
      --lr 5e-6 \
      --max_steps 0 \
      --no-answer_after_cot_only \
      --seed "${seed}" \
      2>&1 | tee "${log_root}/${stage_tag}/train/train_medical_stagec_muon_off.log"
    local model_for_eval="${out_root}/sft_seq2_medical_qwen7b_instruct_stagec_muon_off"
  elif [[ "${STAGE3_MODE}" == "muon_on_low_eta" ]]; then
    python -u train_medical_muon_ogd.py \
      --model_id "${out_root}/sft_seq2_math_qwen7b_instruct_muon_ogd_from_coding" \
      --ci_model_id "${out_root}/sft_seq2_math_qwen7b_instruct_muon_ogd_from_coding" \
      --output_dir "${out_root}/sft_seq2_medical_qwen7b_instruct_stagec_muon_on_low_eta" \
      --max_length 1536 \
      --batch_size 1 \
      --grad_accum 8 \
      --epochs 1 \
      --lr 5e-6 \
      --max_steps 0 \
      --no-answer_after_cot_only \
      --seed "${seed}" \
      --muon_ogd \
      --muon_use_optimizer_class \
      --muon_k 3 \
      --muon_T 1 \
      --muon_eta 1e-5 \
      --muon_eta_dual 1e-5 \
      --muon_warm_start \
      --muon_layers gate_proj,up_proj,down_proj \
      --muon_momentum 0.95 \
      2>&1 | tee "${log_root}/${stage_tag}/train/train_medical_stagec_muon_on_low_eta.log"
    local model_for_eval="${out_root}/sft_seq2_medical_qwen7b_instruct_stagec_muon_on_low_eta"
  else
    python -u train_medical_muon_ogd.py \
      --model_id "${out_root}/sft_seq2_math_qwen7b_instruct_muon_ogd_from_coding" \
      --ci_model_id "${out_root}/sft_seq2_math_qwen7b_instruct_muon_ogd_from_coding" \
      --output_dir "${out_root}/sft_seq2_medical_qwen7b_instruct_stagec_muon_on_low_eta_2048" \
      --max_length 2048 \
      --batch_size 1 \
      --grad_accum 8 \
      --epochs 1 \
      --lr 5e-6 \
      --max_steps 0 \
      --no-answer_after_cot_only \
      --seed "${seed}" \
      --muon_ogd \
      --muon_use_optimizer_class \
      --muon_k 3 \
      --muon_T 1 \
      --muon_eta 1e-5 \
      --muon_eta_dual 1e-5 \
      --muon_warm_start \
      --muon_layers gate_proj,up_proj,down_proj \
      --muon_momentum 0.95 \
      2>&1 | tee "${log_root}/${stage_tag}/train/train_medical_stagec_muon_on_low_eta_2048.log"
    local model_for_eval="${out_root}/sft_seq2_medical_qwen7b_instruct_stagec_muon_on_low_eta_2048"
  fi

  echo "[7B][${STAGE3_MODE}][Run ${RUN_IDX}] Stage C eval"
  run_stage_c_eval \
    "${model_for_eval}" \
    "${res_root}" \
    "${log_root}" \
    "${stage_tag}"
}

if [[ "$MODEL" == "1p5b" || "$MODEL" == "both" ]]; then
  run_stage_c_1p5b
fi

if [[ "$MODEL" == "7b" || "$MODEL" == "both" ]]; then
  run_stage_c_7b
fi

echo "Done. Reran Stage C for model=${MODEL}, run_idx=${RUN_IDX}, seed=${seed}, mode=${STAGE3_MODE}."
