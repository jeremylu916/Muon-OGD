#!/usr/bin/env bash
set -euo pipefail

# Sequential evaluation pipeline for Muon-OGD checkpoints.
# (Filename kept as requested: eval_muon_god.sh)

# Optional overrides for coding eval A/B checks:
#   NUM_TASKS=100 MAX_NEW_TOKENS=256 SPLIT=instruct nohup ./eval_muon_god.sh > logs_seq2_muon/eval_muon_god.log 2>&1 &
NUM_TASKS="${NUM_TASKS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SPLIT="${SPLIT:-complete}"

mkdir -p \
  logs_seq2_muon/stage_a/coding logs_seq2_muon/stage_a/math logs_seq2_muon/stage_a/medical \
  logs_seq2_muon/stage_b/coding logs_seq2_muon/stage_b/math logs_seq2_muon/stage_b/medical \
  logs_seq2_muon/stage_c/coding logs_seq2_muon/stage_c/math logs_seq2_muon/stage_c/medical \
  results_seq2_muon/stage_a/coding results_seq2_muon/stage_a/math results_seq2_muon/stage_a/medical \
  results_seq2_muon/stage_b/coding results_seq2_muon/stage_b/math results_seq2_muon/stage_b/medical \
  results_seq2_muon/stage_c/coding results_seq2_muon/stage_c/math results_seq2_muon/stage_c/medical

eval_stage() {
  local stage_tag="$1"
  local model_id="$2"

  echo "[${stage_tag}] Coding eval (BigCodeBench)"
  python -u eval_bigcodebench_remote.py \
    --model_id "${model_id}" \
    --num_tasks "${NUM_TASKS}" \
    --split "${SPLIT}" \
    --subset full \
    --seed 42 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --use_rest_split \
    --train_size 800 \
    --out_dir "results_seq2_muon/${stage_tag}/coding/bcb_muon_ogd_${stage_tag}" \
    --no-debug_first_sample \
    2>&1 | tee "logs_seq2_muon/${stage_tag}/coding/eval_bcb_muon_ogd_${stage_tag}.log"

  echo "[${stage_tag}] Math eval (GSM8K)"
  python -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --num_examples 500 \
    --seed 42 \
    --out_file "results_seq2_muon/${stage_tag}/math/gsm8k_muon_ogd_${stage_tag}.json" \
    2>&1 | tee "logs_seq2_muon/${stage_tag}/math/eval_gsm8k_muon_ogd_${stage_tag}.log"

  echo "[${stage_tag}] Medical eval"
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
    --out_file "results_seq2_muon/${stage_tag}/medical/medical_muon_ogd_${stage_tag}.json" \
    2>&1 | tee "logs_seq2_muon/${stage_tag}/medical/eval_medical_muon_ogd_${stage_tag}.log"
}

echo "[1/3] Evaluate Stage A Muon-OGD checkpoint"
eval_stage "stage_a" "outputs/sft_seq2_coding_qwen1.5b_instruct_muon_ogd"

echo "[2/3] Evaluate Stage B Muon-OGD checkpoint"
eval_stage "stage_b" "outputs/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding"

echo "[3/3] Evaluate Stage C Muon-OGD checkpoint"
eval_stage "stage_c" "outputs/sft_seq2_medical_qwen1.5b_instruct_muon_ogd_from_coding_math"

echo "Muon-OGD evaluation pipeline completed successfully."
