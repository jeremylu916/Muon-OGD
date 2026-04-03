#!/usr/bin/env bash
set -euo pipefail

# Quick ablation runner (reduced-cost sanity check):
# 1) Coding robustness ablation (short train + smaller eval)
# 2) Medical dataset alignment ablation (short train + smaller eval)

BASE_MODEL="Qwen/Qwen2.5-1.5B"
STAGE_B_ADAMW_MODEL="outputs/sft_seq2_math_qwen1.5b_adamw_from_coding"

CODING_ABLATION_MODEL="outputs/sft_seq2_coding_qwen1.5b_adamw_ablation_quick"
CODING_BCB_OUTDIR="results_seq2/ablation_quick/stage_a_coding_robust/bcb_complete_full"
CODING_TRAIN_LOG="logs_seq2/ablation_quick/train_coding_adamw_ablation_quick.log"
CODING_EVAL_LOG="logs_seq2/ablation_quick/eval_bcb_coding_adamw_ablation_quick.log"

MED_ALIGN_MODEL="outputs/sft_seq2_medical_qwen1.5b_adamw_from_coding_math_aligned_quick"
MED_ALIGN_TRAIN_LOG="logs_seq2/ablation_quick/train_medical_adamw_aligned_quick.log"
MED_ALIGN_EVAL_LOG="logs_seq2/ablation_quick/eval_medical_adamw_aligned_quick.log"
MED_ALIGN_OUT="results_seq2/ablation_quick/stage_c_medical_align/medical_adamw_aligned_quick.json"

CURRENT_STAGE_C_MEDICAL="results_seq2/stage_c/medical/medical_adamw_stage_c.json"
if [[ ! -f "${CURRENT_STAGE_C_MEDICAL}" ]]; then
  CURRENT_STAGE_C_MEDICAL="results_seq2/stage_c/medical/medical_adamw_final.json"
fi

mkdir -p \
  logs_seq2/ablation_quick \
  results_seq2/ablation_quick/stage_a_coding_robust \
  results_seq2/ablation_quick/stage_c_medical_align \
  outputs

echo "==== [1/4] QUICK coding robustness ablation: train Stage A model ===="
python -u train_coding_bigcodebench_sft.py \
  --model_id "${BASE_MODEL}" \
  --split complete \
  --output_dir "${CODING_ABLATION_MODEL}" \
  --num_train_examples 400 \
  --train_size 0 \
  --max_length 2048 \
  --lr 2e-6 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 2 \
  --max_steps 60 \
  --val_ratio 0 \
  --no-normalize_targets \
  --save_strategy no \
  --seed 42 \
  2>&1 | tee "${CODING_TRAIN_LOG}"

echo "==== [2/4] QUICK coding robustness ablation: eval Stage A coding ===="
python -u eval_bigcodebench_remote.py \
  --model_id "${CODING_ABLATION_MODEL}" \
  --num_tasks 120 \
  --split complete \
  --subset full \
  --seed 42 \
  --use_rest_split \
  --train_size 800 \
  --out_dir "${CODING_BCB_OUTDIR}" \
  --no-debug_first_sample \
  2>&1 | tee "${CODING_EVAL_LOG}"

echo "==== [3/4] QUICK medical dataset alignment ablation: train Stage C model ===="
python -u train_medical_sft.py \
  --model_id "${STAGE_B_ADAMW_MODEL}" \
  --output_dir "${MED_ALIGN_MODEL}" \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --train_split train \
  --num_train_examples 2000 \
  --max_length 2048 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --max_steps 60 \
  --lr 5e-6 \
  --no-answer_after_cot_only \
  --seed 42 \
  2>&1 | tee "${MED_ALIGN_TRAIN_LOG}"

echo "==== [4/4] QUICK medical dataset alignment ablation: eval Stage C medical ===="
python -u eval_medical.py \
  --model_id "${MED_ALIGN_MODEL}" \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 120 \
  --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file "${MED_ALIGN_OUT}" \
  2>&1 | tee "${MED_ALIGN_EVAL_LOG}"

echo "==== QUICK Summary ===="
python - <<'PY'
import json
import os
import re

coding_eval_log = "logs_seq2/ablation_quick/eval_bcb_coding_adamw_ablation_quick.log"
coding_pass_file = "results_seq2/ablation_quick/stage_a_coding_robust/bcb_complete_full/pass_at_k.json"
med_align_file = "results_seq2/ablation_quick/stage_c_medical_align/medical_adamw_aligned_quick.json"
current_candidates = [
    "results_seq2/stage_c/medical/medical_adamw_stage_c.json",
    "results_seq2/stage_c/medical/medical_adamw_final.json",
]

final_valid = None
if os.path.exists(coding_eval_log):
    with open(coding_eval_log, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    m = re.search(r"final_valid=(\d+)/(\d+)", txt)
    if m:
        final_valid = (int(m.group(1)), int(m.group(2)))

coding_pass = None
if os.path.exists(coding_pass_file):
    with open(coding_pass_file, "r", encoding="utf-8") as f:
        coding_pass = json.load(f).get("pass@1")

aligned_med_acc = None
if os.path.exists(med_align_file):
    with open(med_align_file, "r", encoding="utf-8") as f:
    payload = json.load(f)
    aligned_med_acc = payload.get("accuracy")
    if aligned_med_acc is None:
      aligned_med_acc = payload.get("judge_accuracy")

current_med_acc = None
current_path = None
for p in current_candidates:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
      payload = json.load(f)
      current_med_acc = payload.get("accuracy")
      if current_med_acc is None:
        current_med_acc = payload.get("judge_accuracy")
        current_path = p
        break

print("Quick coding robustness ablation:")
print(f"  final_valid: {final_valid[0]}/{final_valid[1]}" if final_valid else "  final_valid: N/A")
print(f"  pass@1: {coding_pass:.6f}" if coding_pass is not None else "  pass@1: N/A")
print()
print("Quick medical dataset alignment ablation:")
print(f"  aligned Stage C medical accuracy: {aligned_med_acc:.6f}" if aligned_med_acc is not None else "  aligned Stage C medical accuracy: N/A")
if current_med_acc is not None and aligned_med_acc is not None:
    delta = aligned_med_acc - current_med_acc
    sign = "+" if delta >= 0 else ""
    print(f"  current Stage C medical accuracy ({current_path}): {current_med_acc:.6f}")
    print(f"  delta (aligned - current): {sign}{delta:.6f}")
elif current_med_acc is not None:
    print(f"  current Stage C medical accuracy ({current_path}): {current_med_acc:.6f}")
else:
    print("  current Stage C medical accuracy: N/A (no baseline file found)")
PY

echo "Quick ablation run finished."
