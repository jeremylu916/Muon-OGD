#!/usr/bin/env bash
set -euo pipefail

# Stage C-only ablation runner for Muon-OGD.
# Trains and evaluates multiple Stage C variants to diagnose performance drops.

STAGE_A_MUON_MODEL="${STAGE_A_MUON_MODEL:-1.5B-instruct/outputs/sft_seq2_coding_qwen1.5b_instruct_muon_ogd}"
STAGE_B_MUON_MODEL="${STAGE_B_MUON_MODEL:-1.5B-instruct/outputs/sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding}"

STAGE_C_ABLATION_DIR="results_seq2/ablation/stage_c_medical_muon"
STAGE_C_LOG_DIR="logs_seq2/ablation/stage_c_medical_muon"

CURRENT_STAGE_C_MEDICAL="results_seq2/stage_c/medical/medical_adamw_stage_c.json"
if [[ ! -f "${CURRENT_STAGE_C_MEDICAL}" ]]; then
  CURRENT_STAGE_C_MEDICAL="results_seq2/stage_c/medical/medical_adamw_final.json"
fi

mkdir -p \
  logs_seq2/ablation \
  "${STAGE_C_ABLATION_DIR}" \
  "${STAGE_C_LOG_DIR}" \
  outputs

resolve_model_dir() {
  local input_path="$1"
  local fallback_name="$2"

  if [[ -d "${input_path}" ]]; then
    echo "${input_path}"
    return 0
  fi

  if [[ -d "./1.5B-instruct/outputs/${fallback_name}" ]]; then
    echo "./1.5B-instruct/outputs/${fallback_name}"
    return 0
  fi

  local found
  found="$(find . -maxdepth 8 -type d -name "${fallback_name}" | head -n 1 || true)"
  if [[ -n "${found}" && -d "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

if ! STAGE_A_MUON_MODEL="$(resolve_model_dir "${STAGE_A_MUON_MODEL}" "sft_seq2_coding_qwen1.5b_instruct_muon_ogd")"; then
  echo "[ERROR] Stage A Muon model directory not found: ${STAGE_A_MUON_MODEL}" >&2
  echo "Set STAGE_A_MUON_MODEL to an existing local checkpoint path before running." >&2
  exit 1
fi

if ! STAGE_B_MUON_MODEL="$(resolve_model_dir "${STAGE_B_MUON_MODEL}" "sft_seq2_math_qwen1.5b_instruct_muon_ogd_from_coding")"; then
  echo "[ERROR] Stage B Muon model directory not found: ${STAGE_B_MUON_MODEL}" >&2
  echo "Set STAGE_B_MUON_MODEL to an existing local checkpoint path before running." >&2
  exit 1
fi

echo "Using Stage A Muon model: ${STAGE_A_MUON_MODEL}"
echo "Using Stage B Muon model: ${STAGE_B_MUON_MODEL}"

run_stage_c_variant() {
  local tag="$1"
  local extra_args="$2"

  local out_model="outputs/sft_seq2_medical_${tag}"
  local train_log="${STAGE_C_LOG_DIR}/train_${tag}.log"
  local eval_log="${STAGE_C_LOG_DIR}/eval_${tag}.log"
  local eval_out="${STAGE_C_ABLATION_DIR}/medical_${tag}.json"

  echo "==== [Stage C][${tag}] train ===="
  # shellcheck disable=SC2086
  python -u train_medical_muon_ogd.py \
    --model_id "${STAGE_B_MUON_MODEL}" \
    --ci_model_id "${STAGE_B_MUON_MODEL}" \
    --output_dir "${out_model}" \
    --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT\
    --dataset_config default \
    --train_split train \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --lr 5e-6 \
    --seed 42 \
    --muon_ogd \
    --muon_use_optimizer_class \
    --muon_layers gate_proj,up_proj,down_proj \
    --muon_k 3 \
    --muon_T 1 \
    --muon_eta 1e-5 \
    --muon_eta_dual 1e-5 \
    --muon_warm_start \
    --muon_momentum 0.95 \
    ${extra_args} \
    2>&1 | tee "${train_log}"

  echo "==== [Stage C][${tag}] eval ===="
  python -u eval_medical.py \
    --model_id "${out_model}" \
    --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
    --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
    --dataset_config default \
    --split train \
    --num_examples 500 \
    --seed 42 \
    --progress_every 20 \
    --judge_final_answer_only \
    --out_file "${eval_out}" \
    2>&1 | tee "${eval_log}"
}

echo "==== [1/2] Stage C Muon ablations: train+eval variants ===="
# Baseline Stage C behavior in your Muon run script.
run_stage_c_variant "muon_baseline_no_answer_after_cot" "--no-answer_after_cot_only"
# Keep only final answer span as supervision target.
run_stage_c_variant "muon_answer_only" ""
# Accumulate protected directions from Stage A + Stage B checkpoints.
run_stage_c_variant "muon_answer_only_multi_source" "--ci_model_ids ${STAGE_A_MUON_MODEL},${STAGE_B_MUON_MODEL} --ci_k_per_source 2"

echo "==== [2/2] Stage C Muon ablations complete; generating summary ===="

echo "==== Summary ===="
python - <<'PY'
import json
import os
import glob

stage_c_dir = "results_seq2/ablation/stage_c_medical_muon"

current_candidates = [
  "results_seq2/stage_c/medical/medical_muon_stage_c.json",
  "results_seq2/stage_c/medical/medical_muon_final.json",
  "results_seq2/stage_c/medical/medical_adamw_stage_c.json",
  "results_seq2/stage_c/medical/medical_adamw_final.json",
]

def load_med_acc(path):
  if not os.path.exists(path):
    return None
  with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  acc = payload.get("accuracy")
  if acc is None:
    acc = payload.get("judge_accuracy")
  return acc

ablation_files = sorted(glob.glob(os.path.join(stage_c_dir, "medical_*.json")))
ablation_rows = []
for p in ablation_files:
  tag = os.path.basename(p).replace("medical_", "").replace(".json", "")
  ablation_rows.append((tag, load_med_acc(p)))

current_med_acc = None
current_path = None
for p in current_candidates:
    current_med_acc = load_med_acc(p)
    if current_med_acc is not None:
        current_path = p
        break

print("Stage C Muon ablations (medical accuracy):")
if not ablation_rows:
  print("  No ablation outputs found.")
else:
  for tag, acc in ablation_rows:
    if acc is None:
      print(f"  {tag}: N/A")
    else:
      print(f"  {tag}: {acc:.6f}")

if current_med_acc is not None:
  print(f"\nCurrent Stage C reference ({current_path}): {current_med_acc:.6f}")
  for tag, acc in ablation_rows:
    if acc is None:
      continue
    delta = acc - current_med_acc
    sign = "+" if delta >= 0 else ""
    print(f"  delta ({tag} - current): {sign}{delta:.6f}")
else:
  print("\nCurrent Stage C reference: N/A (no baseline file found)")
PY

echo "Ablation run finished."
