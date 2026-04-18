#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-1.5B-Instruct runner with O-LoRA (parameter-efficient) updates:
# - O-LoRA runs: 2 repeats
# - Same stage/task/eval process as qwen1p5b_run.sh (Stage A/B/C + eval after each stage)
#
# This script enforces O-LoRA training in every SFT stage.

USER_NAME="${USER_NAME:-blu7}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/nvme/bgeo/${USER_NAME}/muon_CL}"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/1.5B-instruct}"

NUM_REPEATS="${NUM_REPEATS:-2}"
BASE_SEED="${BASE_SEED:-42}"
NUM_TASKS="${NUM_TASKS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SUBMIT_RETRIES="${SUBMIT_RETRIES:-10}"
SUBMIT_RETRY_DELAY_SEC="${SUBMIT_RETRY_DELAY_SEC:-60}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/O-LoRA}"
mkdir -p "${RUN_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"

# O-LoRA hyperparameters (override via env vars if needed)
OLORA_R="${OLORA_R:-16}"
OLORA_ALPHA="${OLORA_ALPHA:-32}"
OLORA_DROPOUT="${OLORA_DROPOUT:-0.05}"
OLORA_TARGET_MODULES="${OLORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Shared O-LoRA args used in all three training stages.
# Adjust these flags to match your training scripts if argument names differ.
O_LORA_ARGS=(
  --use_olora
  --olora_r "${OLORA_R}"
  --olora_alpha "${OLORA_ALPHA}"
  --olora_dropout "${OLORA_DROPOUT}"
  --olora_target_modules "${OLORA_TARGET_MODULES}"
)

preflight_olora_checks() {
  local scripts=(
    "train_coding_bigcodebench_sft.py"
    "train_math_sft.py"
    "train_medical_sft.py"
  )

  echo "[preflight] checking PEFT installation..."
  if ! python -c 'import peft' >/dev/null 2>&1; then
    echo "[ERROR] PEFT is not installed (or not importable). Install with: pip install peft" >&2
    exit 1
  fi
  echo "[preflight] PEFT found"

  for f in "${scripts[@]}"; do
    if [[ ! -f "${f}" ]]; then
      echo "[ERROR] missing training script: ${f}" >&2
      exit 1
    fi
    if ! grep -q -- "--use_olora" "${f}"; then
      echo "[ERROR] ${f} does not expose --use_olora" >&2
      exit 1
    fi
    if ! grep -q -- 'init_lora_weights="olora"' "${f}"; then
      echo "[ERROR] ${f} does not configure O-LoRA init_lora_weights=\"olora\"" >&2
      exit 1
    fi
  done

  echo "[preflight] O-LoRA support checks passed"
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
    --out_dir "${res_root}/${stage_tag}/coding/${bcb_tag}" \
    --no-debug_first_sample \
    2>&1 | tee "${log_root}/${stage_tag}/coding/eval_bcb_${stage_tag}.log"; then
    echo "[WARN] Coding eval failed at ${stage_tag} (${bcb_tag}); continuing to next steps." \
      | tee -a "${log_root}/${stage_tag}/coding/eval_bcb_${stage_tag}.log"
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

run_olora() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="olora_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}" "${log_root}/stage_a/train" "${log_root}/stage_b/train" "${log_root}/stage_c/train"

  echo "[O-LoRA][Run ${run_idx}] Stage A train"
  python -u train_coding_bigcodebench_sft.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_qwen1.5b_instruct_olora" \
    --max_length 2048 \
    --lr 5e-6 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --save_strategy no \
    --seed "${seed}" \
    "${O_LORA_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_a/train/train_coding_olora.log"

  echo "[O-LoRA][Run ${run_idx}] Stage A eval"
  eval_stage \
    "${out_root}/sft_seq2_coding_qwen1.5b_instruct_olora" \
    "${res_root}" \
    "${log_root}" \
    "stage_a" \
    "bcb_hard_olora"

  echo "[O-LoRA][Run ${run_idx}] Stage B train"
  python -u train_math_sft.py \
    --model_id "${out_root}/sft_seq2_coding_qwen1.5b_instruct_olora" \
    --output_dir "${out_root}/sft_seq2_math_qwen1.5b_instruct_olora_from_coding" \
    --num_train_examples 2000 \
    --max_length 512 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 4 \
    --lr 2e-5 \
    --max_steps 1000 \
    --val_ratio 0.02 \
    --val_every 100 \
    --val_max_batches 32 \
    --probe_every 100 \
    --probe_max_new_tokens 64 \
    --save_strategy no \
    --seed "${seed}" \
    "${O_LORA_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_b/train/train_math_olora_from_coding.log"

  echo "[O-LoRA][Run ${run_idx}] Stage B eval"
  eval_stage \
    "${out_root}/sft_seq2_math_qwen1.5b_instruct_olora_from_coding" \
    "${res_root}" \
    "${log_root}" \
    "stage_b" \
    "bcb_hard_olora_from_coding_math"

  echo "[O-LoRA][Run ${run_idx}] Stage C train"
  python -u train_medical_sft.py \
    --model_id "${out_root}/sft_seq2_math_qwen1.5b_instruct_olora_from_coding" \
    --output_dir "${out_root}/sft_seq2_medical_qwen1.5b_instruct_olora_from_coding_math" \
    --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT \
    --dataset_config en \
    --train_split train \
    --question_field Question \
    --answer_field Response \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --lr 5e-6 \
    --answer_after_cot_only \
    --seed "${seed}" \
    "${O_LORA_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_olora_from_coding_math.log"

  echo "[O-LoRA][Run ${run_idx}] Stage C eval"
  eval_stage \
    "${out_root}/sft_seq2_medical_qwen1.5b_instruct_olora_from_coding_math" \
    "${res_root}" \
    "${log_root}" \
    "stage_c" \
    "bcb_hard_olora_final"
}

preflight_olora_checks
for i in $(seq 1 "${NUM_REPEATS}"); do
  seed=$((BASE_SEED + i - 1))
  run_olora "${i}" "${seed}"
done

python - <<PY
import json
import math
import os

root = "${RUN_ROOT}"

def load_metric(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    if "pass@1" in j:
        return float(j["pass@1"]) * 100
    if "accuracy" in j:
        return float(j["accuracy"]) * 100
    if "judge_accuracy" in j:
        return float(j["judge_accuracy"]) * 100
    return None

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)

optimizers = ["olora"]
stages = ["stage_a", "stage_b", "stage_c"]

summary = {}
for opt in optimizers:
    runs = sorted([d for d in os.listdir(root) if d.startswith(opt + "_run")])
    summary[opt] = {"runs": runs, "stages": {}}
    for s in stages:
        coding_vals, math_vals, med_vals, avg_vals = [], [], [], []
        for r in runs:
            rr = os.path.join(root, r, "results", s)
            coding = load_metric(os.path.join(rr, "coding", {
                "stage_a": "bcb_hard_olora",
                "stage_b": "bcb_hard_olora_from_coding_math",
                "stage_c": "bcb_hard_olora_final",
            }[s], "pass_at_k.json"))
            mathv = load_metric(os.path.join(rr, "math", f"gsm8k_{s}.json"))
            med = load_metric(os.path.join(rr, "medical", f"medical_{s}.json"))

            coding_vals.append(coding)
            math_vals.append(mathv)
            med_vals.append(med)
            row = [v for v in (coding, mathv, med) if v is not None]
            avg_vals.append(sum(row) / len(row) if row else None)

        cm, cs = mean_std(coding_vals)
        mm, ms = mean_std(math_vals)
        hm, hs = mean_std(med_vals)
        am, avs = mean_std(avg_vals)

        summary[opt]["stages"][s] = {
            "coding": {"mean": cm, "std": cs, "values": coding_vals},
            "math": {"mean": mm, "std": ms, "values": math_vals},
            "medical": {"mean": hm, "std": hs, "values": med_vals},
            "average": {"mean": am, "std": avs, "values": avg_vals},
        }

out_path = os.path.join(root, "summary_mean_std_olora.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Wrote summary: {out_path}")
for opt in optimizers:
    print(f"\\n[{opt}] runs={summary[opt]['runs']}")
    for s in stages:
        row = summary[opt]["stages"][s]
        c = row["coding"]
        m = row["math"]
        h = row["medical"]
        a = row["average"]
        print(
            f"{s}: coding={c['mean']:.3f}+-{c['std']:.3f}, "
            f"math={m['mean']:.3f}+-{m['std']:.3f}, "
            f"medical={h['mean']:.3f}+-{h['std']:.3f}, "
            f"avg={a['mean']:.3f}+-{a['std']:.3f}"
        )
PY
