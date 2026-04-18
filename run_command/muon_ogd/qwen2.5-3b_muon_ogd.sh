#!/usr/bin/env bash
set -euo pipefail

# Unified Muon-OGD runner for Qwen2.5-3B-Instruct.
# - Muon-OGD: repeated runs
# - Eval after every stage in every run
# - Final mean/std summary by stage

PROJECT_ROOT="/work/nvme/bgeo/blu7/muon_CL"
RUN_ROOT="${PROJECT_ROOT}/3B-instruct/Muon_OGD"
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

mkdir -p "${RUN_ROOT}"

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

  if ! "${PYTHON_BIN}" -u eval_bigcodebench_remote.py \
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
    2>&1 | tee "${log_root}/${stage_tag}/coding/eval_bcb_${stage_tag}.log"; then
    echo "[WARN] Coding eval failed at ${stage_tag} (${bcb_tag}); continuing to next steps." \
      | tee -a "${log_root}/${stage_tag}/coding/eval_bcb_${stage_tag}.log"
  fi

  "${PYTHON_BIN}" -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --num_examples 500 \
    --seed 42 \
    --out_file "${res_root}/${stage_tag}/math/gsm8k_${stage_tag}.json" \
    2>&1 | tee "${log_root}/${stage_tag}/math/eval_gsm8k_${stage_tag}.log"

  "${PYTHON_BIN}" -u eval_medical.py \
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

run_muon() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="muon_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p \
    "${out_root}" \
    "${log_root}/stage_a/train" \
    "${log_root}/stage_b/train" \
    "${log_root}/stage_c/train"

  echo "[Muon][Run ${run_idx}] Stage A train"
  "${PYTHON_BIN}" -u train_coding_bigcodebench_muon_ogd.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --ci_model_id Qwen/Qwen2.5-3B-Instruct \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_qwen3b_instruct_muon_ogd" \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --save_strategy no \
    --seed "${seed}" \
    --muon_ogd \
    --muon_use_optimizer_class \
    --muon_layers gate_proj,up_proj,down_proj \
    --muon_k 10 \
    --muon_T 1 \
    --muon_eta 1e-5 \
    --muon_eta_dual 1e-5 \
    --muon_warm_start \
    --muon_momentum 0.95 \
    2>&1 | tee "${log_root}/stage_a/train/train_coding_muon_ogd.log"

  echo "[Muon][Run ${run_idx}] Stage A eval"
  eval_stage \
    "${out_root}/sft_seq2_coding_qwen3b_instruct_muon_ogd" \
    "${res_root}" \
    "${log_root}" \
    "stage_a" \
    "bcb_hard_muon"

  echo "[Muon][Run ${run_idx}] Stage B train"
  "${PYTHON_BIN}" -u train_math_sft_svd.py \
    --model_id "${out_root}/sft_seq2_coding_qwen3b_instruct_muon_ogd" \
    --ci_model_id "${out_root}/sft_seq2_coding_qwen3b_instruct_muon_ogd" \
    --output_dir "${out_root}/sft_seq2_math_qwen3b_instruct_muon_ogd_from_coding" \
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
    --muon_ogd \
    --muon_use_optimizer_class \
    --muon_k 3 \
    --muon_T 1 \
    --muon_eta 1e-4 \
    --muon_eta_dual 1e-4 \
    --muon_warm_start \
    --muon_layers gate_proj,up_proj,down_proj \
    --muon_momentum 0.95 \
    2>&1 | tee "${log_root}/stage_b/train/train_math_muon_ogd_from_coding.log"

  echo "[Muon][Run ${run_idx}] Stage B eval"
  eval_stage \
    "${out_root}/sft_seq2_math_qwen3b_instruct_muon_ogd_from_coding" \
    "${res_root}" \
    "${log_root}" \
    "stage_b" \
    "bcb_hard_muon_from_coding_math"

  echo "[Muon][Run ${run_idx}] Stage C train"
  "${PYTHON_BIN}" -u train_medical_muon_ogd.py \
    --model_id "${out_root}/sft_seq2_math_qwen3b_instruct_muon_ogd_from_coding" \
    --ci_model_id "${out_root}/sft_seq2_math_qwen3b_instruct_muon_ogd_from_coding" \
    --output_dir "${out_root}/sft_seq2_medical_qwen3b_instruct_muon_ogd_from_coding_math" \
    --max_length 1536 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --lr 2e-5 \
    --max_steps 0 \
    --seed "${seed}" \
    --muon_ogd \
    --muon_use_optimizer_class \
    --muon_k 3 \
    --muon_T 1 \
    --muon_eta 1e-4 \
    --muon_eta_dual 1e-4 \
    --muon_warm_start \
    --muon_layers gate_proj,up_proj,down_proj \
    --muon_momentum 0.95 \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_muon_ogd_from_coding_math.log"

  echo "[Muon][Run ${run_idx}] Stage C eval"
  eval_stage \
    "${out_root}/sft_seq2_medical_qwen3b_instruct_muon_ogd_from_coding_math" \
    "${res_root}" \
    "${log_root}" \
    "stage_c" \
    "bcb_hard_muon_final"
}

for i in $(seq 1 "${NUM_REPEATS}"); do
  seed=$((BASE_SEED + i - 1))
  run_muon "${i}" "${seed}"
done

"${PYTHON_BIN}" - <<PY
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


stages = ["stage_a", "stage_b", "stage_c"]
runs = sorted([d for d in os.listdir(root) if d.startswith("muon_run")])
summary = {"muon": {"runs": runs, "stages": {}}}

for s in stages:
    coding_vals, math_vals, med_vals, avg_vals = [], [], [], []
    for r in runs:
        rr = os.path.join(root, r, "results", s)
        coding = load_metric(
            os.path.join(
                rr,
                "coding",
                {
                    "stage_a": "bcb_hard_muon",
                    "stage_b": "bcb_hard_muon_from_coding_math",
                    "stage_c": "bcb_hard_muon_final",
                }[s],
                "pass_at_k.json",
            )
        )
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

    summary["muon"]["stages"][s] = {
        "coding": {"mean": cm, "std": cs, "values": coding_vals},
        "math": {"mean": mm, "std": ms, "values": math_vals},
        "medical": {"mean": hm, "std": hs, "values": med_vals},
        "average": {"mean": am, "std": avs, "values": avg_vals},
    }

out_path = os.path.join(root, "summary_mean_std.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Wrote summary: {out_path}")


def fmt(v):
    return "NA" if v is None else f"{v:.3f}"


print(f"\\n[muon] runs={summary['muon']['runs']}")
for s in stages:
    row = summary["muon"]["stages"][s]
    print(
        f"{s}: coding={fmt(row['coding']['mean'])}+-{fmt(row['coding']['std'])}, "
        f"math={fmt(row['math']['mean'])}+-{fmt(row['math']['std'])}, "
        f"medical={fmt(row['medical']['mean'])}+-{fmt(row['medical']['std'])}, "
        f"avg={fmt(row['average']['mean'])}+-{fmt(row['average']['std'])}"
    )
PY

echo "All runs complete. Results and logs under ${RUN_ROOT}"


