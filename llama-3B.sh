#!/usr/bin/env bash
set -euo pipefail

# Unified continual-learning runner for Llama-3.2-3B-Instruct.
# Algorithms: AdamW (SeqSFT), O-LoRA, Sculpting Subspace, Muon-OGD.
# Stages: A (coding) -> B (math) -> C (medical), eval after each stage.
#
# Muon-OGD runs MUON_NUM_REPEATS=2 seeds by default so the aggregator can
# report mean ± std. Other methods default to NUM_REPEATS=1.
#
# Eval is vLLM-accelerated (USE_VLLM=1 below). The eval scripts pick up the
# vLLM backend through vllm_eval_backend.py.
#
# Model defaults to a NON-gated mirror (`unsloth/Llama-3.2-3B-Instruct`).
# Override MODEL_ID to a local path if you have weights staged.
#   e.g. MODEL_ID=/path/to/local/Llama-3.2-3B-Instruct bash llama-3B.sh

USER_NAME="${USER_NAME:-${USER:-unknown}}"
PROJECT_ROOT="${PROJECT_ROOT:-${PWD:-.}}"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs}"

MODEL_ID="${MODEL_ID:-unsloth/Llama-3.2-3B-Instruct}"
BASE_MODEL_ID="${BASE_MODEL_ID:-${MODEL_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Per-method seed counts (Muon-OGD gets ≥2 seeds for std).
NUM_REPEATS="${NUM_REPEATS:-1}"
MUON_NUM_REPEATS="${MUON_NUM_REPEATS:-2}"
BASE_SEED="${BASE_SEED:-42}"

RUN_ADAMW="${RUN_ADAMW:-1}"
RUN_OLORA="${RUN_OLORA:-1}"
RUN_SUBSPACE="${RUN_SUBSPACE:-1}"
RUN_MUON="${RUN_MUON:-1}"

NUM_TASKS="${NUM_TASKS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SUBMIT_RETRIES="${SUBMIT_RETRIES:-10}"
SUBMIT_RETRY_DELAY_SEC="${SUBMIT_RETRY_DELAY_SEC:-60}"
BCB_NO_SUBMIT="${BCB_NO_SUBMIT:-1}"

BCB_EVAL_ARGS=()
if [[ "${BCB_NO_SUBMIT}" == "1" ]]; then
  BCB_EVAL_ARGS+=(--no_submit)
fi

# ---- vLLM-accelerated eval defaults (override via env at launch). ----
export USE_VLLM="${USE_VLLM:-1}"
export EVAL_VLLM_TP="${EVAL_VLLM_TP:-1}"
export EVAL_VLLM_MAX_MODEL_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-4096}"
export EVAL_VLLM_GPU_MEM_UTIL="${EVAL_VLLM_GPU_MEM_UTIL:-0.85}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# Training-side cuDNN SDPA workaround for this GH200 stack.
export TRAIN_ATTN_IMPL="${TRAIN_ATTN_IMPL:-eager}"

mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"

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

# Sculpting-subspace hyperparameters
SUBSPACE_TOP_FRACTION="${SUBSPACE_TOP_FRACTION:-0.5}"
SUBSPACE_TARGET_MODULES="${SUBSPACE_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
SUBSPACE_ARGS=(
  --use_sculpt_subspace
  --subspace_top_fraction "${SUBSPACE_TOP_FRACTION}"
  --subspace_target_modules "${SUBSPACE_TARGET_MODULES}"
)

# Best Muon-OGD hyperparameters from the Stage C ablation sweep.
MUON_LR="${MUON_LR:-5e-5}"          # AdamW LR for non-Muon params
MUON_K="${MUON_K:-3}"
MUON_T="${MUON_T:-1}"
MUON_ETA="${MUON_ETA:-1e-6}"        # Muon primal step
MUON_ETA_DUAL="${MUON_ETA_DUAL:-5e-5}"
MUON_LAYERS="${MUON_LAYERS:-gate_proj,up_proj,down_proj}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"

require_cmds() {
  local files=(
    train_coding_bigcodebench_sft.py
    train_coding_bigcodebench_muon_ogd.py
    train_math_sft.py
    train_math_sft_svd.py
    train_medical_sft.py
    train_medical_muon_ogd.py
    eval_bigcodebench_remote.py
    eval_gsm8k.py
    eval_medical.py
    vllm_eval_backend.py
  )
  for f in "${files[@]}"; do
    [[ -f "${f}" ]] || { echo "[ERROR] missing ${f}" >&2; exit 1; }
  done
}

preflight_olora() {
  if [[ "${RUN_OLORA}" != "1" ]]; then
    return
  fi
  if ! "${PYTHON_BIN}" -c 'import peft' >/dev/null 2>&1; then
    echo "[ERROR] O-LoRA requested but PEFT not importable in ${PYTHON_BIN}" >&2
    exit 1
  fi
}

preflight_model_access() {
  "${PYTHON_BIN}" - <<PY
import sys
from transformers import AutoConfig, AutoTokenizer

model_id = ${MODEL_ID@Q}

try:
  AutoConfig.from_pretrained(model_id)
except Exception as e:
  print(f"[ERROR] Cannot load model config for {model_id}: {e}", file=sys.stderr)
  print("[HINT] If gated, run 'huggingface-cli login' or set MODEL_ID to a local snapshot.", file=sys.stderr)
  raise SystemExit(2)

try:
  tok = AutoTokenizer.from_pretrained(model_id)
  msg = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hi"},
  ]
  _ = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
except Exception as e:
  print(f"[ERROR] Tokenizer/chat-template check failed for {model_id}: {e}", file=sys.stderr)
  raise SystemExit(3)

print(f"[preflight] model access and chat-template OK: {model_id}")
PY
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
    echo "[WARN] Coding eval failed at ${stage_tag} (${bcb_tag}); continuing." \
      | tee -a "${log_root}/${stage_tag}/coding/eval_bcb_${stage_tag}.log"
  fi

  "${PYTHON_BIN}" -u eval_gsm8k.py \
    --model_id "${model_id}" \
    --num_examples 500 \
    --seed 42 \
    --out_file "${res_root}/${stage_tag}/math/gsm8k_${stage_tag}.json" \
    2>&1 | tee "${log_root}/${stage_tag}/math/eval_gsm8k_${stage_tag}.log"

  # Medical eval co-hosts the 3B verifier; cap vLLM mem to leave room.
  EVAL_VLLM_GPU_MEM_UTIL="${EVAL_VLLM_GPU_MEM_UTIL_MEDICAL:-0.55}" \
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

run_adamw() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="adamw_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}" "${log_root}/stage_a/train" "${log_root}/stage_b/train" "${log_root}/stage_c/train"

  echo "[AdamW][Run ${run_idx}] Stage A train"
  "${PYTHON_BIN}" -u train_coding_bigcodebench_sft.py \
    --model_id "${MODEL_ID}" \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_llama3b_instruct_adamw" \
    --max_length 2048 \
    --lr 5e-6 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --save_strategy no \
    --seed "${seed}" \
    2>&1 | tee "${log_root}/stage_a/train/train_coding_adamw.log"

  eval_stage "${out_root}/sft_seq2_coding_llama3b_instruct_adamw" "${res_root}" "${log_root}" "stage_a" "bcb_hard_adamw"

  echo "[AdamW][Run ${run_idx}] Stage B train"
  "${PYTHON_BIN}" -u train_math_sft.py \
    --model_id "${out_root}/sft_seq2_coding_llama3b_instruct_adamw" \
    --output_dir "${out_root}/sft_seq2_math_llama3b_instruct_adamw_from_coding" \
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
    2>&1 | tee "${log_root}/stage_b/train/train_math_adamw_from_coding.log"

  eval_stage "${out_root}/sft_seq2_math_llama3b_instruct_adamw_from_coding" "${res_root}" "${log_root}" "stage_b" "bcb_hard_adamw_from_coding_math"

  echo "[AdamW][Run ${run_idx}] Stage C train"
  "${PYTHON_BIN}" -u train_medical_sft.py \
    --model_id "${out_root}/sft_seq2_math_llama3b_instruct_adamw_from_coding" \
    --base_model_id "${BASE_MODEL_ID}" \
    --output_dir "${out_root}/sft_seq2_medical_llama3b_instruct_adamw_from_coding_math" \
    --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT \
    --dataset_config en \
    --train_split train \
    --question_field Question \
    --answer_field Response \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 800 \
    --lr 5e-6 \
    --answer_after_cot_only \
    --seed "${seed}" \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_adamw_from_coding_math.log"

  eval_stage "${out_root}/sft_seq2_medical_llama3b_instruct_adamw_from_coding_math" "${res_root}" "${log_root}" "stage_c" "bcb_hard_adamw_final"
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
  "${PYTHON_BIN}" -u train_coding_bigcodebench_sft.py \
    --model_id "${MODEL_ID}" \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_llama3b_instruct_olora" \
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

  eval_stage "${out_root}/sft_seq2_coding_llama3b_instruct_olora" "${res_root}" "${log_root}" "stage_a" "bcb_hard_olora"

  echo "[O-LoRA][Run ${run_idx}] Stage B train"
  "${PYTHON_BIN}" -u train_math_sft.py \
    --model_id "${out_root}/sft_seq2_coding_llama3b_instruct_olora" \
    --output_dir "${out_root}/sft_seq2_math_llama3b_instruct_olora_from_coding" \
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

  eval_stage "${out_root}/sft_seq2_math_llama3b_instruct_olora_from_coding" "${res_root}" "${log_root}" "stage_b" "bcb_hard_olora_from_coding_math"

  echo "[O-LoRA][Run ${run_idx}] Stage C train"
  "${PYTHON_BIN}" -u train_medical_sft.py \
    --model_id "${out_root}/sft_seq2_math_llama3b_instruct_olora_from_coding" \
    --base_model_id "${BASE_MODEL_ID}" \
    --output_dir "${out_root}/sft_seq2_medical_llama3b_instruct_olora_from_coding_math" \
    --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT \
    --dataset_config en \
    --train_split train \
    --question_field Question \
    --answer_field Response \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 800 \
    --lr 5e-6 \
    --answer_after_cot_only \
    --seed "${seed}" \
    "${O_LORA_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_olora_from_coding_math.log"

  eval_stage "${out_root}/sft_seq2_medical_llama3b_instruct_olora_from_coding_math" "${res_root}" "${log_root}" "stage_c" "bcb_hard_olora_final"
}

run_subspace() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="subspace_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}" "${log_root}/stage_a/train" "${log_root}/stage_b/train" "${log_root}/stage_c/train"

  echo "[Subspace][Run ${run_idx}] Stage A train"
  "${PYTHON_BIN}" -u train_coding_bigcodebench_sft.py \
    --model_id "${MODEL_ID}" \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_llama3b_instruct_subspace" \
    --max_length 2048 \
    --lr 5e-6 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --save_strategy no \
    --seed "${seed}" \
    "${SUBSPACE_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_a/train/train_coding_subspace.log"

  eval_stage "${out_root}/sft_seq2_coding_llama3b_instruct_subspace" "${res_root}" "${log_root}" "stage_a" "bcb_hard_subspace"

  echo "[Subspace][Run ${run_idx}] Stage B train"
  "${PYTHON_BIN}" -u train_math_sft.py \
    --model_id "${out_root}/sft_seq2_coding_llama3b_instruct_subspace" \
    --output_dir "${out_root}/sft_seq2_math_llama3b_instruct_subspace_from_coding" \
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
    "${SUBSPACE_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_b/train/train_math_subspace_from_coding.log"

  eval_stage "${out_root}/sft_seq2_math_llama3b_instruct_subspace_from_coding" "${res_root}" "${log_root}" "stage_b" "bcb_hard_subspace_from_coding_math"

  echo "[Subspace][Run ${run_idx}] Stage C train"
  "${PYTHON_BIN}" -u train_medical_sft.py \
    --model_id "${out_root}/sft_seq2_math_llama3b_instruct_subspace_from_coding" \
    --base_model_id "${BASE_MODEL_ID}" \
    --output_dir "${out_root}/sft_seq2_medical_llama3b_instruct_subspace_from_coding_math" \
    --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT \
    --dataset_config en \
    --train_split train \
    --question_field Question \
    --answer_field Response \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 800 \
    --lr 5e-6 \
    --answer_after_cot_only \
    --seed "${seed}" \
    "${SUBSPACE_ARGS[@]}" \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_subspace_from_coding_math.log"

  eval_stage "${out_root}/sft_seq2_medical_llama3b_instruct_subspace_from_coding_math" "${res_root}" "${log_root}" "stage_c" "bcb_hard_subspace_final"
}

run_muon() {
  local run_idx="$1"
  local seed="$2"
  local run_tag="muon_run${run_idx}"
  local out_root="${RUN_ROOT}/${run_tag}/outputs"
  local log_root="${RUN_ROOT}/${run_tag}/logs"
  local res_root="${RUN_ROOT}/${run_tag}/results"

  mkdir -p "${out_root}" "${log_root}/stage_a/train" "${log_root}/stage_b/train" "${log_root}/stage_c/train"

  # All three stages use the best Muon-OGD hyperparameters from the Stage C ablation:
  #   lr=${MUON_LR}, muon_T=${MUON_T}, muon_eta=${MUON_ETA},
  #   muon_eta_dual=${MUON_ETA_DUAL}, muon_k=${MUON_K}.

  echo "[Muon][Run ${run_idx}] Stage A train (lr=${MUON_LR}, k=${MUON_K}, T=${MUON_T}, eta=${MUON_ETA}, eta_dual=${MUON_ETA_DUAL})"
  "${PYTHON_BIN}" -u train_coding_bigcodebench_muon_ogd.py \
    --model_id "${MODEL_ID}" \
    --ci_model_id "${MODEL_ID}" \
    --split complete \
    --output_dir "${out_root}/sft_seq2_coding_llama3b_instruct_muon_ogd" \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --max_steps 0 \
    --lr "${MUON_LR}" \
    --save_strategy no \
    --seed "${seed}" \
    --muon_ogd \
    --muon_use_optimizer_class \
    --muon_layers "${MUON_LAYERS}" \
    --muon_k "${MUON_K}" \
    --muon_T "${MUON_T}" \
    --muon_eta "${MUON_ETA}" \
    --muon_eta_dual "${MUON_ETA_DUAL}" \
    --muon_warm_start \
    --muon_momentum "${MUON_MOMENTUM}" \
    2>&1 | tee "${log_root}/stage_a/train/train_coding_muon_ogd.log"

  eval_stage "${out_root}/sft_seq2_coding_llama3b_instruct_muon_ogd" "${res_root}" "${log_root}" "stage_a" "bcb_hard_muon"

  echo "[Muon][Run ${run_idx}] Stage B train"
  "${PYTHON_BIN}" -u train_math_sft_svd.py \
    --model_id "${out_root}/sft_seq2_coding_llama3b_instruct_muon_ogd" \
    --ci_model_id "${out_root}/sft_seq2_coding_llama3b_instruct_muon_ogd" \
    --output_dir "${out_root}/sft_seq2_math_llama3b_instruct_muon_ogd_from_coding" \
    --num_train_examples 2000 \
    --max_length 512 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 4 \
    --lr "${MUON_LR}" \
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
    --muon_k "${MUON_K}" \
    --muon_T "${MUON_T}" \
    --muon_eta "${MUON_ETA}" \
    --muon_eta_dual "${MUON_ETA_DUAL}" \
    --muon_warm_start \
    --muon_layers "${MUON_LAYERS}" \
    --muon_momentum "${MUON_MOMENTUM}" \
    2>&1 | tee "${log_root}/stage_b/train/train_math_muon_ogd_from_coding.log"

  eval_stage "${out_root}/sft_seq2_math_llama3b_instruct_muon_ogd_from_coding" "${res_root}" "${log_root}" "stage_b" "bcb_hard_muon_from_coding_math"

  echo "[Muon][Run ${run_idx}] Stage C train"
  "${PYTHON_BIN}" -u train_medical_muon_ogd.py \
    --model_id "${out_root}/sft_seq2_math_llama3b_instruct_muon_ogd_from_coding" \
    --ci_model_id "${out_root}/sft_seq2_math_llama3b_instruct_muon_ogd_from_coding" \
    --output_dir "${out_root}/sft_seq2_medical_llama3b_instruct_muon_ogd_from_coding_math" \
    --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT \
    --dataset_config en \
    --train_split train \
    --question_field Question \
    --answer_field Response \
    --max_length 2048 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 1 \
    --lr "${MUON_LR}" \
    --max_steps 800 \
    --answer_after_cot_only \
    --seed "${seed}" \
    --muon_ogd \
    --muon_use_optimizer_class \
    --muon_k "${MUON_K}" \
    --muon_T "${MUON_T}" \
    --muon_eta "${MUON_ETA}" \
    --muon_eta_dual "${MUON_ETA_DUAL}" \
    --muon_warm_start \
    --muon_layers "${MUON_LAYERS}" \
    --muon_momentum "${MUON_MOMENTUM}" \
    2>&1 | tee "${log_root}/stage_c/train/train_medical_muon_ogd_from_coding_math.log"

  eval_stage "${out_root}/sft_seq2_medical_llama3b_instruct_muon_ogd_from_coding_math" "${res_root}" "${log_root}" "stage_c" "bcb_hard_muon_final"
}

require_cmds
preflight_olora
preflight_model_access

# Single-seed methods
for i in $(seq 1 "${NUM_REPEATS}"); do
  seed=$((BASE_SEED + i - 1))
  [[ "${RUN_ADAMW}" == "1" ]]    && run_adamw    "${i}" "${seed}"
  [[ "${RUN_OLORA}" == "1" ]]    && run_olora    "${i}" "${seed}"
  [[ "${RUN_SUBSPACE}" == "1" ]] && run_subspace "${i}" "${seed}"
done

# Muon-OGD: multi-seed for std
for i in $(seq 1 "${MUON_NUM_REPEATS}"); do
  seed=$((BASE_SEED + i - 1))
  [[ "${RUN_MUON}" == "1" ]] && run_muon "${i}" "${seed}"
done

"${PYTHON_BIN}" - <<PY
import json
import math
import os

root = "${RUN_ROOT}"
optimizers = [
    ("adamw", ["stage_a", "bcb_hard_adamw"], ["stage_b", "bcb_hard_adamw_from_coding_math"], ["stage_c", "bcb_hard_adamw_final"]),
    ("olora", ["stage_a", "bcb_hard_olora"], ["stage_b", "bcb_hard_olora_from_coding_math"], ["stage_c", "bcb_hard_olora_final"]),
    ("subspace", ["stage_a", "bcb_hard_subspace"], ["stage_b", "bcb_hard_subspace_from_coding_math"], ["stage_c", "bcb_hard_subspace_final"]),
    ("muon", ["stage_a", "bcb_hard_muon"], ["stage_b", "bcb_hard_muon_from_coding_math"], ["stage_c", "bcb_hard_muon_final"]),
]

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

summary = {}
for opt, sA, sB, sC in optimizers:
    runs = sorted([d for d in os.listdir(root) if d.startswith(opt + "_run")])
    if not runs:
        continue

    stage_to_tag = {sA[0]: sA[1], sB[0]: sB[1], sC[0]: sC[1]}
    summary[opt] = {"runs": runs, "stages": {}}

    for stage in ["stage_a", "stage_b", "stage_c"]:
        coding_vals, math_vals, med_vals, avg_vals = [], [], [], []
        for r in runs:
            rr = os.path.join(root, r, "results", stage)
            coding = load_metric(os.path.join(rr, "coding", stage_to_tag[stage], "pass_at_k.json"))
            mathv = load_metric(os.path.join(rr, "math", f"gsm8k_{stage}.json"))
            med = load_metric(os.path.join(rr, "medical", f"medical_{stage}.json"))

            coding_vals.append(coding)
            math_vals.append(mathv)
            med_vals.append(med)
            row = [v for v in (coding, mathv, med) if v is not None]
            avg_vals.append(sum(row) / len(row) if row else None)

        cm, cs = mean_std(coding_vals)
        mm, ms = mean_std(math_vals)
        hm, hs = mean_std(med_vals)
        am, avs = mean_std(avg_vals)

        summary[opt]["stages"][stage] = {
            "coding": {"mean": cm, "std": cs, "values": coding_vals},
            "math": {"mean": mm, "std": ms, "values": math_vals},
            "medical": {"mean": hm, "std": hs, "values": med_vals},
            "average": {"mean": am, "std": avs, "values": avg_vals},
        }

out_path = os.path.join(root, "summary_mean_std_all.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Wrote summary: {out_path}")
for opt, payload in summary.items():
    print(f"\\n[{opt}] runs={payload['runs']}")
    for stage in ["stage_a", "stage_b", "stage_c"]:
        row = payload["stages"][stage]
        def fmt(v):
            return "NA" if v is None else f"{v:.3f}"
        print(
            f"{stage}: coding={fmt(row['coding']['mean'])}+-{fmt(row['coding']['std'])}, "
            f"math={fmt(row['math']['mean'])}+-{fmt(row['math']['std'])}, "
            f"medical={fmt(row['medical']['mean'])}+-{fmt(row['medical']['std'])}, "
            f"avg={fmt(row['average']['mean'])}+-{fmt(row['average']['std'])}"
        )
PY

echo "All requested runs complete. Results under ${RUN_ROOT}"
