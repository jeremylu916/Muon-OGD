#!/usr/bin/env bash
set -euo pipefail

# Sequential Muon-OGD SFT pipeline (Coding -> Math -> Medical)

mkdir -p \
  logs_seq2_muon/stage_a/train \
  logs_seq2_muon/stage_b/train \
  logs_seq2_muon/stage_c/train \
  outputs

echo "[1/3] Stage A Muon-OGD: coding SFT (BigCodeBench)"
python -u train_coding_bigcodebench_muon_ogd.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --ci_model_id Qwen/Qwen2.5-1.5B \
  --split complete \
  --output_dir outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --max_length 2048 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --max_steps 0 \
  --save_strategy no \
  --seed 42 \
  --muon_ogd \
  --muon_use_optimizer_class \
  --muon_layers gate_proj,up_proj,down_proj \
  --muon_k 10 \
  --muon_T 1 \
  --muon_eta 1e-5 \
  --muon_eta_dual 1e-5 \
  --muon_warm_start \
  --muon_momentum 0.95 \
  2>&1 | tee logs_seq2_muon/stage_a/train/train_coding_muon_ogd.log

echo "[2/3] Stage B Muon-OGD: math SFT (GSM8K) from Stage A"
python -u train_math_sft_svd.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --ci_model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --output_dir outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
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
  --seed 42 \
  --muon_ogd \
  --muon_use_optimizer_class \
  --muon_k 3 \
  --muon_T 1 \
  --muon_eta 1e-4 \
  --muon_eta_dual 1e-4 \
  --muon_warm_start \
  --muon_layers gate_proj,up_proj,down_proj \
  --muon_momentum 0.95 \
  2>&1 | tee logs_seq2_muon/stage_b/train/train_math_muon_ogd_from_coding.log

echo "[3/3] Stage C Muon-OGD: medical SFT from Stage B"
python -u train_medical_muon_ogd.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --ci_model_id outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --output_dir outputs/sft_seq2_medical_qwen1.5b_muon_ogd_from_coding_math \
  --max_length 1536 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --lr 2e-5 \
  --max_steps 0 \
  --seed 42 \
  --muon_ogd \
  --muon_use_optimizer_class \
  --muon_k 3 \
  --muon_T 1 \
  --muon_eta 1e-4 \
  --muon_eta_dual 1e-4 \
  --muon_warm_start \
  --muon_layers gate_proj,up_proj,down_proj \
  --muon_momentum 0.95 \
  2>&1 | tee logs_seq2_muon/stage_c/train/train_medical_muon_ogd_from_coding_math.log

echo "Muon-OGD SFT pipeline completed successfully."
