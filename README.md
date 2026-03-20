# README: Continual Learning Order = Medical -> Coding -> Math

This protocol uses the sequence below:

1. Stage A: Medical SFT (Huatuo verifiable)
2. Stage B: Coding SFT (BigCodeBench)
3. Stage C: Math SFT (GSM8K)

All logs in this README go to `logs/`.

## 0) Setup

```bash
mkdir -p \
  logs/stage_a/train logs/stage_a/math logs/stage_a/medical logs/stage_a/coding \
  logs/stage_b/train logs/stage_b/math logs/stage_b/medical logs/stage_b/coding \
  logs/stage_c/train logs/stage_c/math logs/stage_c/medical logs/stage_c/coding \
  results/stage_a/math results/stage_a/medical results/stage_a/coding \
  results/stage_b/math results/stage_b/medical results/stage_b/coding \
  results/stage_c/math results/stage_c/medical results/stage_c/coding
```

For medical eval (API judge):

```bash
export OPENAI_API_KEY="<your_key>"
```

---

## 1) Stage A - Medical (Huatuo)

### AdamW train

```bash
nohup python -u train_medical_sft.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/sft_medical_qwen1.5b_adamw_stage_a \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --train_split train \
  --num_train_examples 10000 \
  --max_length 2048 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --lr 2e-6 \
  --no-answer_after_cot_only \
  --max_steps 0 \
  --save_strategy no \
  --seed 42 \
  > logs/stage_a/train/train_medical_adamw.log 2>&1 &
```

### Muon-OGD train

```bash
nohup python -u train_medical_sft_svd.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/sft_medical_qwen1.5b_muon_ogd_stage_a \
  --max_length 1536 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --lr 2e-5 \
  --max_steps 1000 \
  --save_strategy no \
  --seed 42 \
  --muon_ogd \
  --muon_use_optimizer_class \
  --muon_T 1 \
  --muon_eta 2e-5 \
  --muon_eta_dual 1e-4 \
  --muon_warm_start \
  --muon_layers o_proj,down_proj \
  --muon_momentum 0.95 \
  --muon_dynamic_scale \
  --ci_from_grads \
  --ci_replay_dataset gsm8k \
  --ci_replay_config main \
  --ci_replay_examples 128 \
  --muon_k 3 \
  > logs/stage_a/train/train_medical_muon_ogd.log 2>&1 &
```

### Stage A eval (medical + coding + math)

```bash
# Medical
nohup python -u eval_huatuo_verifiable_api.py \
  --model_id outputs/sft_medical_qwen1.5b_adamw_stage_a \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_api_url https://api.openai.com/v1/chat/completions \
  --judge_model gpt-4o-mini \
  --out_file results/stage_a/medical/medical_adamw.json \
  > logs/stage_a/medical/eval_medical_adamw.log 2>&1 &

nohup python -u eval_huatuo_verifiable_api.py \
  --model_id outputs/sft_medical_qwen1.5b_muon_ogd_stage_a \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_api_url https://api.openai.com/v1/chat/completions \
  --judge_model gpt-4o-mini \
  --out_file results/stage_a/medical/medical_muon_ogd.json \
  > logs/stage_a/medical/eval_medical_muon_ogd.log 2>&1 &

# Coding (HumanEval probe)
nohup python -u eval_HumanEval.py \
  --model_id outputs/sft_medical_qwen1.5b_adamw_stage_a \
  --dataset_id openai/openai_humaneval \
  --split test \
  --num_examples 164 \
  --seed 42 \
  --max_new_tokens 256 \
  --out_file results/stage_a/coding/humaneval_adamw.json \
  --samples_file results/stage_a/coding/humaneval_adamw_samples.jsonl \
  > logs/stage_a/coding/eval_humaneval_adamw.log 2>&1 &

nohup python -u eval_HumanEval.py \
  --model_id outputs/sft_medical_qwen1.5b_muon_ogd_stage_a \
  --dataset_id openai/openai_humaneval \
  --split test \
  --num_examples 164 \
  --seed 42 \
  --max_new_tokens 256 \
  --out_file results/stage_a/coding/humaneval_muon_ogd.json \
  --samples_file results/stage_a/coding/humaneval_muon_ogd_samples.jsonl \
  > logs/stage_a/coding/eval_humaneval_muon_ogd.log 2>&1 &

# Math (GSM8K probe)
nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_medical_qwen1.5b_adamw_stage_a \
  --num_examples 500 --seed 42 \
  --out_file results/stage_a/math/gsm8k_adamw.json \
  > logs/stage_a/math/eval_gsm8k_adamw.log 2>&1 &

nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_medical_qwen1.5b_muon_ogd_stage_a \
  --num_examples 500 --seed 42 \
  --out_file results/stage_a/math/gsm8k_muon_ogd.json \
  > logs/stage_a/math/eval_gsm8k_muon_ogd.log 2>&1 &
```

---

## 2) Stage B - Coding (BigCodeBench)

Train from Stage A medical checkpoints.

### AdamW train

```bash
nohup python -u train_coding_bigcodebench_sft.py \
  --model_id outputs/sft_medical_qwen1.5b_adamw_stage_a \
  --output_dir outputs/sft_bigcodebench_qwen1.5b_adamw_after_medical \
  --split instruct \
  --train_size 800 \
  --max_length 1024 \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 3 \
  --lr 5e-6 \
  --max_steps 500 \
  --save_strategy no \
  --seed 42 \
  > logs/stage_b/train/train_bigcodebench_adamw.log 2>&1 &
```

### Muon-OGD train

```bash
nohup python -u train_coding_bigcodebench_muon_ogd_sft.py \
  --model_id outputs/sft_medical_qwen1.5b_muon_ogd_stage_a \
  --output_dir outputs/sft_bigcodebench_qwen1.5b_muon_ogd_after_medical \
  --split instruct \
  --train_size 800 \
  --max_length 2048 \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 3 \
  --lr 5e-6 \
  --max_steps 500 \
  --save_strategy no \
  --seed 42 \
  --muon_ogd \
  --muon_use_optimizer_class \
  --muon_T 1 \
  --muon_eta 1e-4 \
  --muon_eta_dual 1e-4 \
  --muon_warm_start \
  --muon_layers o_proj,down_proj \
  --muon_momentum 0.95 \
  --muon_dynamic_scale \
  > logs/stage_b/train/train_bigcodebench_muon_ogd.log 2>&1 &
```

### Stage B eval (coding + medical + math)

```bash
# Coding (BigCodeBench held-out split)
nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_adamw_after_medical \
  --split instruct \
  --use_rest_split --train_size 800 --num_tasks 0 --seed 42 \
  --out_dir results/stage_b/coding/bcb_adamw \
  > logs/stage_b/coding/eval_bcb_adamw.log 2>&1 &

nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_muon_ogd_after_medical \
  --split instruct \
  --use_rest_split --train_size 800 --num_tasks 0 --seed 42 \
  --out_dir results/stage_b/coding/bcb_muon_ogd \
  > logs/stage_b/coding/eval_bcb_muon_ogd.log 2>&1 &

# Medical retention
nohup python -u eval_huatuo_verifiable_api.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_adamw_after_medical \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_api_url https://api.openai.com/v1/chat/completions \
  --judge_model gpt-4o-mini \
  --out_file results/stage_b/medical/medical_adamw.json \
  > logs/stage_b/medical/eval_medical_adamw.log 2>&1 &

nohup python -u eval_huatuo_verifiable_api.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_muon_ogd_after_medical \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_api_url https://api.openai.com/v1/chat/completions \
  --judge_model gpt-4o-mini \
  --out_file results/stage_b/medical/medical_muon_ogd.json \
  > logs/stage_b/medical/eval_medical_muon_ogd.log 2>&1 &

# Math retention
nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_adamw_after_medical \
  --num_examples 500 --seed 42 \
  --out_file results/stage_b/math/gsm8k_adamw.json \
  > logs/stage_b/math/eval_gsm8k_adamw.log 2>&1 &

nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_muon_ogd_after_medical \
  --num_examples 500 --seed 42 \
  --out_file results/stage_b/math/gsm8k_muon_ogd.json \
  > logs/stage_b/math/eval_gsm8k_muon_ogd.log 2>&1 &
```

---

## 3) Stage C - Math (GSM8K)

Train from Stage B coding checkpoints.

### AdamW train

```bash
nohup python -u train_math_sft.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_adamw_after_medical \
  --output_dir outputs/sft_gsm8k_qwen1.5b_adamw_after_medical_coding \
  --num_train_examples 2000 \
  --max_length 512 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 5 \
  --lr 1e-5 \
  --max_steps 1000 \
  --val_ratio 0.02 \
  --val_every 100 \
  --save_strategy no \
  --seed 42 \
  > logs/stage_c/train/train_gsm8k_adamw.log 2>&1 &
```

### Muon-OGD train

```bash
nohup python -u train_math_sft_svd.py \
  --model_id outputs/sft_bigcodebench_qwen1.5b_muon_ogd_after_medical \
  --output_dir outputs/sft_gsm8k_qwen1.5b_muon_ogd_after_medical_coding \
  --num_train_examples 2000 \
  --max_length 512 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 5 \
  --lr 1e-5 \
  --max_steps 1000 \
  --save_strategy no \
  --seed 42 \
  --muon_ogd --muon_use_optimizer_class \
  --muon_k 3 --muon_T 1 --muon_eta 1e-4 --muon_eta_dual 1e-4 \
  --muon_warm_start --muon_layers o_proj,down_proj \
  --muon_momentum 0.95 \
  > logs/stage_c/train/train_gsm8k_muon_ogd.log 2>&1 &
```

### Stage C eval (math + medical + coding)

```bash
# Math
nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_gsm8k_qwen1.5b_adamw_after_medical_coding \
  --num_examples 500 --seed 42 \
  --out_file results/stage_c/math/gsm8k_adamw.json \
  > logs/stage_c/math/eval_gsm8k_adamw.log 2>&1 &

nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_gsm8k_qwen1.5b_muon_ogd_after_medical_coding \
  --num_examples 500 --seed 42 \
  --out_file results/stage_c/math/gsm8k_muon_ogd.json \
  > logs/stage_c/math/eval_gsm8k_muon_ogd.log 2>&1 &

# Medical retention
nohup python -u eval_huatuo_verifiable_api.py \
  --model_id outputs/sft_gsm8k_qwen1.5b_adamw_after_medical_coding \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_api_url https://api.openai.com/v1/chat/completions \
  --judge_model gpt-4o-mini \
  --out_file results/stage_c/medical/medical_adamw.json \
  > logs/stage_c/medical/eval_medical_adamw.log 2>&1 &

nohup python -u eval_huatuo_verifiable_api.py \
  --model_id outputs/sft_gsm8k_qwen1.5b_muon_ogd_after_medical_coding \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_api_url https://api.openai.com/v1/chat/completions \
  --judge_model gpt-4o-mini \
  --out_file results/stage_c/medical/medical_muon_ogd.json \
  > logs/stage_c/medical/eval_medical_muon_ogd.log 2>&1 &

# Coding retention
nohup python -u eval_HumanEval.py \
  --model_id outputs/sft_gsm8k_qwen1.5b_adamw_after_medical_coding \
  --dataset_id openai/openai_humaneval \
  --split test \
  --num_examples 164 \
  --seed 42 \
  --max_new_tokens 256 \
  --out_file results/stage_c/coding/humaneval_adamw.json \
  --samples_file results/stage_c/coding/humaneval_adamw_samples.jsonl \
  > logs/stage_c/coding/eval_humaneval_adamw.log 2>&1 &

nohup python -u eval_HumanEval.py \
  --model_id outputs/sft_gsm8k_qwen1.5b_muon_ogd_after_medical_coding \
  --dataset_id openai/openai_humaneval \
  --split test \
  --num_examples 164 \
  --seed 42 \
  --max_new_tokens 256 \
  --out_file results/stage_c/coding/humaneval_muon_ogd.json \
  --samples_file results/stage_c/coding/humaneval_muon_ogd_samples.jsonl \
  > logs/stage_c/coding/eval_humaneval_muon_ogd.log 2>&1 &
```

---

## Quick monitor commands

```bash
tail -f logs/stage_a/train/train_medical_adamw.log
tail -f logs/stage_a/train/train_medical_muon_ogd.log
tail -f logs/stage_b/train/train_bigcodebench_adamw.log
tail -f logs/stage_b/train/train_bigcodebench_muon_ogd.log
tail -f logs/stage_c/train/train_gsm8k_adamw.log
tail -f logs/stage_c/train/train_gsm8k_muon_ogd.log
```
