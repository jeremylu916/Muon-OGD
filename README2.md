# README2: Continual Learning Order = Coding -> Math -> Medical

This protocol uses the sequence:

1. Stage A: Coding (BigCodeBench)
2. Stage B: MFreedomIntelligence/medical-o1-verifiable-problem(GSM8K)
3. Stage C: Medical (Huatuo)

All logs in this README go to `logs_seq2/`.
All metrics/results in this README go to `results_seq2/`.

## 0) Setup

```bash
mkdir -p \
  logs_seq2/stage_a/train logs_seq2/stage_a/coding logs_seq2/stage_a/math logs_seq2/stage_a/medical \
  logs_seq2/stage_b/train logs_seq2/stage_b/coding logs_seq2/stage_b/math logs_seq2/stage_b/medical \
  logs_seq2/stage_c/train logs_seq2/stage_c/coding logs_seq2/stage_c/math logs_seq2/stage_c/medical \
    logs/base_model results/base_model logs_seq2/base_model results_seq2/base_model \
  results_seq2/stage_a/coding results_seq2/stage_a/math results_seq2/stage_a/medical \
  results_seq2/stage_b/coding results_seq2/stage_b/math results_seq2/stage_b/medical \
  results_seq2/stage_c/coding results_seq2/stage_c/math results_seq2/stage_c/medical
```

For medical eval with `eval_medical.py`:

```bash
# No OpenAI key is needed. The script uses local verifier model:
# FreedomIntelligence/medical_o1_verifier_3B
```

---

## 0.5) Base Model Eval (Qwen2.5-1.5B)

```bash
# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --num_tasks 0 --split complete --subset full --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/base_model/bcb_hard_qwen2.5_1p5b_base \
  --no-debug_first_sample \
  > logs_seq2/base_model/eval_bcb_qwen2.5_1p5b_base.log 2>&1 &


# Math (GSM8K)
nohup python -u eval_gsm8k.py \
  --model_id Qwen/Qwen2.5-1.5B\
  --num_examples 500 --seed 42 \
  --out_file results_seq2/base_model/gsm8k_qwen2.5_1p5b.json \
  > logs_seq2/base_model/eval_gsm8k_qwen2.5_1p5b.log 2>&1 &

# Medical (Huatuo verifiable)

nohup python -u eval_medical.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default \
  --split train \
  --num_examples 500 \
  --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/base_model/medical_qwen2.5_1p5b.json \
  > logs_seq2/base_model/eval_medical_qwen2.5_1p5b.log 2>&1 &
```

---

## 1) Stage A - Coding (BigCodeBench)

### AdamW train

```bash
nohup python -u train_coding_bigcodebench_sft.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --split complete \
  --output_dir outputs/sft_seq2_coding_qwen1.5b_adamw \
  --max_length 2048 \
  --lr 2e-6\
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --max_steps 0 \
  --save_strategy no \
  --seed 42 \
  > logs_seq2/stage_a/train/train_coding_adamw.log 2>&1 &
```



### Muon-OGD train

```bash
nohup python -u train_coding_bigcodebench_muon_ogd_sft.py \
  --model_id Qwen/Qwen2.5-1.5B \
  --ci_model_id Qwen/Qwen2.5-1.5B \
  --output_dir outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --split complete \
  --max_length 2048 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --max_steps 0 \
  --save_strategy no \
  --seed 42 \
  --muon_ogd --muon_use_optimizer_class \
  --muon_layers gate_proj,up_proj,down_proj\
  --muon_k 10 --muon_T 1 --muon_eta 1e-5 --muon_eta_dual 1e-5 \
  --muon_warm_start --muon_momentum 0.95 \
  > logs_seq2/stage_a/train/train_coding_muon_ogd.log 2>&1 &
```

### Stage A eval (coding + math + medical)

```bash
# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_adamw \
  --num_tasks 0 --split complete --subset full --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/stage_a/coding/bcb_hard_adamw \
  > logs_seq2/stage_a/coding/eval_bcb_adamw.log 2>&1 &

nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --num_tasks 0 --split complete --subset full --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/stage_a/coding/bcb_hard_muon \
  > logs_seq2/stage_a/coding/eval_bcb_muon.log 2>&1 &

# Math (GSM8K)
nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_adamw \
  --num_examples 500 --seed 42 \
  --out_file results_seq2/stage_a/math/gsm8k_adamw.json \
  > logs_seq2/stage_a/math/eval_gsm8k_adamw.log 2>&1 &

nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --num_examples 500 --seed 42 \
  --out_file results_seq2/stage_a/math/gsm8k_muon_ogd.json \
  > logs_seq2/stage_a/math/eval_gsm8k_muon_ogd.log 2>&1 &

# Medical (Huatuo verifiable)
nohup python -u eval_medical.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_adamw  \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default --split train --num_examples 500 --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/stage_a/medical/medical_adamw.json  \
  > logs_seq2/stage_a/medical/eval_medical_adamw.log 2>&1 &

nohup python -u eval_medical.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default --split train --num_examples 500 --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/stage_a/medical/medical_muon_ogd.json \
  > logs_seq2/stage_a/medical/eval_medical_muon_ogd.log 2>&1 &
```

---

## 2) Stage B - Math (GSM8K, initialized from Stage A)

### AdamW train (from Stage A AdamW)

```bash
nohup python -u train_math_sft.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_adamw \
  --output_dir outputs/sft_seq2_math_qwen1.5b_adamw_from_coding \
  --num_train_examples 2000 --max_length 512 --batch_size 1 --grad_accum 8 \
  --epochs 4 --lr 2e-5 --max_steps 1000 \
  --val_ratio 0.02 --val_every 100 --val_max_batches 32 \
  --probe_every 100 --probe_max_new_tokens 64 \
  --save_strategy no --seed 42 \
  > logs_seq2/stage_b/train/train_math_adamw_from_coding.log 2>&1 &
```

### Muon-OGD train (from Stage A Muon)

```bash
nohup python -u train_math_sft_svd.py \
  --model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --ci_model_id outputs/sft_seq2_coding_qwen1.5b_muon_ogd \
  --output_dir outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --num_train_examples 2000 --max_length 512 --batch_size 1 --grad_accum 8 \
  --epochs 4 --lr 2e-5 --max_steps 1000 \
  --val_ratio 0.02 --val_every 100 --val_max_batches 32 \
  --probe_every 100 --probe_max_new_tokens 64 \
  --save_strategy no --seed 42 \
  --muon_ogd --muon_use_optimizer_class \
  --muon_k 3 --muon_T 1 --muon_eta 1e-4 --muon_eta_dual 1e-4 \
  --muon_warm_start --muon_layers gate_proj,up_proj,down_proj --muon_momentum 0.95 \
  > logs_seq2/stage_b/train/train_math_muon_ogd_from_coding.log 2>&1 &
```

### Stage B eval (coding + math + medical)

```bash
# Math (GSM8K)
nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_adamw_from_coding \
  --num_examples 500 --seed 42 \
  --out_file results_seq2/stage_b/math/gsm8k_adamw_from_coding.json \
  > logs_seq2/stage_b/math/eval_gsm8k_adamw_from_coding.log 2>&1 &

nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --num_examples 500 --seed 42 \
  --out_file results_seq2/stage_b/math/gsm8k_muon_ogd_from_coding.json \
  > logs_seq2/stage_b/math/eval_gsm8k_muon_ogd_from_coding.log 2>&1 &

# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_adamw_from_coding \
  --num_tasks 0 --split instruct --subset hard --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/stage_b/coding/bcb_hard_adamw_from_coding_math \
  > logs_seq2/stage_b/coding/eval_bcb_adamw_from_coding_math.log 2>&1 &

nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --num_tasks 0 --split instruct --subset hard --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/stage_b/coding/bcb_hard_muon_from_coding_math \
  > logs_seq2/stage_b/coding/eval_bcb_muon_from_coding_math.log 2>&1 &

# Medical (Huatuo verifiable)
nohup python -u eval_medical.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_adamw_from_coding \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default --split train --num_examples 100 --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/stage_b/medical/medical_adamw_from_coding_math.json \
  > logs_seq2/stage_b/medical/eval_medical_adamw_from_coding_math.log 2>&1 &

nohup python -u eval_medical.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default --split train --num_examples 100 --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/stage_b/medical/medical_muon_ogd_from_coding_math.json \
  > logs_seq2/stage_b/medical/eval_medical_muon_ogd_from_coding_math.log 2>&1 &
```

---

## 3) Stage C - Medical 

### AdamW train (from Stage B AdamW)

```bash
nohup python -u train_medical_sft.py\
  --model_id outputs/sft_seq2_math_qwen1.5b_adamw_from_coding \
  --output_dir outputs/sft_seq2_medical_qwen1.5b_adamw_from_coding_math \
  --max_length 2048 --batch_size 1 --grad_accum 8 \
  --epochs 1\
  --max_steps 0 \
  --lr 5e-6 \
  --no-answer_after_cot_only \
  --seed 42 \
  > logs_seq2/stage_c/train/train_medical_adamw_from_coding_math.log 2>&1 &
```

### Muon-OGD train (from Stage B Muon)

```bash
nohup python -u train_MedQuad_sft.py \
  --model_id outputs/sft_seq2_math_qwen1.5b_muon_ogd_from_coding \
  --output_dir outputs/sft_seq2_medical_qwen1.5b_muon_ogd_from_coding_math \
  --max_length 1536 --batch_size 1 --grad_accum 8 \
  --epochs 1 --lr 2e-5 \
  --seed 42 \
  > logs_seq2/stage_c/train/train_medical_muon_ogd_from_coding_math.log 2>&1 &
```

### Stage C eval (coding + math + medical)

```bash
# Medical (Huatuo verifiable)
nohup python -u eval_medical.py \
  --model_id outputs/sft_seq2_medical_qwen1.5b_adamw_from_coding_math \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default --split train --num_examples 500 --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/stage_c/medical/medical_adamw_final.json \
  > logs_seq2/stage_c/medical/eval_medical_adamw_final.log 2>&1 &

nohup python -u eval_medical.py \
  --model_id outputs/sft_seq2_medical_qwen1.5b_muon_ogd_from_coding_math \
  --verifier_model_id FreedomIntelligence/medical_o1_verifier_3B \
  --dataset_id FreedomIntelligence/medical-o1-verifiable-problem \
  --dataset_config default --split train --num_examples 500 --seed 42 \
  --progress_every 20 \
  --judge_final_answer_only \
  --out_file results_seq2/stage_c/medical/medical_muon_ogd_final.json \
  > logs_seq2/stage_c/medical/eval_medical_muon_ogd_final.log 2>&1 &

# Math (GSM8K)
nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_seq2_medical_qwen1.5b_adamw_from_coding_math \
  --num_examples 500 --seed 42 \
  --out_file results_seq2/stage_c/math/gsm8k_adamw_final.json \
  > logs_seq2/stage_c/math/eval_gsm8k_adamw_final.log 2>&1 &

nohup python -u eval_gsm8k.py \
  --model_id outputs/sft_seq2_medical_qwen1.5b_muon_ogd_from_coding_math \
  --num_examples 500 --seed 42 \
  --out_file results_seq2/stage_c/math/gsm8k_muon_ogd_final.json \
  > logs_seq2/stage_c/math/eval_gsm8k_muon_ogd_final.log 2>&1 &

# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_seq2_medical_qwen1.5b_adamw_from_coding_math \
  --num_tasks 0 --split instruct --subset hard --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/stage_c/coding/bcb_hard_adamw_final \
  > logs_seq2/stage_c/coding/eval_bcb_adamw_final.log 2>&1 &

nohup python -u eval_bigcodebench_remote.py \
  --model_id outputs/sft_seq2_medical_qwen1.5b_muon_ogd_from_coding_math \
  --num_tasks 0 --split instruct --subset hard --seed 42 \
  --use_rest_split --train_size 800 \
  --out_dir results_seq2/stage_c/coding/bcb_hard_muon_final \
  > logs_seq2/stage_c/coding/eval_bcb_muon_final.log 2>&1 &
```

---

## Quick monitor commands

```bash
tail -f logs_seq2/stage_a/train/train_coding_adamw.log
tail -f logs_seq2/stage_a/train/train_coding_muon_ogd.log
tail -f logs_seq2/stage_b/train/train_math_adamw_from_coding.log
tail -f logs_seq2/stage_b/train/train_math_muon_ogd_from_coding.log
tail -f logs_seq2/stage_c/train/train_medical_adamw_from_coding_math.log
tail -f logs_seq2/stage_c/train/train_medical_muon_ogd_from_coding_math.log
```