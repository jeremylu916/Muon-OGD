# muon_CL

Minimal scripts for quick continual-learning pilots.

## Common gotcha

If you run:

* `python --model_id outputs/sft_bigcodebench ...`

you'll get `unknown option --model_id` because you didn't provide a script name.

Always include the script name:

* `python eval_gsm8k.py --model_id outputs/sft_bigcodebench ...`
* `python eval_bigcodebench_remote.py --model_id outputs/sft_bigcodebench ...`

## Canonical experiment protocol (recommended)

Use this section as the single source of truth for fair CL comparisons.

**Target budget:** `750 optimizer steps` per stage (batch size `1`, grad_accum `8`).

### Stage A — GSM8K

AdamW:

```bash
nohup python -u train_gsm8k_sft.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/sft_gsm8k_qwen0.5b_adam \
  --num_train_examples 2000 \
  --max_length 512 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 1e-5 \
  --max_steps 1000 \
  --save_strategy "steps" \
  --save_steps 250 \
  --seed 42 \
  > logs/train_gsm8k_adamw.log 2>&1 &
```

Muon-OGD:

```bash
nohup python -u train_gsm8k_sft_svd.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --ci_model_id Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/sft_gsm8k_qwen0.5b_muon_ogd \
  --num_train_examples 2000 \
  --max_length 512 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 1e-5 \
  --max_steps 1000 \
  --save_strategy "steps" \
  --save_steps 250 \
  --seed 42 \
  --muon_ogd --muon_use_optimizer_class \
  --muon_k 3 --muon_T 1 --muon_eta 1e-4 --muon_eta_dual 1e-4 \
  --muon_warm_start --muon_layers o_proj,down_proj \
  --muon_momentum 0.95 \
  > logs/train_gsm8k_muon_ogd.log 2>&1 &
```

Post-SFT eval (all domains, Stage A checkpoints):

```bash
# Math (GSM8K)
nohup python -u eval_gsm8k.py --model_id outputs/sft_gsm8k_qwen0.5b_adam --num_examples 500 --seed 42 --out_file results/stage_a/gsm8k_adam_post_sft.json > logs/eval_stage_a_gsm8k_adam.log 2>&1 &
nohup python -u eval_gsm8k.py --model_id outputs/sft_gsm8k_qwen0.5b_muon_ogd --num_examples 500 --seed 42 --out_file results/stage_a/gsm8k_muon_post_sft.json > logs/eval_stage_a_gsm8k_muon.log 2>&1 &

# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py --model_id outputs/sft_gsm8k_qwen0.5b_adam --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/stage_a/bcb_hard_adam_post_sft > logs/eval_stage_a_bcb_adam.log 2>&1 &
nohup python -u eval_bigcodebench_remote.py --model_id outputs/sft_gsm8k_qwen0.5b_muon_ogd --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/stage_a/bcb_hard_muon_post_sft > logs/eval_stage_a_bcb_muon.log 2>&1 &

# Medical (Huatuo verifiable, API judge)
nohup python -u eval_huatuo_verifiable_api.py --model_id outputs/sft_gsm8k_qwen0.5b_adam --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 500 --seed 42 --progress_every 20 --judge_api_url https://api.openai.com/v1/chat/completions --judge_model gpt-4o-mini --out_file results/stage_a/medical_adam_post_sft.json > logs/eval_stage_a_medical_adam.log 2>&1 &
nohup python -u eval_huatuo_verifiable_api.py --model_id outputs/sft_gsm8k_qwen0.5b_muon_ogd --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 500 --seed 42 --progress_every 20 --judge_api_url https://api.openai.com/v1/chat/completions --judge_model gpt-4o-mini --out_file results/stage_a/medical_muon_post_sft.json > logs/eval_stage_a_medical_muon.log 2>&1 &
```

### Stage B — BigCodeBench

`train_ids_nonhard.json` is small, so keep higher epochs to reach `max_steps`.

AdamW:

```bash
nohup python -u train_bigcodebench_sft.py \
  --model_id outputs/sft_gsm8k_qwen0.5b_adam \
  --output_dir outputs/sft_bigcodebench_qwen0.5b_adamw \
  --split instruct \
  --task_ids_file results/coding/train_ids_nonhard.json \
  --max_length 1024 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 24 \
  --lr 1e-5 \
  --max_steps 1000 \
  --save_strategy "steps" \
  --save_steps 250 \
  --seed 42 \
  > logs/train_bigcodebench_adamw.log 2>&1 &
```

Muon-OGD:

```bash
nohup python -u train_bigcodebench_svd_sft.py \
  --model_id outputs/sft_gsm8k_qwen0.5b_muon_ogd \
  --ci_model_id outputs/sft_gsm8k_qwen0.5b_muon_ogd \
  --output_dir outputs/sft_bigcodebench_qwen0.5b_muon_ogd \
  --split instruct \
  --task_ids_file results/coding/train_ids_nonhard.json \
  --max_length 1024 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 24 \
  --lr 1e-5 \
  --max_steps 1000 \
  --save_strategy "steps" \
  --save_steps 250 \
  --seed 42 \
  --muon_ogd --muon_use_optimizer_class \
  --muon_k 3 --muon_T 1 --muon_eta 1e-4 --muon_eta_dual 1e-4 \
  --muon_warm_start --muon_layers o_proj,down_proj \
  --muon_momentum 0.95 \
  > logs/train_bigcodebench_muon_ogd.log 2>&1 &
```

Post-SFT eval (all domains, Stage B checkpoints):

```bash
# Math (GSM8K)
nohup python -u eval_gsm8k.py --model_id outputs/sft_bigcodebench_qwen0.5b_adamw --num_examples 500 --seed 42 --out_file results/stage_b/gsm8k_adam_post_sft.json > logs/eval_stage_b_gsm8k_adam.log 2>&1 &
nohup python -u eval_gsm8k.py --model_id outputs/sft_bigcodebench_qwen0.5b_muon_ogd --num_examples 500 --seed 42 --out_file results/stage_b/gsm8k_muon_post_sft.json > logs/eval_stage_b_gsm8k_muon.log 2>&1 &

# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py --model_id outputs/sft_bigcodebench_qwen0.5b_adamw --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/stage_b/bcb_hard_adam_post_sft > logs/eval_stage_b_bcb_adam.log 2>&1 &
nohup python -u eval_bigcodebench_remote.py --model_id outputs/sft_bigcodebench_qwen0.5b_muon_ogd --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/stage_b/bcb_hard_muon_post_sft > logs/eval_stage_b_bcb_muon.log 2>&1 &

# Medical (Huatuo verifiable, API judge)
nohup python -u eval_huatuo_verifiable_api.py --model_id outputs/sft_bigcodebench_qwen0.5b_adamw --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 500 --seed 42 --progress_every 20 --judge_api_url https://api.openai.com/v1/chat/completions --judge_model gpt-4o-mini --out_file results/stage_b/medical_adam_post_sft.json > logs/eval_stage_b_medical_adam.log 2>&1 &
nohup python -u eval_huatuo_verifiable_api.py --model_id outputs/sft_bigcodebench_qwen0.5b_muon_ogd --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 500 --seed 42 --progress_every 20 --judge_api_url https://api.openai.com/v1/chat/completions --judge_model gpt-4o-mini --out_file results/stage_b/medical_muon_post_sft.json > logs/eval_stage_b_medical_muon.log 2>&1 &
```

### Stage C — Huatuo

AdamW:

```bash
nohup python -u train_huatuo_sft.py \
  --model_id outputs/sft_bigcodebench_qwen0.5b_adamw \
  --output_dir outputs/sft_huatuo_qwen0.5b_adamw \
  --max_length 1536 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --lr 2e-5 \
  --max_steps 1000 \
  --save_strategy "steps" \
  --save_steps 250 \
  --seed 42 \
  > logs/train_huatuo_adamw.log 2>&1 &
```

Muon-OGD:

```bash
nohup python -u train_huatuo_sft_svd.py \
  --model_id outputs/sft_bigcodebench_qwen0.5b_muon_ogd \
  --ci_model_ids outputs/sft_gsm8k_qwen0.5b_muon_ogd,outputs/sft_bigcodebench_qwen0.5b_muon_ogd \
  --ci_k_per_source 2 \
  --output_dir outputs/sft_huatuo_qwen0.5b_muon_ogd \
  --max_length 1536 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 1 \
  --lr 2e-5 \
  --max_steps 1000 \
  --save_strategy "steps" \
  --save_steps 250 \
  --seed 42 \
  --muon_ogd --muon_use_optimizer_class \
  --muon_T 1 --muon_eta 1e-4 --muon_eta_dual 1e-4 \
  --muon_warm_start --muon_layers o_proj,down_proj \
  --muon_momentum 0.95 \
  > logs/train_huatuo_muon_ogd.log 2>&1 &
```

Post-SFT eval (all domains, Stage C checkpoints):

```bash
# Math (GSM8K)
nohup python -u eval_gsm8k.py --model_id outputs/sft_huatuo_qwen0.5b_adamw --num_examples 500 --seed 42 --out_file results/stage_c/gsm8k_adam_post_sft.json > logs/eval_stage_c_gsm8k_adam.log 2>&1 &
nohup python -u eval_gsm8k.py --model_id outputs/sft_huatuo_qwen0.5b_muon_ogd --num_examples 500 --seed 42 --out_file results/stage_c/gsm8k_muon_post_sft.json > logs/eval_stage_c_gsm8k_muon.log 2>&1 &

# Coding (BigCodeBench hard)
nohup python -u eval_bigcodebench_remote.py --model_id outputs/sft_huatuo_qwen0.5b_adamw --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/stage_c/bcb_hard_adam_post_sft > logs/eval_stage_c_bcb_adam.log 2>&1 &
nohup python -u eval_bigcodebench_remote.py --model_id outputs/sft_huatuo_qwen0.5b_muon_ogd --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/stage_c/bcb_hard_muon_post_sft > logs/eval_stage_c_bcb_muon.log 2>&1 &

# Medical (Huatuo verifiable, API judge)
nohup python -u eval_huatuo_verifiable_api.py --model_id outputs/sft_huatuo_qwen0.5b_adamw --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 500 --seed 42 --progress_every 20 --judge_api_url https://api.openai.com/v1/chat/completions --judge_model gpt-4o-mini --out_file results/stage_c/medical_adam_post_sft.json > logs/eval_stage_c_medical_adam.log 2>&1 &
nohup python -u eval_huatuo_verifiable_api.py --model_id outputs/sft_huatuo_qwen0.5b_muon_ogd --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 500 --seed 42 --progress_every 20 --judge_api_url https://api.openai.com/v1/chat/completions --judge_model gpt-4o-mini --out_file results/stage_c/medical_muon_post_sft.json > logs/eval_stage_c_medical_muon.log 2>&1 &
```

If your dataset uses different field names, override:

* `--question_field ... --answer_field ... --language_field ... --verifiable_field ...`

## Legacy / Archive

Previous command blocks mixed older defaults, inconsistent model names, and non-matched step budgets. Use the canonical section above for current runs.

## Push updates to GitHub

* `git status`
* `git add .`
* `git commit -m "Update training/eval scripts and results"`
* `git push`
