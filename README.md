# muon_CL

Minimal scripts for quick continual-learning pilots.

## Common gotcha

If you run:

* `python --model_id outputs/sft_bigcodebench ...`

you'll get `unknown option --model_id` because you didn't provide a script name.

Always include the script name:

* `python eval_gsm8k.py --model_id outputs/sft_bigcodebench ...`
* `python eval_bigcodebench_remote.py --model_id outputs/sft_bigcodebench ...`

## Examples

### GSM8K eval 

* `python eval_gsm8k.py --model_id Qwen/Qwen2.5-0.5B-Instruct --num_examples 1319 --seed 42 --out_file results/math/gsm8k_qwen0.5b_base`

 SFT on GSM8k
* 'python -u train_gsm8k_sft.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/sft_gsm8k_qwen0.5b \
  --num_train_examples 2000 \
  --max_length 512 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 1e-5 \
  --max_steps 1800 \
  --seed 42'

after GSM8K sft do eval
* `python eval_gsm8k.py --model_id outputs/sft_gsm8k_qwen0.5b --num_examples 1319 --seed 42 --out_file results/math/gsm8k_qwen0.5b_post_sft`



### BigCodeBench eval (remote)

Base model on hard tasks (all hard tasks in v0.1.4 => 148):

* `python eval_bigcodebench_remote.py --model_id outputs/sft_gsm8k_qwen0.5b  --num_tasks 150 --split instruct --subset hard --seed 42 --out_dir results/coding/bigcodebench_qwen0.5b_base_hard`



SFT on bigcodebench
* python -u train_bigcodebench_sft.py \
  --model_id outputs/sft_gsm8k_qwen0.5b \
  --output_dir outputs/sft_bigcodebench_qwen0.5b_nonhard_v2 \
  --split instruct \
  --task_ids_file results/coding/train_ids_nonhard.json \
  --max_length 1024 \
  --batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 1e-5 \
  --max_steps 500 \
  --seed 42


Eval Post-SFT on the same hard protocol:

* `python eval_bigcodebench_remote.py \
  --model_id outputs/sft_bigcodebench_qwen0.5b_nonhard_v2 \
  --num_tasks 150 \
  --split instruct \
  --subset hard \
  --seed 42 \
  --out_dir results/coding/bigcodebench_qwen0.5b_post_sft_hard`


after SFT on BigCodeBench eval on GSM8K
* `python eval_gsm8k.py --model_id outputs/sft_bigcodebench_qwen0.5b_nonhard_v2  --num_examples 1319 --seed 42 --out_file results/coding/after_coding_sft_gsm8k`

### Medical domain (HuatuoGPT-o1 style)

Train SFT (English-only, 20k by default):

* `python train_huatuo_sft.py --model_id outputs/sft_bigcodebench_qwen0.5b_nonhard_v2 --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT --dataset_config en --train_split train --num_train_examples 20000 --max_steps 1200 --output_dir outputs/sft_huatuo_qwen0.5b`

Evaluate with GPT judge (matches paper-style setting; requires OpenAI key):

* `export OPENAI_API_KEY="sk-..."`
* `python eval_huatuo_verifiable.py --model_id Qwen/Qwen2.5-0.5B-Instruct --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 4000 --progress_every 5 --judge_model gpt-5-mini --judge_max_retries 6 --judge_min_sleep 2.0 --judge_max_sleep 30 --out_file results/medical/huatuo_zero_shot_qwen0.5b_judge_n200.json`

Evaluate bigcodebench
python eval_bigcodebench_remote.py \
  --model_id outputs/sft_huatuo_qwen0.5b \
  --num_tasks 150 \
  --split instruct \
  --subset hard \
  --seed 42 \
  --out_dir results/coding/after_medical_sft_bigcodebench_hard
  

Evaluate gsm8k
python eval_gsm8k.py \
  --model_id outputs/sft_huatuo_qwen0.5b \
  --num_examples 1319 \
  --seed 42 \
  --out_file results/medical/after_medical_sft_gsm8k.json

  

If your dataset uses different field names, override:

* `--question_field ... --answer_field ... --language_field ... --verifiable_field ...`

## Push updates to GitHub

* `git status`
* `git add .`
* `git commit -m "Update training/eval scripts and results"`
* `git push`