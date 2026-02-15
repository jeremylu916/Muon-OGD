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

### GSM8K eval (100 examples)

* `python eval_gsm8k.py --model_id Qwen/Qwen2.5-0.5B-Instruct --num_examples 100 --seed 42`

### BigCodeBench eval (remote)

Baseline (samples tasks and saves them to results/bigcodebench/task_ids.json):

* `python eval_bigcodebench_remote.py --model_id Qwen/Qwen2.5-0.5B-Instruct --num_tasks 150 --split instruct --subset hard --seed 42`

Re-eval on the exact same tasks (after sft):

* `python eval_bigcodebench_remote.py --model_id outputs/sft_bigcodebench --split instruct --subset hard --task_ids_file results/bigcodebench/task_ids.json --num_tasks 150 --seed 42`

### BigCodeBench SFT (pilot)

* `python train_bigcodebench_sft.py --num_train_examples 128 --max_steps 200 --output_dir outputs/sft_bigcodebench`

### Medical domain (HuatuoGPT-o1 style)

Train SFT (English-only, 20k by default):

* `python train_huatuo_sft.py --model_id outputs/sft_bigcodebench_qwen1.5b --dataset_id FreedomIntelligence/medical-o1-reasoning-SFT --dataset_config en --train_split train --num_train_examples 20000 --max_steps 1200 --output_dir outputs/sft_huatuo_qwen1.5b`

Evaluate on verifiable-style subset (sample 1000):

* `python eval_huatuo_verifiable.py --model_id outputs/sft_huatuo_qwen1.5b --dataset_id FreedomIntelligence/medical-o1-verifiable-problem --dataset_config default --split train --num_examples 1000 --out_file results/medical/huatuo_qwen1.5b_eval.json`

If your dataset uses different field names, override:

* `--question_field ... --answer_field ... --language_field ... --verifiable_field ...`