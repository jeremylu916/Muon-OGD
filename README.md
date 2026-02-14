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

* `python eval_bigcodebench_remote.py --num_tasks 150 --split instruct --subset hard --seed 42`

Re-eval on the exact same tasks (after sft):

* `python eval_bigcodebench_remote.py --model_id outputs/sft_bigcodebench --split instruct --subset hard --task_ids_file results/bigcodebench/task_ids.json --num_tasks 150 --seed 42`

### BigCodeBench SFT (pilot)

* `python train_bigcodebench_sft.py --num_train_examples 128 --max_steps 200 --output_dir outputs/sft_bigcodebench`