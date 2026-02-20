import argparse
import ast
import json
import os
import random
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import torch
from datasets import load_dataset
from gradio_client import Client, handle_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BCB_SPLIT = "instruct"   # bigcodebench split: instruct|complete
DEFAULT_BCB_SUBSET = "hard"      # bigcodebench subset: hard|full
DEFAULT_BCB_VERSION = "v0.1.4"   # dataset version on HF hub
DEFAULT_NUM_TASKS = 10
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0 # Greedy for pass@1
DEFAULT_NUM_CANDIDATES = 1
DEFAULT_OUT_DIR = "results/bigcodebench"
DEFAULT_GRADIO_ENDPOINT = "https://bigcode-bigcodebench-evaluator.hf.space/"

# Regex for markdown code blocks
_CODEBLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)

def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

def extract_code(text: str) -> str:
    # 1. Try Markdown
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    t = text.strip()

    # 2. Handle unclosed markdown (common with max_new_tokens limit)
    if t.startswith("```"):
        t = re.sub(r"^```(?:python)?\s*", "", t, flags=re.IGNORECASE)
        t = t.replace("```", "").strip()

    # 3. Fallback: Find first line that looks like Python code start.
    # Use line-start anchors to avoid false matches like "from user" in comments/text.
    candidates = []
    for pat in [r"(?m)^\s*def\s+task_func\s*\(", r"(?m)^\s*import\s+", r"(?m)^\s*from\s+"]:
        m = re.search(pat, t)
        if m:
            candidates.append(m.start())
    if candidates:
        t = t[min(candidates):]

    # 4. Aggressive cleanup of trailing chat
    # If the model adds text after the code, it breaks execution.
    # We look for common "end of code" markers or just double newlines followed by text
    stoppers = [
        "\n\nExplanation", "\n\nHere is", "\n\nThis code", 
        "\n# Example", "\nprint(", "\nif __name__"
    ]
    for stopper in stoppers:
        pos = t.find(stopper)
        if pos >= 0:
            t = t[:pos].rstrip()

    return t

def _extract_task_func_signature(task_prompt: str) -> Optional[str]:
    m = re.search(r"def\s+task_func\s*\((.*?)\)\s*:", task_prompt, flags=re.DOTALL)
    if m:
        args = m.group(1).strip()
        return f"def task_func({args}):"
    return None

def normalize_solution(task_prompt: str, code: str) -> str:
    code = code.rstrip()
    if re.search(r"^\s*def\s+task_func\s*\(", code, flags=re.MULTILINE):
        return code
    
    sig = _extract_task_func_signature(task_prompt)
    if not sig:
        return code
        
    body = textwrap.dedent(code).strip("\n")
    if not body:
        body = "pass"
    return sig + "\n" + textwrap.indent(body, "    ")

def build_instruct_prompt(task_prompt: str) -> str:
    # Matches the SFT training script format
    return (
        "Write Python code that solves the task. "
        "Output ONLY valid Python code (no markdown, no explanation).\n\n"
        + task_prompt.strip()
    )

@dataclass
class Task:
    task_id: str
    prompt: str

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--bcb_version", type=str, default=DEFAULT_BCB_VERSION)
    p.add_argument("--split", type=str, default=DEFAULT_BCB_SPLIT)
    p.add_argument("--subset", type=str, default=DEFAULT_BCB_SUBSET)
    p.add_argument("--num_tasks", type=int, default=DEFAULT_NUM_TASKS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--num_candidates", type=int, default=DEFAULT_NUM_CANDIDATES)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--gradio_endpoint", type=str, default=DEFAULT_GRADIO_ENDPOINT)
    p.add_argument("--task_ids", type=str, default="")
    p.add_argument("--task_ids_file", type=str, default="")
    return p.parse_args()

def select_tasks(ds, split: str, num_tasks: int, seed: int, task_ids_csv: str) -> List[Task]:
    if task_ids_csv.strip():
        wanted = [t.strip() for t in task_ids_csv.split(",") if t.strip()]
        wanted_set = set(wanted)
        by_id = {ex["task_id"]: ex for ex in ds if ex["task_id"] in wanted_set}
        tasks = []
        for tid in wanted:
            if tid in by_id:
                ex = by_id[tid]
                tasks.append(Task(task_id=tid, prompt=ex[f"{split}_prompt"]))
        return tasks

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_tasks]
    tasks = []
    for i in indices:
        ex = ds[i]
        tasks.append(Task(task_id=ex["task_id"], prompt=ex[f"{split}_prompt"]))
    return tasks

def task_id_to_index(task_id: str) -> str:
    if "/" not in task_id: return task_id
    return task_id.split("/", 1)[1]

def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()
    os.makedirs(args.out_dir, exist_ok=True)
    samples_path = os.path.join(args.out_dir, "samples.jsonl")

    # Load Dataset
    print(f"Loading BigCodeBench {args.subset} subset...")
    ds = load_dataset("bigcode/bigcodebench", split=args.bcb_version, cache_dir=cache_dir)
    
    if args.subset == "hard":
        try:
            from bigcodebench.data import get_bigcodebench
            hard_ids = set(get_bigcodebench(subset="hard").keys())
            ds = ds.filter(lambda ex: ex["task_id"] in hard_ids)
        except Exception:
            print("Warning: bigcodebench module not found or failed. Falling back to full dataset.")

    # Select Tasks
    if args.task_ids_file.strip():
        with open(args.task_ids_file, "r") as f:
            task_ids_list = json.load(f)
        tasks = select_tasks(ds, args.split, args.num_tasks, args.seed, ",".join(task_ids_list))
    else:
        tasks = select_tasks(ds, args.split, args.num_tasks, args.seed, args.task_ids)
    
    task_indices = [task_id_to_index(t.task_id) for t in tasks]

    # Load Model
    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    # Generate
    print(f"Generating solutions for {len(tasks)} tasks...")
    results_list = []
    
    with open(samples_path, "w") as f:
        for n, task in enumerate(tasks, start=1):
            if args.split == "instruct":
                user_prompt = build_instruct_prompt(task.prompt)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                full_prompt = task.prompt

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs, 
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=0.95 if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            prompt_len = inputs["input_ids"].shape[1]
            raw_output = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            
            # --- DEBUG: Print first sample to check format ---
            if n == 1:
                print("\n" + "="*40)
                print(f"DEBUG SAMPLE (Task: {task.task_id})")
                print("-" * 20)
                print(f"PROMPT:\n{full_prompt}")
                print("-" * 20)
                print(f"RAW OUTPUT:\n{raw_output}")
                print("-" * 20)
                extracted = extract_code(raw_output)
                print(f"EXTRACTED CODE:\n{extracted}")
                print("="*40 + "\n")
            # -----------------------------------------------

            extracted = extract_code(raw_output)
            final_code = normalize_solution(task.prompt, extracted)
            
            # Use 'completion' key for compatibility with standard evaluators
            record = {"task_id": task.task_id, "completion": final_code}
            f.write(json.dumps(record) + "\n")
            
            if n % 10 == 0:
                print(f"Generated {n}/{len(tasks)}")

    # Evaluate
    print("Submitting to Remote Evaluator...")
    try:
        client = Client(args.gradio_endpoint)
        results, pass_at_k = client.predict(
            split=args.split,
            subset=args.subset,
            samples=handle_file(samples_path),
            pass_k="1",
            parallel=1,
            min_time_limit=1,
            max_as_limit=30 * 1024,
            max_data_limit=30 * 1024,
            max_stack_limit=10,
            calibrated=False,
            check_gt_only=False,
            no_gt=False,
            selective_evaluate=",".join(task_indices),
            api_name="/predict",
        )
        print("Results received.")
        
        with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(args.out_dir, "pass_at_k.json"), "w") as f:
            json.dump(pass_at_k, f, indent=2)
            
        print(f"pass@1: {pass_at_k.get('pass@1')}")
        
    except Exception as e:
        print(f"\nEvaluation Failed: {e}")
        print("Check if the generated file 'samples.jsonl' looks correct.")

if __name__ == "__main__":
    main()