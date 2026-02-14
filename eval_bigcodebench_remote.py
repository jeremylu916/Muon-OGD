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


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BCB_SPLIT = "instruct"   # bigcodebench split: instruct|complete
DEFAULT_BCB_SUBSET = "hard"      # bigcodebench subset: hard|full
DEFAULT_BCB_VERSION = "v0.1.4"   # dataset version on HF hub
DEFAULT_NUM_TASKS = 10
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_CANDIDATES = 1
DEFAULT_OUT_DIR = "results/bigcodebench"
DEFAULT_GRADIO_ENDPOINT = "https://bigcode-bigcodebench-evaluator.hf.space/"


_CODEBLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with best-effort compatibility flags.

    Some Transformers versions emit a warning about an incorrect regex pattern
    for certain tokenizers. Newer versions support fix_mistral_regex=True.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def extract_code(text: str) -> str:
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    t = text.strip()

    # Handle incomplete markdown fences such as "```python\n..." without closing fence.
    if t.startswith("```"):
        t = re.sub(r"^```(?:python)?\s*", "", t, flags=re.IGNORECASE)
        t = t.replace("```", "").strip()

    # If the model adds explanation before code, keep content starting at first def/import/class.
    anchors = [
        t.find("def task_func"),
        t.find("def "),
        t.find("import "),
        t.find("from "),
        t.find("class "),
    ]
    anchors = [a for a in anchors if a >= 0]
    if anchors:
        t = t[min(anchors):]

    # Trim obvious assistant chatter tails.
    for stopper in ("\n\nExplanation", "\n\nHere", "\n\nThis code"):
        pos = t.find(stopper)
        if pos >= 0:
            t = t[:pos].rstrip()

    return t


def _extract_task_func_signature(task_prompt: str) -> Optional[str]:
    """Best-effort extraction of `def task_func(...):` from the benchmark prompt."""
    # Common case: prompt explicitly includes function signature.
    m = re.search(r"def\s+task_func\s*\((.*?)\)\s*:", task_prompt, flags=re.DOTALL)
    if m:
        args = m.group(1).strip()
        return f"def task_func({args}):"
    return None


def normalize_solution(task_prompt: str, code: str) -> str:
    """Normalize model output into executable Python for BigCodeBench.

    If a model emits only an indented function body (common failure mode),
    reconstruct a top-level `def task_func(...)` wrapper from the prompt.
    """
    code = code.rstrip()

    # Already has an explicit task entry-point.
    if re.search(r"^\s*def\s+task_func\s*\(", code, flags=re.MULTILINE):
        return code

    sig = _extract_task_func_signature(task_prompt)
    if not sig:
        return code

    # Dedent body so syntax is valid after wrapping.
    body = textwrap.dedent(code).strip("\n")
    if not body:
        body = "pass"

    return sig + "\n" + textwrap.indent(body, "    ")


def build_instruct_prompt(task_prompt: str) -> str:
    # BigCodeBench-Instruct expects a Python solution defining the required entry point.
    return (
        "Write Python code that solves the task. "
        "Output ONLY valid Python code (no markdown, no explanation).\n\n"
        + task_prompt.strip()
        + "\n"
    )


@dataclass
class Task:
    task_id: str
    prompt: str


def parse_args():
    p = argparse.ArgumentParser(description="Generate BigCodeBench solutions and evaluate remotely (Gradio endpoint).")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--bcb_version", type=str, default=DEFAULT_BCB_VERSION)
    p.add_argument("--split", type=str, default=DEFAULT_BCB_SPLIT, choices=["instruct", "complete"])
    p.add_argument("--subset", type=str, default=DEFAULT_BCB_SUBSET, choices=["hard", "full"])
    p.add_argument("--num_tasks", type=int, default=DEFAULT_NUM_TASKS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument(
        "--num_candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help="Generate N candidates per task and keep the best heuristic candidate.",
    )
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--gradio_endpoint", type=str, default=DEFAULT_GRADIO_ENDPOINT)
    p.add_argument(
        "--task_ids",
        type=str,
        default="",
        help="Comma-separated BigCodeBench task IDs (e.g., BigCodeBench/0,BigCodeBench/1). If empty, sample randomly.",
    )
    p.add_argument(
        "--task_ids_file",
        type=str,
        default="",
        help="Path to a JSON file containing a list of task_id strings. Overrides --task_ids if provided.",
    )
    return p.parse_args()


def select_tasks(ds, split: str, num_tasks: int, seed: int, task_ids_csv: str) -> List[Task]:
    if task_ids_csv.strip():
        wanted = [t.strip() for t in task_ids_csv.split(",") if t.strip()]
        wanted_set = set(wanted)
        by_id = {ex["task_id"]: ex for ex in ds if ex["task_id"] in wanted_set}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            raise ValueError(f"Task IDs not found in dataset: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        tasks = []
        for tid in wanted:
            ex = by_id[tid]
            prompt = ex[f"{split}_prompt"]
            tasks.append(Task(task_id=tid, prompt=prompt))
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
    """Convert "BigCodeBench/123" -> "123" for BigCodeBench's selective_evaluate."""
    if "/" not in task_id:
        return task_id
    return task_id.split("/", 1)[1]


def score_candidate(code: str) -> int:
    score = 0
    if code.strip():
        score += 1
    if "def task_func" in code:
        score += 3
    try:
        ast.parse(code)
        score += 2
    except Exception:
        pass
    if "```" not in code:
        score += 1
    return score


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples_path = os.path.join(args.out_dir, "samples.jsonl")

    # Load BigCodeBench tasks (prompts only).
    ds = load_dataset("bigcode/bigcodebench", split=args.bcb_version)

    # If evaluating the "hard" subset, restrict sampling to tasks that exist in that subset.
    # The HF dataset doesn't provide a "subset" column, so we try to get the hard task IDs
    # from the BigCodeBench python package (if installed). If that fails, fall back to "full".
    if args.subset == "hard":
        try:
            from bigcodebench.data import get_bigcodebench

            hard_ids = set(get_bigcodebench(subset="hard").keys())
            ds = ds.filter(lambda ex: ex["task_id"] in hard_ids)
        except Exception:
            print("Warning: could not load hard subset task IDs; falling back to subset=full")
            args.subset = "full"

    if args.task_ids_file.strip():
        with open(args.task_ids_file, "r") as f:
            task_ids_list = json.load(f)
        if not isinstance(task_ids_list, list) or not all(isinstance(x, str) for x in task_ids_list):
            raise ValueError("--task_ids_file must contain a JSON list of task_id strings")
        tasks = select_tasks(ds, args.split, args.num_tasks, args.seed, ",".join(task_ids_list))
    else:
        tasks = select_tasks(ds, args.split, args.num_tasks, args.seed, args.task_ids)
    task_ids = [t.task_id for t in tasks]
    task_indices = [task_id_to_index(tid) for tid in task_ids]

    with open(os.path.join(args.out_dir, "task_ids.json"), "w") as f:
        json.dump(task_ids, f, indent=2)

    tokenizer = load_tokenizer(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    # Normalize generation config for deterministic mode to avoid warning noise.
    # Some checkpoints carry sampling params in generation_config even when do_sample=False.
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    # Generate solutions.
    with open(samples_path, "w") as f:
        for n, task in enumerate(tasks, start=1):
            if args.split == "instruct":
                user_prompt = build_instruct_prompt(task.prompt)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # "complete" split is docstring completion; just pass the prompt directly.
                prompt = task.prompt

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            candidates = []
            with torch.no_grad():
                for _ in range(max(1, args.num_candidates)):
                    multi = args.num_candidates > 1
                    do_sample = multi or (args.temperature > 0)
                    gen_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": do_sample,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    if do_sample:
                        gen_kwargs["temperature"] = args.temperature if args.temperature > 0 else 0.2
                        gen_kwargs["top_p"] = 0.95
                    out = model.generate(**inputs, **gen_kwargs)

                    prompt_len = inputs["input_ids"].shape[1]
                    decoded = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                    cand = extract_code(decoded)
                    cand = normalize_solution(task.prompt, cand)
                    candidates.append(cand)

            solution = max(candidates, key=score_candidate)
            record = {"task_id": task.task_id, "solution": solution}
            f.write(json.dumps(record) + "\n")

            if n % 5 == 0 or n == len(tasks):
                print(f"Generated {n}/{len(tasks)}")

    # Remote evaluation via the official gradio endpoint.
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
        # The Gradio endpoint expects indices like "1,2,3" (it will map to "BigCodeBench/1", etc.).
        selective_evaluate=",".join(task_indices),
        api_name="/predict",
    )

    # Save outputs.
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.out_dir, "pass_at_k.json"), "w") as f:
        json.dump(pass_at_k, f, indent=2)

    meta = {
        "task": "bigcodebench",
        "model_id": args.model_id,
        "bcb_version": args.bcb_version,
        "split": args.split,
        "subset": args.subset,
        "num_tasks": len(tasks),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "num_candidates": args.num_candidates,
        "task_ids": task_ids,
        "task_indices": task_indices,
        "pass@1": pass_at_k.get("pass@1"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("pass@1:", pass_at_k.get("pass@1"))


if __name__ == "__main__":
    main()
