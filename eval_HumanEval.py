import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_ID = "openai/openai_humaneval"
DEFAULT_SPLIT = "test"
DEFAULT_NUM_EXAMPLES = 164
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OUT_FILE = ""
DEFAULT_SAMPLES_FILE = ""

_CODEBLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pass@1 on HumanEval.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE, help="Optional JSON summary path")
    parser.add_argument("--samples_file", type=str, default=DEFAULT_SAMPLES_FILE, help="Optional generated samples jsonl path")
    return parser.parse_args()


def extract_code(text: str) -> str:
    m = _CODEBLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    code = text.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\\s*", "", code, flags=re.IGNORECASE)
        code = code.replace("```", "").strip()

    return code


def build_prompt(tokenizer: AutoTokenizer, humaneval_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Complete the following Python function. "
                "Output ONLY valid Python code, no markdown, no explanation.\n\n"
                f"{humaneval_prompt.rstrip()}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def completion_to_candidate(example: dict, generated_code: str) -> str:
    entry = str(example.get("entry_point", "")).strip()
    has_def = re.search(r"(?m)^\s*def\s+", generated_code) is not None
    has_entry_def = bool(entry) and (re.search(rf"(?m)^\s*def\s+{re.escape(entry)}\s*\(", generated_code) is not None)

    if has_def or has_entry_def:
        return generated_code
    return str(example.get("prompt", "")) + generated_code


def build_reference(example: dict) -> str:
    test = str(example.get("test", ""))
    entry = str(example.get("entry_point", "")).strip()
    if entry:
        return test + f"\ncheck({entry})"
    return test


def maybe_eval_code_eval(predictions: List[List[str]], references: List[str], timeout: float, num_workers: int):
    try:
        import importlib

        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
        evaluate = importlib.import_module("evaluate")
        metric = evaluate.load("code_eval")
        pass_at_k, _ = metric.compute(
            references=references,
            predictions=predictions,
            k=[1],
            timeout=timeout,
            num_workers=num_workers,
        )
        return float(pass_at_k.get("pass@1", 0.0)), None
    except Exception as exc:
        return None, str(exc)


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=cache_dir)
    total_available = len(dataset)

    indices = list(range(total_available))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: min(args.num_examples, total_available)]

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, dtype=dtype, device_map="auto")
    model.eval()

    samples = []
    predictions = []
    references = []

    for n, idx in enumerate(indices, start=1):
        ex = dataset[idx]
        prompt_text = build_prompt(tokenizer, ex["prompt"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.95 if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        generated_code = extract_code(generated_text)
        candidate = completion_to_candidate(ex, generated_code)

        samples.append(
            {
                "task_id": ex.get("task_id", f"task_{idx}"),
                "entry_point": ex.get("entry_point", ""),
                "prompt": ex.get("prompt", ""),
                "completion": generated_code,
                "candidate_for_eval": candidate,
            }
        )
        predictions.append([candidate])
        references.append(build_reference(ex))

        if n % 10 == 0:
            print(f"Processed {n}/{len(indices)}")

    pass_at_1, eval_error = maybe_eval_code_eval(
        predictions=predictions,
        references=references,
        timeout=args.timeout,
        num_workers=args.num_workers,
    )

    if pass_at_1 is not None:
        print(f"HumanEval pass@1: {pass_at_1:.6f}")
    else:
        print("HumanEval pass@1 unavailable (code_eval backend failed).")
        print(f"Reason: {eval_error}")

    if args.samples_file.strip():
        os.makedirs(os.path.dirname(args.samples_file) or ".", exist_ok=True)
        with open(args.samples_file, "w") as f:
            for rec in samples:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote samples: {args.samples_file}")

    if args.out_file.strip():
        os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
        payload = {
            "task": "HumanEval",
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "split": args.split,
            "num_examples": len(indices),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "pass_at_1": pass_at_1,
            "eval_error": eval_error,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote summary: {args.out_file}")


if __name__ == "__main__":
    main()
