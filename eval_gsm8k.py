import argparse
import json
import os
import re
import random
from datetime import datetime, timezone

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SPLIT = "test"
DEFAULT_NUM_EXAMPLES = 100
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_OUT_FILE = ""


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    except Exception:
        if os.path.isdir(model_id):
            cfg_path = os.path.join(model_id, "tokenizer_config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        cfg = json.load(f)
                    if isinstance(cfg.get("extra_special_tokens"), list):
                        cfg.pop("extra_special_tokens", None)
                        with open(cfg_path, "w") as f:
                            json.dump(cfg, f, indent=2)
                        print("Patched tokenizer_config.json: removed incompatible extra_special_tokens list.")
                except Exception:
                    pass
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def extract_gsm8k_answer(answer_text: str):
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
    if not match:
        return None
    return match.group(1).replace(",", "")


def extract_pred_answer(text: str):
    text = text.strip()

    patterns = [
        r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"[Tt]he answer is\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"[Ff]inal answer[:\s]*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).replace(",", "")

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def normalize_num(x):
    if x is None:
        return None
    try:
        v = float(x.replace(",", ""))
        if v.is_integer():
            return str(int(v))
        return str(v)
    except Exception:
        return x.strip()


def build_prompt(tokenizer, question: str):
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": f"Solve the following math word problem. End your response with the final numeric answer.\n\n{question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K exact-match accuracy.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    dataset = load_dataset("gsm8k", "main", split=args.split, cache_dir=cache_dir)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_examples]

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="auto",
    )

    correct = 0
    total = 0

    for idx in indices:
        example = dataset[idx]
        question = example["question"]
        gold = extract_gsm8k_answer(example["answer"])

        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        pred = extract_pred_answer(decoded)

        total += 1
        if gold is not None and pred is not None and normalize_num(pred) == normalize_num(gold):
            correct += 1

        if total % 10 == 0:
            print(f"Processed {total}/{args.num_examples}")

    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.3f} ({correct}/{total})")

    if args.out_file.strip():
        os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
        payload = {
            "task": "gsm8k",
            "model_id": args.model_id,
            "split": args.split,
            "num_examples": args.num_examples,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()