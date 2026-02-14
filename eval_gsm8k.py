import argparse
import json
import os
import re
import random
from datetime import datetime, timezone
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SPLIT = "test"
DEFAULT_NUM_EXAMPLES = 100
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_OUT_FILE = ""


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with best-effort compatibility flags.

    Some Transformers versions emit a warning about an incorrect regex pattern
    for certain tokenizers. Newer versions support fix_mistral_regex=True.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def extract_last_number(text: str):
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not numbers:
        return None
    last = numbers[-1]
    return last.replace(",", "")


def extract_gsm8k_answer(answer_text: str):
    # GSM8K gold format: "#### 42"
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if not match:
        return None
    return match.group(1)


def build_prompt(tokenizer, question: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer with a final numeric result."},
        {"role": "user", "content": f"Solve the problem and give only the final number.\n\n{question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K exact-match accuracy.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--out_file",
        type=str,
        default=DEFAULT_OUT_FILE,
        help="Optional path to write a JSON result (e.g., results/gsm8k/run.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset("gsm8k", "main", split=args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_examples]

    tokenizer = load_tokenizer(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
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
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = extract_last_number(decoded)

        total += 1
        if gold is not None and pred is not None and pred == gold:
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
