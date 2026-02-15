import argparse
import json
import os
import random
import re
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_ID = "FreedomIntelligence/medical-o1-verifiable-problem"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_QUESTION_FIELD = "Open-ended Verifiable Question"
DEFAULT_ANSWER_FIELD = "Ground-True Answer"
DEFAULT_LANGUAGE_FIELD = "language"
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_PROGRESS_EVERY = 20
DEFAULT_OUT_FILE = "results/medical/huatuo_zero_shot_tokenf1.json"


def load_tokenizer(model_id: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Huatuo verifiable set with Exact Match + Token F1.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--question_field", type=str, default=DEFAULT_QUESTION_FIELD)
    p.add_argument("--answer_field", type=str, default=DEFAULT_ANSWER_FIELD)
    p.add_argument("--language_field", type=str, default=DEFAULT_LANGUAGE_FIELD)
    p.add_argument("--english_only", action="store_true", default=True)
    p.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY)
    p.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE)
    return p.parse_args()


def to_text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join([str(x) for x in v])
    if isinstance(v, dict):
        return "\n".join([f"{k}: {v[k]}" for k in sorted(v.keys())])
    return str(v)


def looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_count / max(len(text), 1)
    alpha = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    return ratio > 0.85 and alpha >= 10


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    g_count = {}
    for t in g:
        g_count[t] = g_count.get(t, 0) + 1

    common = 0
    for t in p:
        if g_count.get(t, 0) > 0:
            common += 1
            g_count[t] -= 1

    if common == 0:
        return 0.0

    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def build_prompt(tokenizer, question: str):
    messages = [
        {
            "role": "system",
            "content": "You are a careful medical reasoning assistant. Answer clearly and concisely.",
        },
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split)

    if args.english_only:
        if args.language_field in ds.column_names:
            ds = ds.filter(lambda ex: str(ex.get(args.language_field, "")).lower().startswith("en"))
        else:
            ds = ds.filter(lambda ex: looks_english(to_text(ex.get(args.question_field, ""))))

    ds = ds.filter(
        lambda ex: to_text(ex.get(args.question_field, "")).strip() != ""
        and to_text(ex.get(args.answer_field, "")).strip() != ""
    )

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_examples]

    tokenizer = load_tokenizer(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16, device_map="auto")

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    exact = 0
    f1_sum = 0.0
    rows = []

    for i, idx in enumerate(indices, start=1):
        ex = ds[idx]
        q = to_text(ex.get(args.question_field, ""))
        gold = to_text(ex.get(args.answer_field, ""))

        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        pred = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold)
        em = int(pred_n == gold_n)
        f1 = token_f1(pred, gold)

        exact += em
        f1_sum += f1

        if i <= 30:
            rows.append(
                {
                    "idx": int(idx),
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "em": em,
                    "f1": f1,
                }
            )

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(indices)):
            print(f"Processed {i}/{len(indices)}")

    total = len(indices)
    em_score = exact / total if total else 0.0
    f1_score = f1_sum / total if total else 0.0

    payload = {
        "task": "huatuo_verifiable",
        "metric": "exact_match_and_token_f1",
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
        "english_only": args.english_only,
        "num_examples": total,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "exact_match": em_score,
        "token_f1": f1_score,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_predictions": rows,
    }

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Exact match: {em_score:.4f} ({exact}/{total})")
    print(f"Token F1: {f1_score:.4f}")
    print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()
