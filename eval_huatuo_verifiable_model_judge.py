import argparse
import json
import os
import random
import re
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_DATASET_ID = "FreedomIntelligence/medical-o1-verifiable-problem"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_QUESTION_FIELD = "Open-ended Verifiable Question"
DEFAULT_ANSWER_FIELD = "Ground-True Answer"
DEFAULT_LANGUAGE_FIELD = "language"
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_JUDGE_MAX_NEW_TOKENS = 16
DEFAULT_PROGRESS_EVERY = 20
DEFAULT_OUT_FILE = "results/medical/huatuo_zero_shot_model_judge.json"


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Huatuo verifiable set with a local LLM judge model.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--judge_model_id", type=str, default=DEFAULT_JUDGE_MODEL_ID)
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
    p.add_argument("--judge_max_new_tokens", type=int, default=DEFAULT_JUDGE_MAX_NEW_TOKENS)
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


def build_prompt(tokenizer, question: str):
    messages = [
        {
            "role": "system",
            "content": "You are a careful medical reasoning assistant. Answer clearly and concisely.",
        },
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_judge_prompt(question: str, reference_answer: str, model_response: str) -> str:
    return (
        "You are an expert medical evaluator assessing whether a model's response correctly answers a medical question. "
        "Your task is to compare the model's response to the reference answer and determine if the model's response is:\n"
        "1. CORRECT: The response contains the key medical information from the reference answer, even if phrased differently "
        "or includes additional correct medical details.\n"
        "2. INCORRECT: The response is medically wrong, misses the main point, or provides incorrect medical information.\n"
        "Focus on medical accuracy and completeness, not on writing style or verbosity.\n"
        "[Medical Question]\n"
        f"{question}\n"
        "[Reference Answer]\n"
        f"{reference_answer}\n"
        "[Model Response]\n"
        f"{model_response}\n"
        "Evaluate the model's response. Output ONLY one of: \"CORRECT\" or \"INCORRECT\"."
    )


def build_judge_chat_prompt(tokenizer, judge_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a strict medical answer evaluator."},
        {"role": "user", "content": judge_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_judge_verdict(text: str) -> str:
    content = (text or "").strip().upper()
    if "CORRECT" in content and "INCORRECT" not in content:
        return "CORRECT"
    if "INCORRECT" in content:
        return "INCORRECT"
    first = re.split(r"\s+", content)[0] if content else ""
    if first in {"CORRECT", "INCORRECT"}:
        return first
    return "INCORRECT"


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split, cache_dir=cache_dir)

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

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map="auto",
    )

    judge_tokenizer = load_tokenizer(args.judge_model_id, cache_dir=cache_dir)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model_id,
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map="auto",
    )

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    judge_correct = 0
    rows = []

    for i, idx in enumerate(tqdm(indices, total=len(indices), desc="Judging", unit="sample"), start=1):
        ex = ds[idx]
        q = to_text(ex.get(args.question_field, ""))
        gold = to_text(ex.get(args.answer_field, ""))

        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        pred = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        judge_prompt = build_judge_prompt(q, gold, pred)
        judge_chat_prompt = build_judge_chat_prompt(judge_tokenizer, judge_prompt)
        judge_inputs = judge_tokenizer(judge_chat_prompt, return_tensors="pt").to(judge_model.device)

        with torch.no_grad():
            judge_out = judge_model.generate(
                **judge_inputs,
                max_new_tokens=args.judge_max_new_tokens,
                do_sample=False,
            )

        judge_prompt_len = judge_inputs["input_ids"].shape[1]
        judge_text = judge_tokenizer.decode(judge_out[0][judge_prompt_len:], skip_special_tokens=True)
        verdict = parse_judge_verdict(judge_text)
        ok = int(verdict == "CORRECT")
        judge_correct += ok

        if i <= 30:
            rows.append(
                {
                    "idx": int(idx),
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "judge_raw": judge_text.strip(),
                    "judge_verdict": verdict,
                }
            )

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(indices)):
            print(f"Processed {i}/{len(indices)}")

    total = len(indices)
    judge_accuracy = judge_correct / total if total else 0.0

    payload = {
        "task": "huatuo_verifiable",
        "metric": "local_model_judge_accuracy",
        "model_id": args.model_id,
        "judge_model_id": args.judge_model_id,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
        "english_only": args.english_only,
        "num_examples": total,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "judge_accuracy": judge_accuracy,
        "judge_correct": judge_correct,
        "judge_total": total,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_predictions": rows,
    }

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Judge accuracy: {judge_accuracy:.4f} ({judge_correct}/{total})")
    print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()
