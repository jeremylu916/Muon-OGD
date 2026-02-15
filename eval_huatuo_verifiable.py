import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET_ID = "FreedomIntelligence/medical-o1-verifiable-problem"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_QUESTION_FIELD = "Open-ended Verifiable Question"
DEFAULT_ANSWER_FIELD = "Ground-True Answer"
DEFAULT_LANGUAGE_FIELD = "language"
DEFAULT_VERIFIABLE_FIELD = "verifiable"
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_OUT_FILE = "results/medical/huatuo_verifiable_eval.json"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_JUDGE_MAX_RETRIES = 10
DEFAULT_JUDGE_MIN_SLEEP = 0.5
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_JUDGE_MAX_SLEEP = 30.0


def load_tokenizer(model_id: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on HuatuoGPT-o1 verifiable medical QA.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--question_field", type=str, default=DEFAULT_QUESTION_FIELD)
    p.add_argument("--answer_field", type=str, default=DEFAULT_ANSWER_FIELD)
    p.add_argument("--language_field", type=str, default=DEFAULT_LANGUAGE_FIELD)
    p.add_argument("--verifiable_field", type=str, default=DEFAULT_VERIFIABLE_FIELD)
    p.add_argument("--english_only", action="store_true", default=True)
    p.add_argument("--require_verifiable", action="store_true", default=True)
    p.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge_max_retries", type=int, default=DEFAULT_JUDGE_MAX_RETRIES)
    p.add_argument("--judge_min_sleep", type=float, default=DEFAULT_JUDGE_MIN_SLEEP)
    p.add_argument("--judge_max_sleep", type=float, default=DEFAULT_JUDGE_MAX_SLEEP)
    p.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY)
    p.add_argument(
        "--openai_api_key",
        type=str,
        default="",
        help="Optional OpenAI API key. If empty, reads OPENAI_API_KEY from environment.",
    )
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


def is_trueish(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).lower() in {"true", "yes", "1", "verifiable"}


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


def call_openai_judge(
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int,
    min_sleep: float,
    max_sleep: float,
) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a strict medical answer evaluator."},
            {"role": "user", "content": prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = obj["choices"][0]["message"]["content"].strip().upper()
            if "CORRECT" in content and "INCORRECT" not in content:
                return "CORRECT"
            if "INCORRECT" in content:
                return "INCORRECT"
            # Fallback parse: accept exact first token.
            first = re.split(r"\s+", content)[0] if content else ""
            if first in {"CORRECT", "INCORRECT"}:
                return first
            return "INCORRECT"
        except urllib.error.HTTPError as e:
            # Auth/config errors should fail fast.
            if e.code in {401, 403}:
                raise RuntimeError(f"Judge API HTTP error: {e.code} {e.reason}")

            # Retry transient/rate-limit errors with backoff.
            if e.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                retry_after = 0.0
                try:
                    hdr = e.headers.get("Retry-After") if e.headers is not None else None
                    if hdr:
                        retry_after = float(hdr)
                except Exception:
                    retry_after = 0.0

                wait_s = min(max_sleep, max(min_sleep, retry_after, min_sleep * (2 ** attempt)))
                print(
                    f"Judge retry {attempt + 1}/{max_retries} after HTTP {e.code}; sleeping {wait_s:.1f}s",
                    flush=True,
                )
                time.sleep(wait_s)
                continue

            if attempt == max_retries - 1:
                raise RuntimeError(f"Judge API HTTP error: {e.code} {e.reason}")
            wait_s = min(max_sleep, max(min_sleep, min_sleep * (2 ** attempt)))
            print(
                f"Judge retry {attempt + 1}/{max_retries} after HTTP {e.code}; sleeping {wait_s:.1f}s",
                flush=True,
            )
            time.sleep(wait_s)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Judge API failed: {e}")
            wait_s = min(max_sleep, max(min_sleep, min_sleep * (2 ** attempt)))
            print(
                f"Judge retry {attempt + 1}/{max_retries} after exception; sleeping {wait_s:.1f}s",
                flush=True,
            )
            time.sleep(wait_s)

    return "INCORRECT"


def main():
    args = parse_args()
    api_key = args.openai_api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for GPT-judge evaluation. Set env var or pass --openai_api_key.")

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split)

    if args.english_only:
        if args.language_field in ds.column_names:
            ds = ds.filter(lambda ex: str(ex.get(args.language_field, "")).lower().startswith("en"))
        else:
            ds = ds.filter(lambda ex: looks_english(to_text(ex.get(args.question_field, ""))))

    if args.require_verifiable:
        if args.verifiable_field in ds.column_names:
            ds = ds.filter(lambda ex: is_trueish(ex.get(args.verifiable_field, False)))
        else:
            print(f"Warning: verifiable field '{args.verifiable_field}' not found; no verifiable filtering applied.")

    # Keep only records with non-empty question/answer.
    ds = ds.filter(
        lambda ex: to_text(ex.get(args.question_field, "")).strip() != ""
        and to_text(ex.get(args.answer_field, "")).strip() != ""
    )

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_examples]

    tokenizer = load_tokenizer(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16, device_map="auto")

    # Avoid noisy warnings from baked-in sampling params when do_sample=False.
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    correct = 0
    judge_errors = 0
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

        judge_prompt = build_judge_prompt(q, gold, pred)
        try:
            verdict = call_openai_judge(
                api_key,
                args.judge_model,
                judge_prompt,
                args.judge_max_retries,
                args.judge_min_sleep,
                args.judge_max_sleep,
            )
        except RuntimeError as e:
            # Keep long evaluation running even if individual judge calls fail.
            judge_errors += 1
            verdict = "INCORRECT"
            if i <= 5 or i % 50 == 0:
                print(f"Judge warning at sample {i}: {e}")
        ok = int(verdict == "CORRECT")
        correct += ok

        if i <= 30:
            rows.append(
                {
                    "idx": int(idx),
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "judge_verdict": verdict,
                }
            )

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(indices)):
            print(
                f"Processed {i}/{len(indices)} "
                f"(judge_correct={correct}, judge_errors={judge_errors})",
                flush=True,
            )

        if i % 20 == 0:
            print(f"Processed {i}/{len(indices)}")

    total = len(indices)
    judge_accuracy = correct / total if total else 0.0

    payload = {
        "task": "huatuo_verifiable",
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "split": args.split,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
        "language_field": args.language_field,
        "verifiable_field": args.verifiable_field,
        "english_only": args.english_only,
        "require_verifiable": args.require_verifiable,
        "num_examples": total,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "judge_model": args.judge_model,
        "judge_max_retries": args.judge_max_retries,
        "judge_min_sleep": args.judge_min_sleep,
        "judge_max_sleep": args.judge_max_sleep,
        "judge_accuracy": judge_accuracy,
        "judge_correct": correct,
        "judge_total": total,
        "judge_errors": judge_errors,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_predictions": rows,
    }

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Judge accuracy: {judge_accuracy:.4f} ({correct}/{total})")
    print(f"Judge API errors (counted as INCORRECT): {judge_errors}")
    print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()
