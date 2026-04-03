import argparse
import json
import os
import random
import re
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEFAULT_VERIFIER_MODEL_ID = "FreedomIntelligence/medical_o1_verifier_3B"
DEFAULT_DATASET_ID = "FreedomIntelligence/medical-o1-verifiable-problem"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_QUESTION_FIELD = "Open-ended Verifiable Question"
DEFAULT_ANSWER_FIELD = "Ground-True Answer"
DEFAULT_LANGUAGE_FIELD = "language"
DEFAULT_VERIFIABLE_FIELD = "verifiable"
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 348
DEFAULT_OUT_FILE = "results/medical/medical_eval.json"
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_JUDGE_FINAL_ANSWER_ONLY = True

VERIFIER_TEMPLATE = """<Model Response>\n{}\n</Model Response>\n\n<Reference Answer>\n{}\n</Reference Answer>\n\nYour task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output \"True\" . If it is incorrect or fails to select the correct option (if options are provided), output \"False\" . {}"""


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate medical QA with local medical_o1_verifier_3B.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--verifier_model_id", type=str, default=DEFAULT_VERIFIER_MODEL_ID)
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
    p.add_argument(
        "--judge_final_answer_only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_JUDGE_FINAL_ANSWER_ONLY,
        help="If true, judge only the final answer span after CoT/reasoning markers for both prediction and reference.",
    )
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
            "content": "You are a careful medical assistant. Answer the medical question directly and concisely. Give the final answer first. Do not include unnecessary explanation.",
        },
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_final_answer_only(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    if "</think>" in s:
        tail = s.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail

    for pat in [
        r"(?is)final\s*answer\s*[:：]\s*(.+)$",
        r"(?is)answer\s*[:：]\s*(.+)$",
        r"(?is)####\s*(.+)$",
    ]:
        m = re.search(pat, s)
        if m:
            tail = m.group(1).strip()
            if tail:
                return tail

    return s


def build_verifier_input(tokenizer, pred_for_judge: str, gold_for_judge: str) -> dict:
    text = VERIFIER_TEMPLATE.format(pred_for_judge, gold_for_judge, tokenizer.eos_token)
    return tokenizer([text], return_tensors="pt")


def load_verifier(verifier_model_id: str, cache_dir=None):
    verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_id, cache_dir=cache_dir)
    try:
        verifier_model = AutoModelForSequenceClassification.from_pretrained(
            verifier_model_id,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=2,
        )
    except Exception:
        verifier_model = AutoModelForSequenceClassification.from_pretrained(
            verifier_model_id,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto",
            num_labels=2,
        )
    return verifier_tokenizer, verifier_model


def verify_prediction(verifier_tokenizer, verifier_model, pred_for_judge: str, gold_for_judge: str) -> str:
    input_batch = build_verifier_input(verifier_tokenizer, pred_for_judge, gold_for_judge)
    input_batch = {k: v.to(verifier_model.device) for k, v in input_batch.items()}
    with torch.no_grad():
        logits = verifier_model(**input_batch, return_dict=True).logits
        probabilities = F.softmax(logits, dim=-1)
    return "CORRECT" if float(probabilities[0, 1]) > 0.5 else "INCORRECT"


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split, cache_dir=cache_dir)

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

    verifier_tokenizer, verifier_model = load_verifier(args.verifier_model_id, cache_dir=cache_dir)

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50

    correct = 0
    judge_errors = 0
    final_trimmed_pred = 0
    final_trimmed_gold = 0
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

        pred_for_judge = pred
        gold_for_judge = gold
        if args.judge_final_answer_only:
            pred_for_judge = extract_final_answer_only(pred)
            gold_for_judge = extract_final_answer_only(gold)
            if pred_for_judge != pred:
                final_trimmed_pred += 1
            if gold_for_judge != gold:
                final_trimmed_gold += 1

        try:
            verdict = verify_prediction(verifier_tokenizer, verifier_model, pred_for_judge, gold_for_judge)
        except Exception as e:
            judge_errors += 1
            verdict = "INCORRECT"
            if i <= 5 or i % 50 == 0:
                print(f"Verifier warning at sample {i}: {e}")

        ok = int(verdict == "CORRECT")
        correct += ok

        if i <= 30:
            rows.append(
                {
                    "idx": int(idx),
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "gold_for_judge": gold_for_judge,
                    "pred_for_judge": pred_for_judge,
                    "judge_verdict": verdict,
                }
            )

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(indices)):
            print(
                f"Processed {i}/{len(indices)} "
                f"(judge_correct={correct}, judge_errors={judge_errors})",
                flush=True,
            )

    total = len(indices)
    judge_accuracy = correct / total if total else 0.0

    payload = {
        "task": "medical_verifiable",
        "model_id": args.model_id,
        "verifier_model_id": args.verifier_model_id,
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
        "judge_final_answer_only": args.judge_final_answer_only,
        "final_trimmed_pred": final_trimmed_pred,
        "final_trimmed_gold": final_trimmed_gold,
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
    print(f"Verifier errors (counted as INCORRECT): {judge_errors}")
    print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()
