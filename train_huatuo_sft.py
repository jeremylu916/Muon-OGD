import argparse
import os
import random
import time
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_huatuo"
DEFAULT_DATASET_ID = "FreedomIntelligence/medical-o1-reasoning-SFT"
DEFAULT_DATASET_CONFIG = "en"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_QUESTION_FIELD = "Question"
DEFAULT_ANSWER_FIELD = "Response"
DEFAULT_LANGUAGE_FIELD = "language"
DEFAULT_MAX_LENGTH = 1536
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = 20000
DEFAULT_MAX_STEPS = 1200
DEFAULT_WARMUP_RATIO = 0.03


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    p = argparse.ArgumentParser(description="SFT on HuatuoGPT-o1 style medical QA.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--train_split", type=str, default=DEFAULT_TRAIN_SPLIT)
    p.add_argument("--question_field", type=str, default=DEFAULT_QUESTION_FIELD)
    p.add_argument("--answer_field", type=str, default=DEFAULT_ANSWER_FIELD)
    p.add_argument("--language_field", type=str, default=DEFAULT_LANGUAGE_FIELD)
    p.add_argument("--english_only", action="store_true", default=True)
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_train_examples", type=int, default=DEFAULT_NUM_TRAIN_EXAMPLES)
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")
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


def build_messages(question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a careful medical reasoning assistant. Provide concise, clinically grounded answers.",
        },
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.train_split, cache_dir=cache_dir)

    if args.english_only:
        if args.language_field in ds.column_names:
            ds = ds.filter(lambda ex: str(ex.get(args.language_field, "")).lower().startswith("en"))
        else:
            ds = ds.filter(lambda ex: looks_english(to_text(ex.get(args.question_field, ""))))

    if args.num_train_examples and args.num_train_examples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_train_examples))

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(example):
        q = to_text(example.get(args.question_field, ""))
        a = to_text(example.get(args.answer_field, ""))

        prompt_only = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a careful medical reasoning assistant. Provide concise, clinically grounded answers.",
                },
                {"role": "user", "content": q.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        full = tokenizer.apply_chat_template(
            build_messages(q, a),
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tok = tokenizer(prompt_only, truncation=True, max_length=args.max_length, add_special_tokens=False)
        full_tok = tokenizer(
            full,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            add_special_tokens=False,
        )

        labels = full_tok["input_ids"].copy()
        prompt_len = min(len(prompt_tok["input_ids"]), args.max_length)
        for i in range(prompt_len):
            labels[i] = -100
        for i, m in enumerate(full_tok["attention_mask"]):
            if m == 0:
                labels[i] = -100

        full_tok["labels"] = labels
        return full_tok

    tokenized = ds.map(tok, remove_columns=ds.column_names)

    use_cuda = torch.cuda.is_available()
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, dtype=dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.train()

    def collate(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    max_train_steps = args.max_steps if args.max_steps else (args.epochs * len(loader) // args.grad_accum)
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)
    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    opt.zero_grad(set_to_none=True)
    seen = 0
    opt_step = 0
    accum_loss = 0.0

    pbar = tqdm(total=max_train_steps, unit="step")
    _step_time_accum = 0.0
    _step_time_count = 0
    _t0 = time.time()

    for _epoch in range(args.epochs):
        print(f"Epoch {_epoch + 1}/{args.epochs}")
        for batch in loader:
            seen += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                print(f"[skip] opt_step={opt_step} non-finite loss={loss.item():.4f}, skipping.", flush=True)
                opt.zero_grad(set_to_none=True)
                accum_loss = 0.0
                seen -= 1
                continue
            accum_loss += loss.item()
            (loss / args.grad_accum).backward()

            if seen % args.grad_accum == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                if args.save_strategy == "steps" and args.save_steps > 0 and opt_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{opt_step}")
                    print(f"\nSaving intermediate checkpoint at step {opt_step} to {checkpoint_dir} ...", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                _step_s = time.time() - _t0
                _t0 = time.time()
                _step_time_accum += _step_s
                _step_time_count += 1
                avg_step_s = _step_time_accum / _step_time_count
                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step_s=f"{avg_step_s:.2f}s")
                pbar.update(1)

                if opt_step % 10 == 0:
                    print(f"Step {opt_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s", flush=True)

                if args.max_steps and opt_step >= args.max_steps:
                    break
        if args.max_steps and opt_step >= args.max_steps:
            break

    pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
