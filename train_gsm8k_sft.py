import argparse
import os
import time
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_gsm8k"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_TRAIN_EXAMPLES = 512
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    """Load tokenizer with best-effort compatibility flags.

    Some Transformers versions emit a warning about an incorrect regex pattern
    for certain tokenizers. Newer versions support fix_mistral_regex=True.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on GSM8K train split.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=DEFAULT_NUM_TRAIN_EXAMPLES,
        help="Subsample GSM8K train for a quick pilot.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max optimizer steps (after grad accumulation).",
    )
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")
    return parser.parse_args()


def build_prompt(tokenizer, question: str, answer: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Solve the problem and give the final answer.\n\n{question}"},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    torch.manual_seed(args.seed)

    dataset = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    if args.num_train_examples and args.num_train_examples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        text = build_prompt(tokenizer, example["question"], example["answer"])
        out = tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    use_cuda = torch.cuda.is_available()
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, dtype=model_dtype)

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    if not use_cuda:
        model = model.float()

    print(f"Using device={device} dtype={next(model.parameters()).dtype}")

    model.train()

    def collate(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)
    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    optimizer.zero_grad(set_to_none=True)
    seen_steps = 0
    opt_step = 0
    total_loss = 0.0
    _step_time_accum = 0.0
    _step_time_count = 0

    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for batch in loader:
            batch_start_time = time.perf_counter()
            seen_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum

            if not torch.isfinite(outputs.loss):
                print(f"[skip] opt_step={opt_step}: non-finite loss={outputs.loss.item():.4f}, skipping.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                seen_steps -= 1
                continue

            loss.backward()
            total_loss += loss.item()

            if seen_steps % args.grad_accum == 0:
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if has_bad_grad:
                    print(f"[skip] opt_step={opt_step}: NaN/Inf in gradients, skipping.", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if args.save_strategy == "steps" and args.save_steps > 0 and opt_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{opt_step}")
                    print(f"\nSaving intermediate checkpoint at step {opt_step} to {checkpoint_dir} ...", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                iter_dt = time.perf_counter() - batch_start_time
                _step_time_accum += iter_dt
                _step_time_count += 1

                if opt_step % args.log_every == 0:
                    avg_loss = total_loss / args.log_every
                    lr = scheduler.get_last_lr()[0]
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"})
                    print(f"Step {opt_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s")
                    total_loss = 0.0

                pbar.update(1)

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
