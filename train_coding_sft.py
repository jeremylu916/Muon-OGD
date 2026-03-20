import argparse
import math
import os
import random
import re
import time
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, **kwargs):
        return x


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_mbpp_qwen0.5b_adamw"
DEFAULT_DATASET_ID = "mbpp"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 1000  # ignored when max_steps > 0
DEFAULT_LR = 1e-5
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = 0  # 0 = full split
DEFAULT_MAX_STEPS = 50
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10
DEFAULT_PROBE_EVERY = 25
DEFAULT_PROBE_MAX_NEW_TOKENS = 96
DEFAULT_VAL_RATIO = 0.05
DEFAULT_VAL_EVERY = 25
DEFAULT_VAL_MAX_BATCHES = 16
DEFAULT_TARGET_CLEANING = "fences"

FIXED_PROBES = [
    ("Q1", "Explain Machine Learning in 1 sentence.", "Machine learning is a method where models learn patterns from data to make predictions or decisions without explicit rules."),
    ("Q2", "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "$10"),
    (
        "Q3",
        "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?",
        "Patent foramen ovale (PFO) causing paradoxical embolism.",
    ),
]

_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _looks_like_code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith(("def ", "class ", "import ", "from ", "if ", "for ", "while ", "try:", "with ", "return ", "@")):
        return True
    if any(tok in s for tok in [" = ", ":", "(", ")", "[", "]", "{", "}"]):
        return True
    return False


def clean_code_target(raw_text: str, mode: str = "fences") -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    if mode == "none":
        return text

    m = _CODEBLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()

    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    if mode == "fences":
        return text

    lines = text.splitlines()
    if not lines:
        return text

    start_idx = 0
    while start_idx < len(lines):
        line = lines[start_idx].strip()
        if not line:
            start_idx += 1
            continue
        lowered = line.lower()
        if _looks_like_code_line(line):
            break
        if lowered.startswith((
            "here is", "here's", "sure", "of course", "certainly", "the code", "this code",
            "you can", "we can", "let's", "below is", "explanation", "to solve"
        )):
            start_idx += 1
            continue
        start_idx += 1

    cleaned_lines = lines[start_idx:] if start_idx < len(lines) else lines
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else text


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def parse_args():
    p = argparse.ArgumentParser(description="HumanEval-oriented SFT on MBPP with AdamW.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_train_examples", type=int, default=DEFAULT_NUM_TRAIN_EXAMPLES)
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    p.add_argument("--probe_every", type=int, default=DEFAULT_PROBE_EVERY)
    p.add_argument("--probe_max_new_tokens", type=int, default=DEFAULT_PROBE_MAX_NEW_TOKENS)
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--val_every", type=int, default=DEFAULT_VAL_EVERY)
    p.add_argument("--val_max_batches", type=int, default=DEFAULT_VAL_MAX_BATCHES)
    p.add_argument(
        "--target_cleaning",
        type=str,
        default=DEFAULT_TARGET_CLEANING,
        choices=["none", "fences", "aggressive"],
    )
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"])
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument(
        "--prefer_prompt_field",
        action="store_true",
        default=True,
        help="Prefer an existing code-style prompt field over plain text when available.",
    )
    return p.parse_args()


def build_prompt_solution(user_prompt: str, solution_text: str):
    user_instruction = (
        "Complete the following Python function. "
        "Output ONLY valid Python code, no markdown, no explanation.\n\n"
        f"{user_prompt.rstrip()}"
    )

    messages_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_instruction},
    ]
    return messages_prompt, solution_text.rstrip() + "\n"


def infinite_loader(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def choose_mbpp_prompt(example: dict, prefer_prompt_field: bool = True) -> str:
    """
    Prefer a code-like prompt if present. Fall back to text.
    Avoid test_list injection because it shifts distribution away from HumanEval.
    """
    candidate_keys = []
    if prefer_prompt_field:
        candidate_keys.extend(["prompt", "starter_code"])
    candidate_keys.extend(["text"])

    for key in candidate_keys:
        value = str(example.get(key, "")).strip()
        if value:
            return value

    return ""


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=cache_dir)

    if args.num_train_examples and args.num_train_examples > 0:
        n = min(args.num_train_examples, len(dataset))
        dataset = dataset.shuffle(seed=args.seed).select(range(n))

    print("Dataset columns:", dataset.column_names)
    try:
        print("Sample example keys:", list(dataset[0].keys()))
    except Exception:
        pass

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    def tokenize_fn(example):
        # MBPP: prefer prompt-like field, otherwise text. Do NOT append test_list.
        if "code" in example:
            user_prompt = choose_mbpp_prompt(example, prefer_prompt_field=args.prefer_prompt_field)
            solution = clean_code_target(str(example.get("code", "")), mode=args.target_cleaning)

        elif "instruction" in example and "output" in example:
            instruction = str(example.get("instruction", "")).strip()
            extra_input = str(example.get("input", "")).strip()
            user_prompt = instruction if not extra_input else f"{instruction}\n\nInput:\n{extra_input}"
            solution = clean_code_target(str(example.get("output", "")), mode=args.target_cleaning)

        else:
            user_prompt = str(example.get("prompt", "")).strip()
            solution = clean_code_target(str(example.get("canonical_solution", "")), mode=args.target_cleaning)

        if not user_prompt:
            user_prompt = str(example)

        prompt_messages, solution = build_prompt_solution(user_prompt, solution)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer(solution + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + solution_ids
        labels = ([-100] * len(prompt_ids)) + solution_ids

        if len(input_ids) > args.max_length:
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]

        pad_len = args.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    print(f"Tokenizing dataset: {args.dataset_id} ({args.split})...")
    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    if args.val_ratio > 0 and len(tokenized) > 1:
        shuffled = tokenized.shuffle(seed=args.seed)
        val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * args.val_ratio)))
        val_tokenized = shuffled.select(range(val_count))
        train_tokenized = shuffled.select(range(val_count, len(shuffled)))
        print(f"Train split: {len(train_tokenized)} | Val split: {len(val_tokenized)}")
    else:
        train_tokenized = tokenized
        val_tokenized = None

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(
        train_tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    ) if val_tokenized is not None else None

    updates_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    print(f"Micro-batches per epoch: {len(loader)}")
    print(f"Approx optimizer updates per epoch: {updates_per_epoch}")

    use_cuda = torch.cuda.is_available()
    model_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, dtype=model_dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    if not use_cuda:
        model = model.float()
    model.train()

    def run_probe(step_idx: int):
        if args.probe_every <= 0:
            return
        model.eval()
        for probe_id, probe_q, probe_t in FIXED_PROBES:
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": probe_q},
            ]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.probe_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            pred_ids = out[0, enc["input_ids"].shape[1]:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            print(
                f"[probe] step={step_idx} | {probe_id} | q={probe_q[:120]!r} | "
                f"pred={pred[:180]!r} | target={probe_t[:180]!r}",
                flush=True,
            )
        model.train()

    def compute_val_loss():
        if val_loader is None:
            return None
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                if torch.isfinite(out.loss):
                    val_loss_sum += out.loss.item()
                    val_batches += 1
                if args.val_max_batches > 0 and val_batches >= args.val_max_batches:
                    break
        model.train()
        return (val_loss_sum / val_batches) if val_batches > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_train_steps = args.max_steps if args.max_steps > 0 else max(1, args.epochs * updates_per_epoch)
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    optimizer.zero_grad(set_to_none=True)
    seen_micro_steps = 0
    opt_step = 0
    total_loss = 0.0
    step_time_accum = 0.0
    step_time_count = 0

    batch_iter = infinite_loader(loader)
    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")

    while opt_step < max_train_steps:
        batch_start = time.perf_counter()
        batch = next(batch_iter)
        seen_micro_steps += 1
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        raw_loss = outputs.loss

        if not torch.isfinite(raw_loss):
            print(f"[skip] opt_step={opt_step}: non-finite loss={raw_loss.item():.4f}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            seen_micro_steps -= 1
            continue

        (raw_loss / args.grad_accum).backward()
        total_loss += raw_loss.item()

        if seen_micro_steps % args.grad_accum == 0:
            has_bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if has_bad_grad:
                print(f"[skip] opt_step={opt_step}: NaN/Inf gradients", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            iter_dt = time.perf_counter() - batch_start
            step_time_accum += iter_dt
            step_time_count += 1

            if opt_step % args.log_every == 0:
                avg_loss = total_loss / (args.log_every * args.grad_accum)
                avg_step_s = step_time_accum / max(1, step_time_count)
                lr = scheduler.get_last_lr()[0]

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "step_s": f"{avg_step_s:.2f}s",
                })

                val_msg = ""
                if args.val_every > 0 and opt_step % args.val_every == 0:
                    val_loss = compute_val_loss()
                    if val_loss is not None:
                        val_msg = f" | ValLoss: {val_loss:.4f}"

                print(
                    f"Step {opt_step}/{max_train_steps} | Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s{val_msg}",
                    flush=True,
                )
                total_loss = 0.0
                step_time_accum = 0.0
                step_time_count = 0

            if args.probe_every > 0 and opt_step % args.probe_every == 0:
                run_probe(opt_step)

            if args.save_strategy == "steps" and args.save_steps > 0 and opt_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{opt_step}")
                print(f"Saving checkpoint at step {opt_step} to {checkpoint_dir}", flush=True)
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

            pbar.update(1)

    pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()