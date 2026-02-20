import argparse
import json
import math
import os
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# --- Default Configuration ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_bigcodebench"
DEFAULT_BCB_VERSION = "v0.1.4"
DEFAULT_SPLIT = "instruct"  # instruct|complete
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 4  # Increased from 1 for efficiency
DEFAULT_GRAD_ACCUM = 4  # Adjusted to keep effective batch size similar (4*4=16)
DEFAULT_EPOCHS = 3      # 1 epoch might be underfitting for small models
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = None  # None = use all
DEFAULT_MAX_STEPS = 500 # Increased slightly
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def parse_args():
    p = argparse.ArgumentParser(description="SFT on BigCodeBench (Corrected).")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bcb_version", type=str, default=DEFAULT_BCB_VERSION)
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["instruct", "complete"])
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
    p.add_argument("--task_ids_file", type=str, default="")
    return p.parse_args()


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    
    # Qwen 2.5 usually has an eos_token. If pad is missing, use eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def build_prompt_text(args, prompt: str) -> str:
    """Builds the prompt string using the chat template format."""
    # We construct the list of messages for the prompt ONLY.
    if args.split == "instruct":
        user_content = (
            "Write Python code that solves the task. "
            "Output ONLY valid Python code (no markdown, no explanation).\n\n" + prompt.strip()
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]
    else:
        # Complete/Docstring style
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.strip()}
        ]
    return messages


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load Dataset
    print(f"Loading dataset {args.bcb_version}...")
    try:
        ds = load_dataset("bigcode/bigcodebench", split="train", cache_dir=cache_dir) # 'train' usually contains the full set on HF
    except Exception:
        # Fallback if specific version/split names differ
        ds = load_dataset("bigcode/bigcodebench", split=args.bcb_version, cache_dir=cache_dir)

    # Filter by task IDs if provided
    if args.task_ids_file.strip():
        with open(args.task_ids_file, "r") as f:
            task_ids = json.load(f)
        if isinstance(task_ids, list):
            keep = set(task_ids)
            ds = ds.filter(lambda ex: ex["task_id"] in keep)
            print(f"Filtered to {len(ds)} tasks from file.")

    # Subsample (for pilots/debugging)
    if args.num_train_examples and args.num_train_examples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_train_examples))
        print(f"Subsampled to {len(ds)} examples.")

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    # --- Corrected Tokenization Logic ---
    def tok(example):
        prompt_key = f"{args.split}_prompt"
        raw_prompt = example[prompt_key]
        solution = example["canonical_solution"]

        # 1. Build Prompt (using chat template, NO tokenization yet)
        messages = build_prompt_text(args, raw_prompt)
        # apply_chat_template with tokenize=False gives us the raw formatted string
        # add_generation_prompt=True ensures it ends with "<|im_start|>assistant\n" (or similar)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 2. Build Solution (Add EOS manually to ensure model stops)
        # We add the eos_token explicitly.
        solution_text = solution + tokenizer.eos_token

        # 3. Tokenize Parts Separately
        # We must NOT allow the tokenizer to merge the last token of prompt with first of answer.
        # We treat them as separate blocks.
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer(solution_text, add_special_tokens=False)["input_ids"]

        # 4. Concatenate
        input_ids = prompt_ids + solution_ids
        
        # 5. Create Labels
        # Mask prompt (-100), keep solution
        labels = ([-100] * len(prompt_ids)) + solution_ids

        # 6. Truncate
        if len(input_ids) > args.max_length:
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]
        
        # 7. Pad (Right padding for simplicity in training)
        pad_len = args.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    print("Tokenizing dataset...")
    tokenized = ds.map(tok, remove_columns=ds.column_names)

    # --- Debug Verification (Crucial Step) ---
    print("\n--- SANITY CHECK: Decoding first example ---")
    ex = tokenized[0]
    valid_labels = [l for l in ex["labels"] if l != -100]
    print(f"Full Input Decoded (First 200 chars): {tokenizer.decode(ex['input_ids'])[:200]}...")
    print(f"Labels Decoded (First 50 chars): {tokenizer.decode(valid_labels)[:50]}...")
    print("--------------------------------------------\n")

    # Load Model
    print(f"Loading model {args.model_id}...")
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, torch_dtype=dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.train()

    # Data Loader
    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # Optimizer & Schedule
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Calculate steps
    num_update_steps_per_epoch = len(loader) // args.grad_accum
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print(f"Starting training: Epochs={args.epochs}, Batch={args.batch_size}, GradAccum={args.grad_accum}")
    print(f"Total optimization steps: {max_train_steps}")

    # Training Loop
    global_step = 0
    total_loss = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Scale loss by grad_accum
            loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                opt.step()
                scheduler.step()
                opt.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = total_loss * args.grad_accum / args.log_every
                    print(f"Step {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0

                if global_step >= max_train_steps:
                    break
        
        if global_step >= max_train_steps:
            break

    # Save
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()