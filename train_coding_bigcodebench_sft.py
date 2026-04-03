import argparse
import json
import math
import os
import random
import re
import textwrap
from typing import Dict, List

import torch
from tqdm.auto import tqdm
import time
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# --- Default Configuration ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DEFAULT_OUTPUT_DIR = "outputs/sft_bigcodebench"
DEFAULT_BCB_VERSION = "v0.1.4"
DEFAULT_SPLIT = "complete"  # instruct|complete
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 4  # Increased from 1 for efficiency
DEFAULT_GRAD_ACCUM = 4  # Adjusted to keep effective batch size similar (4*4=16)
DEFAULT_EPOCHS = 3      # 1 epoch might be underfitting for small models
DEFAULT_LR = 5e-6
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = None  # None = use all
DEFAULT_MAX_STEPS = 500 # Increased slightly
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10
DEFAULT_VAL_RATIO = 0.10
DEFAULT_VAL_EVERY = 25
DEFAULT_VAL_MAX_BATCHES = 32
DEFAULT_PROBE_EVERY = 50
DEFAULT_PROBE_NUM_QUESTIONS = 3
DEFAULT_PROBE_MAX_NEW_TOKENS = 128
DEFAULT_TRAIN_SIZE = 800
DEFAULT_NORMALIZE_TARGETS = True


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
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio from tokenized train set")
    p.add_argument("--val_every", type=int, default=DEFAULT_VAL_EVERY, help="Run validation loss every N optimizer steps; <=0 disables")
    p.add_argument("--val_max_batches", type=int, default=DEFAULT_VAL_MAX_BATCHES, help="Max validation batches per validation run; <=0 uses full validation set")
    p.add_argument("--probe_every", type=int, default=DEFAULT_PROBE_EVERY, help="Run sample inference probe every N optimizer steps; <=0 disables")
    p.add_argument("--probe_num_questions", type=int, default=DEFAULT_PROBE_NUM_QUESTIONS, help="Number of fixed training questions to probe")
    p.add_argument("--probe_max_new_tokens", type=int, default=DEFAULT_PROBE_MAX_NEW_TOKENS, help="Max new tokens for each probe generation")
    p.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of tasks used for training after shuffling; remaining tasks are held out for evaluation")
    p.add_argument(
        "--normalize_targets",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NORMALIZE_TARGETS,
        help="Normalize canonical solutions into stable Python code form before supervision",
    )
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")
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


def build_prompt_text(args, tokenizer, prompt: str) -> str:
    """Build a model input prompt that matches the downstream eval format."""
    if args.split == "instruct":
        user_content = (
            "Write Python code that solves the task. "
            "Output ONLY valid Python code (no markdown, no explanation).\n\n" + prompt.strip()
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Keep complete-mode prompts as raw text to mirror eval_bigcodebench_remote.py.
        return prompt.strip()


def _extract_task_func_signature(task_prompt: str) -> str:
    m = re.search(r"def\s+task_func\s*\((.*?)\)\s*:", task_prompt, flags=re.DOTALL)
    if not m:
        return ""
    return f"def task_func({m.group(1).strip()}):"


def normalize_canonical_solution(task_prompt: str, canonical_solution: str) -> str:
    cleaned = canonical_solution.replace("\r\n", "\n").strip("\n")
    if not cleaned:
        return "pass"

    # Dedent to avoid accidental top-level indentation in targets.
    cleaned = textwrap.dedent(cleaned)

    if re.search(r"(?m)^\s*def\s+task_func\s*\(", cleaned):
        return cleaned

    signature = _extract_task_func_signature(task_prompt)
    if not signature:
        return cleaned

    body = cleaned.strip()
    if not body:
        body = "pass"
    return signature + "\n" + textwrap.indent(body, "    ")


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

    # Deterministic split: first N for training, rest reserved for evaluation.
    if args.train_size > 0:
        if len(ds) <= args.train_size:
            raise ValueError(
                f"Dataset has {len(ds)} examples, cannot reserve a held-out split with train_size={args.train_size}."
            )
        ds = ds.shuffle(seed=args.seed)
        train_ds = ds.select(range(args.train_size))
        eval_ds = ds.select(range(args.train_size, len(ds)))
        print(f"Fixed split: train={len(train_ds)} | held-out eval={len(eval_ds)}")

        os.makedirs(args.output_dir, exist_ok=True)
        train_ids_path = os.path.join(args.output_dir, "train_task_ids.json")
        eval_ids_path = os.path.join(args.output_dir, "eval_task_ids.json")
        with open(train_ids_path, "w", encoding="utf-8") as f:
            json.dump([ex["task_id"] for ex in train_ds], f, indent=2)
        with open(eval_ids_path, "w", encoding="utf-8") as f:
            json.dump([ex["task_id"] for ex in eval_ds], f, indent=2)
        print(f"Wrote train task ids: {train_ids_path}")
        print(f"Wrote held-out eval task ids: {eval_ids_path}")
        ds = train_ds

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    # --- Corrected Tokenization Logic ---
    norm_stats = {"normalized": 0}

    def tok(example):
        prompt_key = f"{args.split}_prompt"
        raw_prompt = example[prompt_key]
        solution = str(example["canonical_solution"])
        if args.normalize_targets:
            normalized_solution = normalize_canonical_solution(raw_prompt, solution)
            if normalized_solution != solution.strip("\n"):
                norm_stats["normalized"] += 1
            solution = normalized_solution

        # 1. Build Prompt (using chat template, NO tokenization yet)
        prompt_text = build_prompt_text(args, tokenizer, raw_prompt)

        # 2. Build Solution (Add EOS manually to ensure model stops)
        # We add the eos_token explicitly.
        solution_text = solution + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer(solution_text, add_special_tokens=False)["input_ids"]
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

    # Keep a few fixed training prompts to monitor whether the model retains prior behavior.
    probe_examples: List[Dict[str, str]] = []
    if args.probe_num_questions > 0:
        probe_source = ds.shuffle(seed=args.seed).select(range(min(len(ds), args.probe_num_questions)))
        for ex in probe_source:
            prompt_key = f"{args.split}_prompt"
            probe_examples.append(
                {
                    "task_id": str(ex.get("task_id", "unknown")),
                    "prompt": str(ex.get(prompt_key, "")).strip(),
                }
            )

    tokenized = ds.map(tok, remove_columns=ds.column_names)
    if args.normalize_targets:
        print(f"Normalized canonical solutions: {norm_stats['normalized']} / {len(ds)}")

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

    def run_learning_probe(step_idx: int):
        if args.probe_every <= 0 or not probe_examples:
            return
        model.eval()
        print(f"\n[probe] step={step_idx} | sample questions from training set", flush=True)
        for ex in probe_examples:
            prompt_text = build_prompt_text(args, tokenizer, ex["prompt"])
            enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.probe_max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            pred_ids = out[0, enc["input_ids"].shape[1] :]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            q_preview = ex["prompt"].replace("\n", " ")[:140]
            a_preview = pred.replace("\n", " ")[:220]
            print(
                f"[probe] task_id={ex['task_id']} | q={q_preview!r} | pred={a_preview!r}",
                flush=True,
            )
        model.train()

    # Split train/validation from tokenized set
    if args.val_ratio > 0 and len(tokenized) > 1:
        shuffled = tokenized.shuffle(seed=args.seed)
        val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * args.val_ratio)))
        val_tokenized = shuffled.select(range(val_count))
        train_tokenized = shuffled.select(range(val_count, len(shuffled)))
        print(f"Train split: {len(train_tokenized)} | Val split: {len(val_tokenized)}")
    else:
        train_tokenized = tokenized
        val_tokenized = None

    # Data Loader
    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(train_tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if val_tokenized is not None else None

    def compute_val_loss():
        if val_loader is None:
            return None
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for vbatch in val_loader:
                vbatch = {k: v.to(device) for k, v in vbatch.items()}
                out = model(**vbatch)
                if torch.isfinite(out.loss):
                    val_loss_sum += out.loss.item()
                    val_batches += 1
                if args.val_max_batches > 0 and val_batches >= args.val_max_batches:
                    break
        model.train()
        return (val_loss_sum / val_batches) if val_batches > 0 else None

    # Optimizer & Schedule
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Calculate steps
    num_update_steps_per_epoch = len(loader) // args.grad_accum
    num_update_steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    max_train_steps = args.epochs * num_update_steps_per_epoch if args.max_steps <= 0 else min(args.max_steps, args.epochs * num_update_steps_per_epoch)
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print(f"Starting training: Epochs={args.epochs}, Batch={args.batch_size}, GradAccum={args.grad_accum}")
    print(f"Total optimization steps: {max_train_steps}")
    # progress bar and timing
    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")
    _step_time_accum = 0.0
    _step_time_count = 0

    # Training Loop
    global_step = 0
    total_loss = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loader):
            batch_start_time = time.perf_counter()
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

                if args.save_strategy == "steps" and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"\nSaving intermediate checkpoint at step {global_step} to {checkpoint_dir} ...", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                # record iteration time (counts only completed optimizer steps)
                iter_dt = time.perf_counter() - batch_start_time
                _step_time_accum += iter_dt
                _step_time_count += 1

                if global_step % args.log_every == 0:
                    avg_loss = total_loss / args.log_every
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    lr = scheduler.get_last_lr()[0]
                    val_msg = ""
                    if args.val_every > 0 and global_step % args.val_every == 0:
                        val_loss = compute_val_loss()
                        if val_loss is not None:
                            val_msg = f" | ValLoss: {val_loss:.4f}"
                    print(f"Step {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s{val_msg}")
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"})
                    total_loss = 0

                if args.probe_every > 0 and global_step % args.probe_every == 0:
                    run_learning_probe(global_step)

                pbar.update(1)
                if global_step >= max_train_steps:
                    break
        
        if global_step >= max_train_steps:
            break

    try:
        pbar.close()
    except Exception:
        pass

    # Save
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()