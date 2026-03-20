import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# --- Default Configuration ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_gsm8k_muon_ogd_algo2"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-5
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = 2000
DEFAULT_MAX_STEPS = 1800
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10
DEFAULT_PROBE_EVERY = 100
DEFAULT_PROBE_MAX_NEW_TOKENS = 64
DEFAULT_VAL_RATIO = 0.02
DEFAULT_VAL_EVERY = 100
DEFAULT_VAL_MAX_BATCHES = 32

FIXED_PROBES = [
    ("Q1", "Explain Machine Learning in 1 sentence.", "Machine learning is a method where models learn patterns from data to make predictions or decisions without explicit rules."),
    ("Q2", "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "$10"),
    (
        "Q3",
        "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?",
        "Patent foramen ovale (PFO) causing paradoxical embolism.",
    ),
]


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def parse_args():
    p = argparse.ArgumentParser(description="SFT on GSM8K with Hybrid AdamW + Muon-OGD (Algorithm 2 Bilinear Subspace).")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
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
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"])
    p.add_argument("--save_steps", type=int, default=500)

    # Performance knobs
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_autocast", action="store_true")
    p.add_argument("--tf32", action="store_true")

    # Muon-OGD options
    p.add_argument("--muon_ogd", action="store_true")
    p.add_argument("--muon_T", type=int, default=1)
    p.add_argument("--muon_eta", type=float, default=1e-5)
    p.add_argument("--muon_eta_dual", type=float, default=1e-4)
    p.add_argument("--muon_k", type=int, default=1)
    p.add_argument("--muon_layers", type=str, default="o_proj,down_proj")
    p.add_argument("--muon_msign_method", type=str, default="ns", choices=["svd", "ns"])
    p.add_argument("--muon_ns_iters", type=int, default=6)
    p.add_argument("--muon_warm_start", action="store_true")
    p.add_argument("--muon_dynamic_scale", action="store_true")
    p.add_argument("--ci_model_id", type=str, default="")
    
    # Less-forgetting knobs (optional)
    p.add_argument("--ci_from_grads", action="store_true")
    p.add_argument("--ci_replay_dataset", type=str, default="")
    p.add_argument("--ci_replay_config", type=str, default="")
    p.add_argument("--ci_replay_split", type=str, default="train")
    p.add_argument("--ci_replay_examples", type=int, default=128)
    p.add_argument("--ci_replay_seed", type=int, default=123)

    return p.parse_args()


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =========================================================================
# Algorithm 2 Math Helpers
# =========================================================================

@torch.no_grad()
def _msgn(X: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """Fast polar factor via Newton-Schulz iterations."""
    dev = X.device
    A = X.detach().to(torch.float32)
    m, n = A.shape
    frob = torch.linalg.norm(A, ord="fro")
    if frob < eps:
        return torch.zeros_like(X)
    A = A / (frob + eps)
    
    transposed = False
    if m < n:
        A = A.transpose(0, 1)
        transposed = True
        m, n = A.shape
        
    I = torch.eye(n, device=dev, dtype=torch.float32)
    Y = A
    Z = I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y.transpose(0, 1) @ Y)
        Y = Y @ T
        Z = T @ Z
        
    if transposed:
        Y = Y.transpose(0, 1)
    return Y

def extract_subspace_matrices(mat: torch.Tensor, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns top-k orthogonal basis matrices U (m x k) and V (n x k) for Algorithm 2."""
    if mat.ndim != 2:
        return None, None
    m = mat.detach().to(torch.float32).to(device)
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    top = min(k, S.shape[0])
    
    U_k = U[:, :top].contiguous()
    V_k = Vh[:top, :].transpose(0, 1).contiguous()
    return U_k, V_k


# =========================
# GSM8K prompt builder
# =========================

def build_prompt(tokenizer, question: str, answer: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Solve the problem and give the final answer.\n\n{question}"},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Dataset ----
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    if args.num_train_examples and args.num_train_examples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))

    # ---- Tokenizer ----
    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    def tok(example):
        text = build_prompt(tokenizer, example["question"], example["answer"])
        prompt_only_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Solve the problem and give the final answer.\n\n{example['question']}"},
        ]
        prompt_only = tokenizer.apply_chat_template(
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        if len(full_ids) > args.max_length:
            full_ids = full_ids[: args.max_length]
            labels = labels[: args.max_length]

        pad_len = args.max_length - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}

    print("Tokenizing dataset...")
    tokenized = dataset.map(tok, remove_columns=dataset.column_names)

    if args.val_ratio > 0 and len(tokenized) > 1:
        shuffled = tokenized.shuffle(seed=args.seed)
        val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * args.val_ratio)))
        val_tokenized = shuffled.select(range(val_count))
        train_tokenized = shuffled.select(range(val_count, len(shuffled)))
    else:
        train_tokenized = tokenized
        val_tokenized = None

    # ---- Model ----
    print(f"Loading model {args.model_id}...")
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (
        torch.float16 if use_cuda else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, torch_dtype=dtype)
    model.to(device)
    model.train()

    def run_probe(step_idx: int):
        if args.probe_every <= 0: return
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
                    **enc, max_new_tokens=args.probe_max_new_tokens, do_sample=False,
                    temperature=1.0, top_p=1.0, top_k=50,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )

            pred_ids = out[0, enc["input_ids"].shape[1]:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            print(f"[probe] step={step_idx} | {probe_id} | q={probe_q[:120]!r} | ****pred={pred[:160]!r} | ****target={probe_t[:160]!r}", flush=True)
        model.train()

    # =========================================================================
    # Algorithm 2 Subspace Extraction (U and V)
    # =========================================================================
    muon_targets = {}
    muon_C_map = {}
    muon_lambda_map = {} # Store dual matrix Lambda

    if args.muon_ogd:
        filters = [s.strip() for s in args.muon_layers.split(",")] if args.muon_layers else []
        print("Collecting Muon target modules...")
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2:
                if filters and not any(sub in name for sub in filters): continue
                if not module.weight.requires_grad: continue
                muon_targets[name] = module
        print(f"Muon targets: {len(muon_targets)} modules")

        if args.ci_from_grads:
            replay_name = args.ci_replay_dataset.strip()
            replay_cfg = args.ci_replay_config.strip()
            replay_split = args.ci_replay_split.strip()

            if replay_name:
                print(f"Loading replay dataset for Ci from grads: {replay_name} ({replay_cfg or 'default'}) split={replay_split}")
                if replay_cfg: replay_ds = load_dataset(replay_name, replay_cfg, split=replay_split, cache_dir=cache_dir)
                else: replay_ds = load_dataset(replay_name, split=replay_split, cache_dir=cache_dir)
            else:
                print("ci_from_grads enabled but no ci_replay_dataset provided; using GSM8K train as replay.")
                replay_ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)

            n_replay = min(args.ci_replay_examples, len(replay_ds))
            replay_ds = replay_ds.shuffle(seed=args.ci_replay_seed).select(range(n_replay))

            def replay_tok(example):
                if "question" not in example or "answer" not in example:
                    raise ValueError("Replay dataset examples must have 'question' and 'answer' fields. Adapt replay_tok() for your dataset.")
                return tok(example)

            print("Tokenizing replay dataset for Ci-from-grads...")
            replay_tokenized = replay_ds.map(replay_tok, remove_columns=replay_ds.column_names)

            def replay_collate(batch):
                return {
                    "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
                    "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
                    "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
                }

            replay_loader = DataLoader(replay_tokenized, batch_size=1, shuffle=False, collate_fn=replay_collate, num_workers=0)

            print(f"Accumulating old-task gradients over {n_replay} replay examples...")
            G_acc: Dict[str, torch.Tensor] = {name: torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device) for name, mod in muon_targets.items()}

            model.zero_grad(set_to_none=True)
            model.train()
            for batch in replay_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                out.loss.backward()
                for name, mod in muon_targets.items():
                    if mod.weight.grad is not None:
                        G_acc[name].add_(mod.weight.grad.detach().to(torch.float32))
                model.zero_grad(set_to_none=True)

            print(f"Algorithm 2: Extracting Orthogonal U/V constraint matrices (k={args.muon_k})...")
            for name, mod in muon_targets.items():
                U_k, V_k = extract_subspace_matrices(G_acc[name], args.muon_k, device)
                muon_C_map[name] = (U_k, V_k)
            print("Ci-from-grads ready.")
            
        else:
            ci_model_id = args.ci_model_id.strip() if args.ci_model_id.strip() else args.model_id
            if ci_model_id != args.model_id:
                print(f"Loading reference model for Ci extraction: {ci_model_id} (CPU, float32)...")
                ci_ref_model = AutoModelForCausalLM.from_pretrained(ci_model_id, cache_dir=cache_dir, torch_dtype=torch.float32)
                ci_ref_model.eval()
                ci_weight_map: Dict[str, torch.Tensor] = {}
                for ref_name, ref_module in ci_ref_model.named_modules():
                    if ref_name in muon_targets and hasattr(ref_module, "weight"):
                        ci_weight_map[ref_name] = ref_module.weight.data.detach().clone().cpu()
                del ci_ref_model
                if use_cuda: torch.cuda.empty_cache()
                print(f"Reference model freed. Extracted weight snapshots for {len(ci_weight_map)} modules.")
            else:
                print("Snapshotting current weights for Ci (ci_model_id == model_id).")
                ci_weight_map = {name: mod.weight.data.detach().clone().cpu() for name, mod in muon_targets.items()}

            print(f"Algorithm 2: Extracting Orthogonal U/V constraint matrices from reference weights (k={args.muon_k})...")
            for name, mod in muon_targets.items():
                src_weight = ci_weight_map.get(name, mod.weight.data)
                U_k, V_k = extract_subspace_matrices(src_weight, args.muon_k, device)
                muon_C_map[name] = (U_k, V_k)

    # ---- Optimizer ----
    muon_weight_ids = set(id(module.weight) for _, module in muon_targets.items()) if args.muon_ogd else set()
    opt_params = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_weight_ids]
    print(f"AdamW params: {len(opt_params)} tensors (excluded {len(muon_weight_ids)} muon weight tensors)")
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(train_tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=max(0, args.num_workers), pin_memory=bool(args.pin_memory and use_cuda), persistent_workers=bool(args.num_workers > 0), prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if val_tokenized is not None else None

    def compute_val_loss():
        if val_loader is None: return None
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                val_out = model(**batch)
                if torch.isfinite(val_out.loss):
                    val_loss_sum += val_out.loss.item()
                    val_batches += 1
                if args.val_max_batches > 0 and val_batches >= args.val_max_batches: break
        model.train()
        return val_loss_sum / val_batches if val_batches > 0 else None

    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")
    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")
    global_step, total_loss = 0, 0.0
    _step_time_accum, _step_time_count = 0.0, 0

    use_autocast = bool(args.use_autocast and use_cuda)
    autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    # =========================================================================
    # Training Loop
    # =========================================================================
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch_start_time = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    outputs = model(**batch)
                    loss = outputs.loss / args.grad_accum
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            if not torch.isfinite(outputs.loss):
                print(f"[skip] Step {global_step} batch {step}: non-finite loss={outputs.loss.item():.4f}, skipping.", flush=True)
                model.zero_grad(set_to_none=True)
                continue

            loss.backward()
            total_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    print(f"[skip] Step {global_step}: NaN/Inf in gradients, skipping.", flush=True)
                    model.zero_grad(set_to_none=True)
                    continue

                if args.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)

                opt.step()
                scheduler.step()

                # =============================================================
                # Algorithm 2: Inline Muon-OGD Matrix Projection Step
                # =============================================================
                if args.muon_ogd:
                    with torch.no_grad():
                        for name, module in muon_targets.items():
                            if module.weight.grad is None: continue
                            
                            if args.weight_decay > 0:
                                module.weight.data.mul_(1.0 - args.muon_eta * args.weight_decay)

                            G_fp32 = module.weight.grad.to(torch.float32)
                            U, V = muon_C_map.get(name, (None, None))
                            
                            if U is not None and V is not None:
                                k = U.shape[1]
                                lam = muon_lambda_map.get(name, torch.zeros((k, k), dtype=torch.float32, device=device))
                                
                                for _ in range(args.muon_T):
                                    H = G_fp32 + U @ lam @ V.transpose(0, 1)
                                    S = _msgn(H, iters=args.muon_ns_iters)
                                    grad_lam = U.transpose(0, 1) @ S @ V
                                    lam = lam - args.muon_eta_dual * grad_lam
                                
                                if args.muon_warm_start: muon_lambda_map[name] = lam
                                H = G_fp32 + U @ lam @ V.transpose(0, 1)
                            else:
                                H = G_fp32
                            
                            S = _msgn(H, iters=args.muon_ns_iters)
                            
                            scale = 1.0
                            if args.muon_dynamic_scale:
                                scale = torch.linalg.norm(G_fp32).item() / max(1e-8, torch.linalg.norm(S).item())
                            
                            Delta = -args.muon_eta * scale * S
                            module.weight.data.add_(Delta.to(module.weight.dtype))
                # =============================================================

                model.zero_grad(set_to_none=True)
                global_step += 1

                if args.save_strategy == "steps" and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                _step_time_accum += time.perf_counter() - batch_start_time
                _step_time_count += 1

                if global_step % args.log_every == 0:
                    avg_loss = total_loss * args.grad_accum / args.log_every
                    lr = scheduler.get_last_lr()[0]
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    val_msg = ""
                    if args.val_every > 0 and global_step % args.val_every == 0:
                        val_loss = compute_val_loss()
                        if val_loss is not None: val_msg = f" | ValLoss: {val_loss:.4f}"
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"})
                    print(f"Step {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s{val_msg}")
                    total_loss = 0.0

                if args.probe_every > 0 and global_step % args.probe_every == 0: run_probe(global_step)
                pbar.update(1)
                if global_step >= max_train_steps: break
        if global_step >= max_train_steps: break

    pbar.close()
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()