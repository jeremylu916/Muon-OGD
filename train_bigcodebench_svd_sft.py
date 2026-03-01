import argparse
import json
import math
import os
import random
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import time
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# --- Default Configuration ---
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_bigcodebench_svd"
DEFAULT_BCB_VERSION = "v0.1.4"
DEFAULT_SPLIT = "instruct"  # instruct|complete
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = None
DEFAULT_MAX_STEPS = 500
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def parse_args():
    p = argparse.ArgumentParser(description="SFT on BigCodeBench with SVD projection to reduce forgetting.")
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
    # SVD projection options
    p.add_argument("--svd_project", action="store_true", help="Enable SVD projection of weights after optimizer step")
    p.add_argument("--svd_every", type=int, default=1, help="Apply SVD projection every N optimizer steps")
    p.add_argument("--svd_layers", type=str, default="", help="Comma-separated substrings to match module names to project (default=all linear weights)")
    p.add_argument("--svd_rank", type=int, default=128, help="Default truncated rank for projection (if --svd_energy not set)")
    p.add_argument("--svd_energy", type=float, default=0.0, help="If >0, choose rank to retain this energy fraction (0-1). Overrides --svd_rank when set)")
    # Muon-OGD options
    p.add_argument("--muon_ogd", action="store_true", help="Enable Muon-OGD spectral-norm constrained projection after optimizer step")
    p.add_argument("--muon_T", type=int, default=5, help="Number of inner dual iterations T")
    p.add_argument("--muon_eta", type=float, default=1e-3, help="Primal step size eta")
    p.add_argument("--muon_eta_dual", type=float, default=1e-3, help="Dual step size eta_lambda")
    p.add_argument("--muon_k", type=int, default=4, help="Number of protected directions (k) per layer to extract from pretrained weights")
    p.add_argument("--muon_layers", type=str, default="", help="Comma-separated substrings to match module names to protect (default=all linear weights)")
    p.add_argument("--muon_msign_method", type=str, default="svd", choices=["svd"], help="Method to compute matrix sign (svd supported)")
    p.add_argument("--muon_warm_start", action="store_true", help="Warm-start dual variables lambda across optimizer steps (recommended)")
    p.add_argument("--svd_method", type=str, default="rand_gpu", choices=["exact_gpu", "rand_gpu"], help="Which SVD implementation to use (GPU exact or randomized on GPU)")
    p.add_argument("--time_profile", action="store_true", help="Print per-step/per-module timing for SVD and Muon-OGD")
    return p.parse_args()


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ----- SVD projection helpers -----
def _truncated_svd_project(W: torch.Tensor, rank: int = None, energy_thresh: float = None, device=None, svd_method: str = "rand_gpu", time_profile: bool = False) -> torch.Tensor:
    """Compute truncated SVD projection for 2D weight matrix W.

    SVD is computed on the same device as W (GPU if available). Supports exact SVD or randomized SVD on GPU.
    """
    assert W.ndim == 2, "SVD projection expects 2D weight matrix"

    orig_device = W.device
    orig_dtype = W.dtype

    X = W.detach().to(torch.float32).to(orig_device)

    # compute SVD on device
    if rank is None and energy_thresh and energy_thresh > 0.0:
        # compute full SVD (or large enough randomized) to get singular values for energy
        U, S, Vh = _compute_svd(X, k=None if svd_method == "exact_gpu" else min(X.shape), method=svd_method, time_profile=time_profile)
        sv2 = S * S
        cs = torch.cumsum(sv2, dim=0)
        total = cs[-1]
        k = int(torch.searchsorted(cs, energy_thresh * total).item()) + 1
    elif rank is not None:
        k = min(rank, min(X.shape))
        U, S, Vh = _compute_svd(X, k=k, method=svd_method, time_profile=time_profile)
    else:
        raise ValueError("Either rank or energy_thresh must be provided")

    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    Wk_fp32 = (U_k * S_k.unsqueeze(0)) @ Vh_k

    # Cast back to original dtype if needed
    Wk = Wk_fp32.to(orig_device).to(orig_dtype)
    return Wk


def _matrix_sign_via_svd(X: torch.Tensor) -> torch.Tensor:
    """Compute matrix sign (msgn) via SVD: msgn(X) = U * sign(S) * Vh.

    Works in float32 on CPU for stability, casts back to original dtype/device.
    """
    orig_device = X.device
    orig_dtype = X.dtype
    # compute SVD on the same device (GPU if available) in float32
    X_fp32 = X.detach().to(torch.float32).to(orig_device)
    U, S, Vh = torch.linalg.svd(X_fp32, full_matrices=False)
    signS = torch.sign(S)
    Ssign = signS.unsqueeze(0)
    msgn_fp32 = (U * Ssign) @ Vh
    return msgn_fp32.to(orig_device).to(orig_dtype)


def _compute_svd(X: torch.Tensor, k: int = None, method: str = "rand_gpu", n_oversamples: int = 8, n_iter: int = 1, time_profile: bool = False):
    """Compute SVD of X on the device of X.

    If method == 'exact_gpu', use torch.linalg.svd directly (on GPU if available).
    If method == 'rand_gpu', use randomized SVD on GPU and return approximate U, S, Vh.
    Returns (U, S, Vh) with shapes compatible with torch.linalg.svd.
    """
    import time
    t0 = time.perf_counter()
    dev = X.device
    Xf = X.detach().to(torch.float32).to(dev)

    if method == "exact_gpu" or k is None:
        U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
        if time_profile:
            print(f"[svd exact_gpu] shape={tuple(Xf.shape)} time={time.perf_counter()-t0:.3f}s")
        return U, S, Vh

    # randomized SVD
    n = Xf.shape[1]
    target = min(n, k + n_oversamples)
    # Draw a random gaussian test matrix on same device/dtype
    Omega = torch.randn((n, target), device=dev, dtype=Xf.dtype)
    Y = Xf @ Omega  # (m, target)
    # power iterations (optional)
    for _ in range(n_iter):
        Y = Xf @ (Xf.transpose(0, 1) @ Y)
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    B = Q.transpose(0, 1) @ Xf  # (target, n)
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    # Truncate to k if requested
    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]
    if time_profile:
        print(f"[svd rand_gpu] shape={tuple(Xf.shape)} k={k} time={time.perf_counter()-t0:.3f}s")
    return U, S, Vh


def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
        """
        Perform SVD on a 2D weight matrix and split into:
            - top_k singular vectors (treated as frozen/buffers)
            - the rest (treated as trainable)
        Returns a dictionary containing:
            {
                "U_high": ...  # buffer
                "S_high": ...  # buffer
                "V_high": ...  # buffer
                "U_low": ...   # parameter
                "S_low": ...   # parameter
                "V_low": ...   # parameter
                "rank_high": top_k
            }
        """
        device_local = weight.device
        W = weight.to(torch.float32)  # ensure float32 for SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Ensure we don’t ask for more than available
        k = min(top_k, S.shape[0])

        # High subspace (frozen)
        U_high = U[:, :k].detach().to(device_local)
        S_high = S[:k].detach().to(device_local)
        V_high = Vt[:k, :].detach().to(device_local)

        # Low subspace (trainable)
        U_low = U[:, k:].detach().to(device_local)
        S_low = S[k:].detach().to(device_local)
        V_low = Vt[k:, :].detach().to(device_local)

        return {
                "U_high": U_high,
                "S_high": S_high,
                "V_high": V_high,
                "U_low": nn.Parameter(U_low),
                "S_low": nn.Parameter(S_low),
                "V_low": nn.Parameter(V_low),
                "rank_high": k,
        }


def _extract_protected_directions_from_weight(W: torch.Tensor, k: int = 4) -> List[torch.Tensor]:
    """Extract up to k rank-1 protected directions Ci from pretrained weight W.

    Uses `decompose_weight_matrix` to split W into high (top-k) and low subspaces,
    then returns the top-k rank-1 components Ci = sigma_i * u_i v_i^T as CPU float32 tensors.
    """
    # Use decomposition helper to get top-k components
    svd_dict = decompose_weight_matrix(W.detach(), top_k=k)
    # Keep components on the original device to avoid transfers
    U_high = svd_dict["U_high"].to(torch.float32)
    S_high = svd_dict["S_high"].to(torch.float32)
    V_high = svd_dict["V_high"].to(torch.float32)

    ks = S_high.shape[0]
    Cis = []
    for i in range(ks):
        Ui = U_high[:, i : i + 1]
        Si = S_high[i]
        Vi = V_high[i : i + 1, :]
        Ci = (Ui * Si.unsqueeze(0)) @ Vi
        Cis.append(Ci)
    return Cis


def _muon_ogd_apply_on_weight(W: torch.nn.Parameter, Cis: List[torch.Tensor], G: torch.Tensor, eta: float, eta_dual: float, T: int, lam_init: torch.Tensor = None, time_profile: bool = False):
    """Apply Muon-OGD per-weight matrix. W is the parameter (tensor), Cis is list of protected directions (on W.device),
    G is the gradient tensor (same device/dtype as W). This function performs T inner dual iterations and returns the primal update Delta and final lambda.
    """
    import time
    t0 = time.perf_counter()
    dev = W.device
    # Move gradient to float32 on the same device
    G_fp32 = G.detach().to(torch.float32).to(dev)
    # Initialize lambda
    k = len(Cis)

    # Initialize lambda: warm-start if provided, else zeros (on device)
    if lam_init is not None:
        try:
            lam = lam_init.detach().to(torch.float32).to(dev).clone()
            if lam.numel() != k:
                lam = torch.zeros(k, dtype=torch.float32, device=dev)
        except Exception:
            lam = torch.zeros(k, dtype=torch.float32, device=dev)
    else:
        lam = torch.zeros(k, dtype=torch.float32, device=dev)

    if k == 0:
        # fallback: use plain sign update; return lam unchanged
        msgn = _matrix_sign_via_svd(G.to(dev))
        if time_profile:
            print(f"[muon] module shape={tuple(W.shape)} k=0 time={time.perf_counter()-t0:.3f}s")
        return (-eta) * msgn, lam

    for t in range(T):
        # Form H = G + sum_i lambda_i * Ci
        H = G_fp32.clone()
        for i in range(k):
            H = H + lam[i] * Cis[i].to(dev)

        # compute matrix sign (on device)
        S = _matrix_sign_via_svd(H).to(dev)

        # dual update: lambda <- lambda - eta_dual * <Ci, S>
        for i in range(k):
            inner = (Cis[i].to(dev) * S).sum()
            lam[i] = lam[i] - eta_dual * inner

    # After inner loop set H_final
    H_final = G_fp32.clone()
    for i in range(k):
        H_final = H_final + lam[i] * Cis[i].to(dev)

    msgn_final = _matrix_sign_via_svd(H_final)
    Delta = (-eta) * msgn_final
    if time_profile:
        print(f"[muon] module shape={tuple(W.shape)} k={k} T={T} time={time.perf_counter()-t0:.3f}s")
    # return both the primal update and the final lambda (on device)
    return Delta, lam


def project_model_weights_svd(model: torch.nn.Module, layer_filters: List[str] = None, rank: int = 128, energy_thresh: float = 0.0, device=None, time_profile: bool = False):
    dev = device or next(model.parameters()).device
    filters = [f for f in (layer_filters or []) if f]
    import time
    total_time = 0.0
    counts = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2:
            if filters:
                if not any(sub in name for sub in filters):
                    continue
            W = module.weight
            try:
                t0 = time.perf_counter()
                svd_method = getattr(model, "svd_method", "rand_gpu")
                Wk = _truncated_svd_project(
                    W.data,
                    rank=rank if energy_thresh <= 0.0 else None,
                    energy_thresh=(energy_thresh if energy_thresh > 0.0 else None),
                    device=dev,
                    svd_method=svd_method,
                    time_profile=time_profile,
                )
                module.weight.data.copy_(Wk)
                dt = time.perf_counter() - t0
                total_time += dt
                counts += 1
            except Exception as e:
                print(f"Warning: SVD project failed for {name}: {e}", flush=True)
    if counts > 0:
        print(f"SVD projection: applied to {counts} modules, total_time={total_time:.3f}s, avg={total_time/counts:.3f}s")

# ----- end SVD helpers -----


def build_prompt_text(args, prompt: str) -> List[Dict[str, str]]:
    if args.split == "instruct":
        user_content = (
            "Write Python code that solves the task. "
            "Output ONLY valid Python code (no markdown, no explanation).\n\n" + prompt.strip()
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.strip()},
        ]
    return messages


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading dataset {args.bcb_version}...")
    try:
        ds = load_dataset("bigcode/bigcodebench", split="train", cache_dir=cache_dir)
    except Exception:
        ds = load_dataset("bigcode/bigcodebench", split=args.bcb_version, cache_dir=cache_dir)

    if args.task_ids_file.strip():
        with open(args.task_ids_file, "r") as f:
            task_ids = json.load(f)
        if isinstance(task_ids, list):
            keep = set(task_ids)
            ds = ds.filter(lambda ex: ex["task_id"] in keep)
            print(f"Filtered to {len(ds)} tasks from file.")

    if args.num_train_examples and args.num_train_examples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_train_examples))
        print(f"Subsampled to {len(ds)} examples.")

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    def tok(example):
        prompt_key = f"{args.split}_prompt"
        raw_prompt = example[prompt_key]
        solution = example["canonical_solution"]

        messages = build_prompt_text(args, raw_prompt)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        solution_text = solution + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer(solution_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + solution_ids
        labels = ([-100] * len(prompt_ids)) + solution_ids

        if len(input_ids) > args.max_length:
            input_ids = input_ids[: args.max_length]
            labels = labels[: args.max_length]

        pad_len = args.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    print("Tokenizing dataset...")
    tokenized = ds.map(tok, remove_columns=ds.column_names)

    print("\n--- SANITY CHECK: Decoding first example ---")
    ex = tokenized[0]
    valid_labels = [l for l in ex["labels"] if l != -100]
    print(f"Full Input Decoded (First 200 chars): {tokenizer.decode(ex['input_ids'])[:200]}...")
    print(f"Labels Decoded (First 50 chars): {tokenizer.decode(valid_labels)[:50]}...")
    print("--------------------------------------------\n")

    print(f"Loading model {args.model_id}...")
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, torch_dtype=dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.train()
    # store chosen svd_method into model object for downstream helpers
    setattr(model, "svd_method", args.svd_method)

    # If Muon-OGD enabled: extract protected directions from pretrained model weights
    muon_C_map = {}
    muon_lambda_map = {}
    if args.muon_ogd:
        filters = [s.strip() for s in args.muon_layers.split(",")] if args.muon_layers else []
        print("Extracting protected directions for Muon-OGD...")
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2:
                if filters and not any(sub in name for sub in filters):
                    continue
                # extract top-k components as Ci (on module device)
                Cis = _extract_protected_directions_from_weight(module.weight.data, k=args.muon_k)
                muon_C_map[name] = Cis
                # initialize lambda per-module on module device; warm-start enabled if requested
                if args.muon_warm_start:
                    dev = module.weight.device
                    muon_lambda_map[name] = torch.zeros(len(Cis), dtype=torch.float32, device=dev)
        print(f"Extracted Muon-OGD Ci for {len(muon_C_map)} modules")

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = len(loader) // args.grad_accum
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    print(f"Starting training: Epochs={args.epochs}, Batch={args.batch_size}, GradAccum={args.grad_accum}")
    print(f"Total optimization steps: {max_train_steps}")

    # progress bar for optimization steps
    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")

    global_step = 0
    total_loss = 0
    # timing accumulators for tqdm reporting
    _step_time_accum = 0.0
    _step_time_count = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loader):
            batch_start_time = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                opt.step()
                scheduler.step()

                # Optionally apply SVD projection to reduce forgetting
                # Use (global_step+1) so projection runs on the just-completed step
                if args.svd_project and (((global_step + 1) % max(1, args.svd_every)) == 0):
                    filters = [s.strip() for s in args.svd_layers.split(",")] if args.svd_layers else []
                    try:
                        project_model_weights_svd(
                            model,
                            layer_filters=filters,
                            rank=args.svd_rank,
                            energy_thresh=args.svd_energy,
                            device=device,
                            time_profile=args.time_profile,
                        )
                    except Exception as e:
                        print(f"SVD projection failed at step {global_step}: {e}", flush=True)

                # Optionally apply Muon-OGD constrained projection per-weight
                if args.muon_ogd:
                    try:
                        filters = [s.strip() for s in args.muon_layers.split(",")] if args.muon_layers else []
                        for name, module in model.named_modules():
                            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2:
                                if filters and not any(sub in name for sub in filters):
                                    continue
                                Cis = muon_C_map.get(name, [])
                                if module.weight.grad is None:
                                    continue
                                try:
                                    lam_init = None
                                    if args.muon_warm_start:
                                        lam_init = muon_lambda_map.get(name, None)
                                    Delta, lam_final = _muon_ogd_apply_on_weight(
                                        module.weight,
                                        Cis,
                                        module.weight.grad.detach(),
                                        eta=args.muon_eta,
                                        eta_dual=args.muon_eta_dual,
                                        T=args.muon_T,
                                        lam_init=lam_init,
                                        time_profile=args.time_profile,
                                    )
                                    # store final lambda if warm-start enabled
                                    if args.muon_warm_start and lam_final is not None:
                                        muon_lambda_map[name] = lam_final.detach()
                                    # apply primal update (in-place)
                                    module.weight.data.add_(Delta)
                                except Exception as e:
                                    print(f"Muon-OGD failed for {name}: {e}", flush=True)
                    except Exception as e:
                        print(f"Muon-OGD outer failure at step {global_step}: {e}", flush=True)

                opt.zero_grad()
                global_step += 1

                # record iteration time (counts only completed optimizer steps)
                iter_dt = time.perf_counter() - batch_start_time
                _step_time_accum += iter_dt
                _step_time_count += 1

                if global_step % args.log_every == 0:
                    avg_loss = total_loss * args.grad_accum / args.log_every
                    lr = scheduler.get_last_lr()[0]
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    msg = {"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"}
                    pbar.set_postfix(msg)
                    print(f"Step {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s")
                    total_loss = 0

                # update tqdm
                pbar.update(1)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")
    try:
        pbar.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
