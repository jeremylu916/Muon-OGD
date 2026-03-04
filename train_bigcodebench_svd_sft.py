import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from muon_ogd_optimizer import MuonOGDOptimizer

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
    p = argparse.ArgumentParser(description="SFT on BigCodeBench with SVD projection + Muon-OGD (fast msgn via Newton-Schulz).")
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
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")
    p.add_argument("--task_ids_file", type=str, default="")

    # SVD projection options
    p.add_argument("--svd_project", action="store_true", help="Enable SVD projection of weights after optimizer step")
    p.add_argument("--svd_every", type=int, default=1, help="Apply SVD projection every N optimizer steps")
    p.add_argument("--svd_layers", type=str, default="", help="Comma-separated substrings to match module names to project (default=all linear weights)")
    p.add_argument("--svd_rank", type=int, default=128, help="Default truncated rank for projection (if --svd_energy not set)")
    p.add_argument("--svd_energy", type=float, default=0.0, help="If >0, choose rank to retain this energy fraction (0-1). Overrides --svd_rank when set)")
    p.add_argument("--svd_method", type=str, default="rand_gpu", choices=["exact_gpu", "rand_gpu"], help="Which SVD implementation to use (GPU exact or randomized on GPU)")

    # Muon-OGD options
    p.add_argument("--muon_ogd", action="store_true", help="Enable Muon-OGD spectral-norm constrained projection/update")
    p.add_argument("--muon_T", type=int, default=5, help="Number of inner dual iterations T")
    p.add_argument("--muon_eta", type=float, default=1e-3, help="Primal step size eta")
    p.add_argument("--muon_eta_dual", type=float, default=1e-3, help="Dual step size eta_lambda")
    p.add_argument("--muon_k", type=int, default=4, help="Number of protected directions (k) per layer to extract from pretrained weights")
    p.add_argument("--muon_layers", type=str, default="", help="Comma-separated substrings to match module names to protect (default=all linear weights)")
    # NEW: allow fast msgn
    p.add_argument("--muon_msign_method", type=str, default="ns", choices=["svd", "ns"], help="Matrix sign / polar factor: 'svd' (slow) or 'ns' (Newton-Schulz fast)")
    p.add_argument("--muon_ns_iters", type=int, default=6, help="Newton-Schulz iterations for polar factor when --muon_msign_method ns")
    p.add_argument("--ci_model_id", type=str, default="", help="Model to extract protected Ci directions FROM (pretrained-task model). Defaults to --model_id if not set. Load on CPU and free after extraction.")
    p.add_argument("--muon_warm_start", action="store_true", help="Warm-start dual variables lambda across optimizer steps (recommended)")
    p.add_argument("--muon_use_optimizer_class", action="store_true", help="Use MuonOGDOptimizer class instead of manual per-layer Muon loop")
    p.add_argument("--muon_momentum", type=float, default=0.0, help="Momentum EMA coefficient for Muon optimizer class")
    p.add_argument("--muon_dynamic_scale", action="store_true", help="Enable dynamic per-layer scaling in Muon optimizer class")

    # Optional: build Ci from replay gradients (same path as GSM8K script)
    p.add_argument("--ci_from_grads", action="store_true", help="Build Ci from reference (old-task) gradients instead of weight SVD")
    p.add_argument("--ci_replay_dataset", type=str, default="", help="HF dataset name for old task replay. If empty and ci_from_grads, falls back to GSM8K.")
    p.add_argument("--ci_replay_config", type=str, default="", help="HF dataset config name for replay dataset (optional)")
    p.add_argument("--ci_replay_split", type=str, default="train", help="Split name for replay dataset")
    p.add_argument("--ci_replay_examples", type=int, default=128, help="Number of replay examples for gradient-based Ci")
    p.add_argument("--ci_replay_seed", type=int, default=123, help="Seed for replay sampling")

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


# =========================
# SVD projection helpers
# =========================
def _compute_svd(
    X: torch.Tensor,
    k: int = None,
    method: str = "rand_gpu",
    n_oversamples: int = 8,
    n_iter: int = 1,
    time_profile: bool = False,
):
    """Compute SVD of X on X.device.

    If method == 'exact_gpu' OR k is None -> torch.linalg.svd (full)
    If method == 'rand_gpu' and k provided -> randomized SVD (still one SVD of small B).
    Returns (U, S, Vh) shaped like torch.linalg.svd(full_matrices=False).
    """
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
    Omega = torch.randn((n, target), device=dev, dtype=Xf.dtype)
    Y = Xf @ Omega
    for _ in range(n_iter):
        Y = Xf @ (Xf.transpose(0, 1) @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.transpose(0, 1) @ Xf
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]
    if time_profile:
        print(f"[svd rand_gpu] shape={tuple(Xf.shape)} k={k} time={time.perf_counter()-t0:.3f}s")
    return U, S, Vh


def _truncated_svd_project(
    W: torch.Tensor,
    rank: int = None,
    energy_thresh: float = None,
    svd_method: str = "rand_gpu",
    time_profile: bool = False,
) -> torch.Tensor:
    """Truncated SVD projection for 2D weight matrix W."""
    assert W.ndim == 2, "SVD projection expects 2D weight matrix"

    orig_device = W.device
    orig_dtype = W.dtype

    X = W.detach().to(torch.float32).to(orig_device)

    if rank is None and energy_thresh and energy_thresh > 0.0:
        # need singular values: full SVD (or large enough)
        U, S, Vh = _compute_svd(
            X,
            k=None if svd_method == "exact_gpu" else min(X.shape),
            method=svd_method,
            time_profile=time_profile,
        )
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
    return Wk_fp32.to(orig_device).to(orig_dtype)


def project_model_weights_svd(
    model: torch.nn.Module,
    layer_filters: List[str] = None,
    rank: int = 128,
    energy_thresh: float = 0.0,
    time_profile: bool = False,
):
    filters = [f for f in (layer_filters or []) if f]
    total_time = 0.0
    counts = 0

    svd_method = getattr(model, "svd_method", "rand_gpu")

    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2:
            if filters and not any(sub in name for sub in filters):
                continue
            W = module.weight
            try:
                t0 = time.perf_counter()
                Wk = _truncated_svd_project(
                    W.data,
                    rank=rank if energy_thresh <= 0.0 else None,
                    energy_thresh=(energy_thresh if energy_thresh > 0.0 else None),
                    svd_method=svd_method,
                    time_profile=time_profile,
                )
                module.weight.data.copy_(Wk)
                total_time += (time.perf_counter() - t0)
                counts += 1
            except Exception as e:
                print(f"Warning: SVD project failed for {name}: {e}", flush=True)

    if counts > 0:
        print(f"SVD projection: applied to {counts} modules, total_time={total_time:.3f}s, avg={total_time/counts:.3f}s")


# =========================
# Muon helpers
# =========================
@dataclass
class Rank1C:
    sigma: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


def decompose_weight_matrix(weight: torch.Tensor, top_k: int):
    """SVD split into high (frozen) and low (trainable). Here we only use high as buffers."""
    device_local = weight.device
    W = weight.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])

    U_high = U[:, :k].detach().to(device_local)
    S_high = S[:k].detach().to(device_local)
    V_high = Vt[:k, :].detach().to(device_local)

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


def _extract_rank1_factors_from_matrix(
    M: torch.Tensor,
    k: int = 1,
    device: Optional[torch.device] = None,
    svd_method: str = "randomized",
    niter: int = 2,
    n_oversamples: int = 8,
) -> List[Rank1C]:
    dev = device or M.device
    Mf = M.detach().to(torch.float32).to(dev)
    k_actual = min(k, min(Mf.shape))
    if k_actual <= 0:
        return []

    if svd_method == "exact":
        U, S, Vh = torch.linalg.svd(Mf, full_matrices=False)
        U, S, Vh = U[:, :k_actual], S[:k_actual], Vh[:k_actual, :]
    else:
        q = min(min(Mf.shape), k_actual + n_oversamples)
        U, S, V = torch.svd_lowrank(Mf, q=q, niter=niter)
        U, S, Vh = U[:, :k_actual], S[:k_actual], V[:, :k_actual].transpose(0, 1)

    Cs: List[Rank1C] = []
    for i in range(k_actual):
        Cs.append(
            Rank1C(
                sigma=S[i].detach(),
                u=U[:, i].detach().contiguous(),
                v=Vh[i, :].detach().contiguous(),
            )
        )
    return Cs


def _rank1_inner_products(Cs: List[Rank1C], S_mat_fp32: torch.Tensor) -> torch.Tensor:
    if not Cs:
        return torch.zeros(0, device=S_mat_fp32.device, dtype=torch.float32)
    vals = []
    for c in Cs:
        vals.append(c.sigma * (c.u @ (S_mat_fp32 @ c.v)))
    return torch.stack(vals, dim=0)


def _add_rank1_shift_(H_fp32: torch.Tensor, Cs: List[Rank1C], lam_fp32: torch.Tensor) -> torch.Tensor:
    for i, c in enumerate(Cs):
        alpha = lam_fp32[i] * c.sigma
        H_fp32.add_(alpha * (c.u.unsqueeze(1) @ c.v.unsqueeze(0)))
    return H_fp32


def _matrix_sign_via_svd(X: torch.Tensor) -> torch.Tensor:
    """Slow: msgn/polar factor via SVD (kept for debugging)."""
    orig_device = X.device
    orig_dtype = X.dtype
    X_fp32 = X.detach().to(torch.float32).to(orig_device)
    U, S, Vh = torch.linalg.svd(X_fp32, full_matrices=False)
    # polar factor: U @ Vh
    P = U @ Vh
    return P.to(orig_device).to(orig_dtype)


@torch.no_grad()
def _polar_newton_schulz(X: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """Fast polar factor via Newton–Schulz iterations (matmul-only).
    Works for rectangular matrices too.

    Returns P approx polar(X), i.e., X ≈ P H, with P having orthonormal cols (if m>=n) or rows (if m<n).
    """
    dev = X.device
    orig_dtype = X.dtype

    A = X.detach().to(torch.float32).to(dev)
    m, n = A.shape

    # Scale for convergence
    frob = torch.linalg.norm(A, ord="fro")
    if frob < eps:
        return torch.zeros_like(X)

    A = A / (frob + eps)

    if m >= n:
        # Y: (m,n), Z: (n,n)
        Y = A
        I = torch.eye(n, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            # T = 0.5 * (3I - ZY)
            ZY = Z @ Y.transpose(0, 1) @ Y  # (n,n) approximate, stabilizes
            T = 0.5 * (3.0 * I - ZY)
            Y = Y @ T
            Z = T @ Z
        P = Y
    else:
        # Use transpose trick when wide: polar(A) = polar(A^T)^T
        At = A.transpose(0, 1)  # (n,m) with n>m
        # Now "tall"
        Y = At
        I = torch.eye(m, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            ZY = Z @ Y.transpose(0, 1) @ Y  # (m,m)
            T = 0.5 * (3.0 * I - ZY)
            Y = Y @ T
            Z = T @ Z
        P = Y.transpose(0, 1)

    return P.to(dev).to(orig_dtype)


def _msgn(X: torch.Tensor, method: str = "ns", ns_iters: int = 6) -> torch.Tensor:
    """Muon uses matrix sign / polar factor. We return polar factor approximation."""
    if method == "svd":
        return _matrix_sign_via_svd(X)
    # default fast
    return _polar_newton_schulz(X, iters=ns_iters)


def _muon_ogd_apply_on_weight_factorized(
    W: torch.nn.Parameter,
    Cs: List[Rank1C],
    G: torch.Tensor,
    eta: float,
    eta_dual: float,
    T: int,
    lam_init: Optional[torch.Tensor] = None,
    msign_method: str = "ns",
    ns_iters: int = 6,
    time_profile: bool = False,
):
    """Muon-OGD on a single 2D weight matrix.

    Returns (Delta, lam_final). Delta is float32 on device; caller can add_ to weight.
    """
    t0 = time.perf_counter()
    dev = W.device

    G_fp32 = G.detach().to(torch.float32).to(dev)
    k = len(Cs)

    if lam_init is not None:
        lam = lam_init.detach().to(torch.float32).to(dev).clone()
        if lam.numel() != k:
            lam = torch.zeros(k, dtype=torch.float32, device=dev)
    else:
        lam = torch.zeros(k, dtype=torch.float32, device=dev)

    if k == 0:
        S = _msgn(G_fp32, method=msign_method, ns_iters=ns_iters).to(torch.float32)
        if time_profile:
            print(f"[muon] shape={tuple(W.shape)} k=0 time={time.perf_counter()-t0:.3f}s")
        return (-eta) * S, lam

    for _ in range(T):
        H = G_fp32.clone()
        _add_rank1_shift_(H, Cs, lam)
        S = _msgn(H, method=msign_method, ns_iters=ns_iters).to(torch.float32)
        inner = _rank1_inner_products(Cs, S)
        lam = lam - eta_dual * inner

    H_final = G_fp32.clone()
    _add_rank1_shift_(H_final, Cs, lam)
    S_final = _msgn(H_final, method=msign_method, ns_iters=ns_iters).to(torch.float32)
    Delta = (-eta) * S_final

    if time_profile:
        print(f"[muon] shape={tuple(W.shape)} k={k} T={T} time={time.perf_counter()-t0:.3f}s")
    return Delta, lam


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
    setattr(model, "svd_method", args.svd_method)

    # -------------------------
    # Prepare Muon target modules
    # -------------------------
    muon_targets: Dict[str, nn.Module] = {}
    muon_C_map: Dict[str, List[Rank1C]] = {}
    muon_lambda_map: Dict[str, torch.Tensor] = {}

    if args.muon_ogd:
        filters = [s.strip() for s in args.muon_layers.split(",")] if args.muon_layers else []
        print("Collecting Muon target modules...")
        for name, module in model.named_modules():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2:
                if filters and not any(sub in name for sub in filters):
                    continue
                if not module.weight.requires_grad:
                    continue
                muon_targets[name] = module
        print(f"Muon targets: {len(muon_targets)} modules")

        if args.ci_from_grads:
            replay_name = args.ci_replay_dataset.strip()
            replay_cfg = args.ci_replay_config.strip()
            replay_split = args.ci_replay_split.strip()

            if replay_name:
                print(f"Loading replay dataset for Ci from grads: {replay_name} ({replay_cfg or 'default'}) split={replay_split}")
                if replay_cfg:
                    replay_ds = load_dataset(replay_name, replay_cfg, split=replay_split, cache_dir=cache_dir)
                else:
                    replay_ds = load_dataset(replay_name, split=replay_split, cache_dir=cache_dir)
            else:
                print("ci_from_grads enabled but no ci_replay_dataset provided; using GSM8K train as replay.")
                replay_ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)

            n_replay = min(args.ci_replay_examples, len(replay_ds))
            replay_ds = replay_ds.shuffle(seed=args.ci_replay_seed).select(range(n_replay))

            def replay_tok(example):
                if "question" not in example or "answer" not in example:
                    raise ValueError(
                        "Replay dataset examples must have 'question' and 'answer' fields "
                        "(GSM8K format). Adapt replay_tok() for your dataset."
                    )
                prompt_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"Solve the problem and give the final answer.\\n\\n{example['question']}",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                solution_text = example["answer"] + tokenizer.eos_token
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
            G_acc: Dict[str, torch.Tensor] = {}
            for name, mod in muon_targets.items():
                G_acc[name] = torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device)

            model.zero_grad(set_to_none=True)
            model.train()
            for replay_batch in replay_loader:
                replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
                out = model(**replay_batch)
                out.loss.backward()
                for name, mod in muon_targets.items():
                    if mod.weight.grad is not None:
                        G_acc[name].add_(mod.weight.grad.detach().to(torch.float32))
                model.zero_grad(set_to_none=True)

            print(f"Extracting protected directions Ci from accumulated replay gradients (k={args.muon_k})...")
            for name, mod in muon_targets.items():
                Cs = _extract_rank1_factors_from_matrix(G_acc[name], k=args.muon_k, device=mod.weight.device)
                muon_C_map[name] = Cs
                if args.muon_warm_start:
                    muon_lambda_map[name] = torch.zeros(len(Cs), dtype=torch.float32, device=mod.weight.device)
            print("Ci-from-grads ready.")
        else:
            ci_model_id = args.ci_model_id.strip() if args.ci_model_id.strip() else args.model_id
            if ci_model_id != args.model_id:
                print(f"Loading reference model for Ci extraction: {ci_model_id} (CPU, float32)...")
                ci_ref_model = AutoModelForCausalLM.from_pretrained(
                    ci_model_id, cache_dir=cache_dir, torch_dtype=torch.float32
                )
                ci_ref_model.eval()
                ci_weight_map: Dict[str, torch.Tensor] = {}
                for ref_name, ref_module in ci_ref_model.named_modules():
                    if ref_name in muon_targets and hasattr(ref_module, "weight"):
                        ci_weight_map[ref_name] = ref_module.weight.data.detach().clone().cpu()
                del ci_ref_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Reference model freed. Extracted weight snapshots for {len(ci_weight_map)} modules.")
            else:
                print(f"Snapshotting weights for Ci from: {ci_model_id} (same as model_id — no extra load)")
                ci_weight_map = {}
                for name, module in muon_targets.items():
                    ci_weight_map[name] = module.weight.data.detach().clone().cpu()

            print(f"Extracting protected directions Ci from reference weights (factorized SVD, k={args.muon_k})...")
            for name, module in muon_targets.items():
                src_weight = ci_weight_map.get(name, module.weight.data)
                Cs = _extract_rank1_factors_from_matrix(src_weight, k=args.muon_k, device=module.weight.device)
                muon_C_map[name] = Cs
                if args.muon_warm_start:
                    muon_lambda_map[name] = torch.zeros(len(Cs), dtype=torch.float32, device=module.weight.device)
            print(f"Extracted Ci for {len(muon_C_map)} modules (source: {ci_model_id})")

    # -------------------------
    # Build optimizer EXCLUDING Muon-targeted weights
    # (so you don't do AdamW + Muon on the same matrix)
    # -------------------------
    muon_weight_ids = set()
    if args.muon_ogd:
        for _, module in muon_targets.items():
            muon_weight_ids.add(id(module.weight))

    opt_params = []
    for p in model.parameters():
        if args.muon_ogd and (id(p) in muon_weight_ids):
            continue
        if p.requires_grad:
            opt_params.append(p)

    print(f"Optimizer params: {len(opt_params)} tensors (excluded {len(muon_weight_ids)} muon weight tensors)")
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    muon_opt = None
    muon_scheduler = None
    if args.muon_ogd and args.muon_use_optimizer_class and len(muon_targets) > 0:
        muon_params = [module.weight for module in muon_targets.values()]
        muon_opt = MuonOGDOptimizer(
            muon_params,
            lr=args.muon_eta,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
            muon_T=args.muon_T,
            muon_eta_dual=args.muon_eta_dual,
            msign_method=args.muon_msign_method,
            ns_iters=args.muon_ns_iters,
            dynamic_scale=args.muon_dynamic_scale,
            warm_start=args.muon_warm_start,
        )
        for name, module in muon_targets.items():
            muon_opt.state[module.weight]["Cs"] = muon_C_map.get(name, [])
            if args.muon_warm_start and name in muon_lambda_map:
                muon_opt.state[module.weight]["lam"] = muon_lambda_map[name].detach().clone()
        muon_scheduler = get_linear_schedule_with_warmup(
            muon_opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

    print(f"Starting training: Epochs={args.epochs}, Batch={args.batch_size}, GradAccum={args.grad_accum}")
    print(f"Total optimization steps: {max_train_steps}")
    if args.muon_ogd:
        print(f"Muon msgn method: {args.muon_msign_method} (ns_iters={args.muon_ns_iters})")

    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")
    global_step = 0
    total_loss = 0.0
    _step_time_accum = 0.0
    _step_time_count = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loader):
            batch_start_time = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum

            # Skip bad batches (NaN/Inf loss) — log and continue
            if not torch.isfinite(outputs.loss):
                print(f"[skip] Step {global_step} batch {step}: non-finite loss={outputs.loss.item():.4f}, skipping update.", flush=True)
                model.zero_grad(set_to_none=True)
                continue

            loss.backward()
            total_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                # Check for NaN/Inf in gradients before stepping
                has_bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_bad_grad = True
                        break
                if has_bad_grad:
                    print(f"[skip] Step {global_step}: NaN/Inf in gradients, skipping optimizer step.", flush=True)
                    model.zero_grad(set_to_none=True)
                    continue

                if args.max_grad_norm > 0:
                    # Clip all grads (including muon weights) before any updates
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # --- AdamW step for non-muon params ---
                opt.step()
                scheduler.step()

                # --- Muon update ---
                if muon_opt is not None:
                    muon_opt.step()
                    muon_scheduler.step()
                elif args.muon_ogd and len(muon_targets) > 0:
                    try:
                        for name, module in muon_targets.items():
                            if module.weight.grad is None:
                                continue
                            Cs = muon_C_map.get(name, [])
                            lam_init = muon_lambda_map.get(name, None) if args.muon_warm_start else None

                            Delta, lam_final = _muon_ogd_apply_on_weight_factorized(
                                module.weight,
                                Cs,
                                module.weight.grad.detach(),
                                eta=args.muon_eta,
                                eta_dual=args.muon_eta_dual,
                                T=args.muon_T,
                                lam_init=lam_init,
                                msign_method=args.muon_msign_method,
                                ns_iters=args.muon_ns_iters,
                                time_profile=args.time_profile,
                            )
                            if args.muon_warm_start and lam_final is not None:
                                muon_lambda_map[name] = lam_final.detach()
                            # Apply update (cast to weight dtype)
                            module.weight.data.add_(Delta.to(module.weight.data.dtype))
                    except Exception as e:
                        print(f"Muon-OGD failure at step {global_step}: {e}", flush=True)

                # --- Optional SVD projection (NOT part of Muon-OGD algorithm; disabled) ---
                # NOTE: SVD weight projection is a separate technique from a different paper.
                # It is NOT a step in the Muon-OGD algorithm. Re-enable only if intentionally
                # combining both methods and using a high rank (e.g. --svd_rank 512+).
                # if args.svd_project and (((global_step + 1) % max(1, args.svd_every)) == 0):
                #     filters = [s.strip() for s in args.svd_layers.split(",")] if args.svd_layers else []
                #     try:
                #         project_model_weights_svd(
                #             model,
                #             layer_filters=filters,
                #             rank=args.svd_rank,
                #             energy_thresh=args.svd_energy,
                #             time_profile=args.time_profile,
                #         )
                #     except Exception as e:
                #         print(f"SVD projection failed at step {global_step}: {e}", flush=True)

                # Clear all grads (important because muon weights are not in optimizer)
                model.zero_grad(set_to_none=True)

                global_step += 1
                if args.save_strategy == "steps" and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"\nSaving intermediate checkpoint at step {global_step} to {checkpoint_dir} ...", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                iter_dt = time.perf_counter() - batch_start_time
                _step_time_accum += iter_dt
                _step_time_count += 1

                if global_step % args.log_every == 0:
                    avg_loss = total_loss * args.grad_accum / args.log_every
                    lr = scheduler.get_last_lr()[0]
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"})
                    print(f"Step {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s")
                    total_loss = 0.0

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