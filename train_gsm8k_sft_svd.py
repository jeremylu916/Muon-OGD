import argparse
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
DEFAULT_OUTPUT_DIR = "outputs/sft_gsm8k_muon_ogd"
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


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def parse_args():
    p = argparse.ArgumentParser(description="SFT on GSM8K with Hybrid AdamW + Muon-OGD (fast factorized Ci).")
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
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")

    # Performance knobs
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--pin_memory", action="store_true", help="Use pin_memory for DataLoader (CUDA)")
    p.add_argument("--use_autocast", action="store_true", help="Use torch.autocast on CUDA for forward/backward")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on Ampere+ (CUDA)")

    # Muon-OGD options
    p.add_argument("--muon_ogd", action="store_true", help="Enable Muon-OGD spectral-norm constrained update")
    p.add_argument("--muon_T", type=int, default=1, help="Number of inner dual iterations T")
    p.add_argument("--muon_eta", type=float, default=1e-4, help="Primal step size eta")
    p.add_argument("--muon_eta_dual", type=float, default=1e-4, help="Dual step size eta_lambda")
    p.add_argument("--muon_k", type=int, default=1, help="Number of protected directions k per layer")
    p.add_argument("--muon_layers", type=str, default="", help="Comma-separated substrings to match module names (default=all 2D linear weights)")
    p.add_argument("--muon_msign_method", type=str, default="ns", choices=["svd", "ns"], help="Polar factor method: 'ns' (Newton-Schulz fast) or 'svd'")
    p.add_argument("--muon_ns_iters", type=int, default=6, help="Newton-Schulz iterations")
    p.add_argument("--muon_warm_start", action="store_true", help="Warm-start dual variables lambda across steps")
    p.add_argument("--muon_use_optimizer_class", action="store_true", help="Use MuonOGDOptimizer class instead of manual per-layer Muon loop")
    p.add_argument("--muon_momentum", type=float, default=0.0, help="Momentum EMA coefficient for Muon optimizer class")
    p.add_argument("--muon_dynamic_scale", action="store_true", help="Enable dynamic per-layer scaling in Muon optimizer class")
    p.add_argument("--ci_model_id", type=str, default="", help="Model to extract protected Ci directions from. Defaults to --model_id if not set.")
    p.add_argument("--time_profile", action="store_true", help="Print per-step timing for Muon-OGD")

    # Less-forgetting knobs (optional)
    p.add_argument("--ci_from_grads", action="store_true", help="Build Ci from reference (old-task) gradients instead of weight SVD")
    p.add_argument("--ci_replay_dataset", type=str, default="", help="HF dataset name for old task replay (e.g., 'bigcodebench'...). If empty and ci_from_grads, falls back to GSM8K.")
    p.add_argument("--ci_replay_config", type=str, default="", help="HF dataset config name for replay dataset (optional)")
    p.add_argument("--ci_replay_split", type=str, default="train", help="Split name for replay dataset")
    p.add_argument("--ci_replay_examples", type=int, default=128, help="Number of replay examples for gradient-based Ci")
    p.add_argument("--ci_replay_seed", type=int, default=123, help="Seed for replay sampling")

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
# Muon helpers (factorized Ci)
# =========================

@dataclass
class Rank1C:
    sigma: torch.Tensor  # scalar fp32 on device
    u: torch.Tensor      # (m,) fp32 on device
    v: torch.Tensor      # (n,) fp32 on device


def _matrix_sign_via_svd(X: torch.Tensor) -> torch.Tensor:
    orig_device = X.device
    orig_dtype = X.dtype
    X_fp32 = X.detach().to(torch.float32).to(orig_device)
    U, _, Vh = torch.linalg.svd(X_fp32, full_matrices=False)
    P = U @ Vh
    return P.to(orig_device).to(orig_dtype)


@torch.no_grad()
def _polar_newton_schulz(X: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    dev = X.device
    orig_dtype = X.dtype
    A = X.detach().to(torch.float32).to(dev)
    m, n = A.shape
    frob = torch.linalg.norm(A, ord="fro")
    if frob < eps:
        return torch.zeros_like(X)
    A = A / (frob + eps)

    if m >= n:
        Y = A
        I = torch.eye(n, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            ZY = Z @ Y.transpose(0, 1) @ Y
            T = 0.5 * (3.0 * I - ZY)
            Y = Y @ T
            Z = T @ Z
        P = Y
    else:
        At = A.transpose(0, 1)
        Y = At
        I = torch.eye(m, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            ZY = Z @ Y.transpose(0, 1) @ Y
            T = 0.5 * (3.0 * I - ZY)
            Y = Y @ T
            Z = T @ Z
        P = Y.transpose(0, 1)

    return P.to(dev).to(orig_dtype)


def _msgn(X: torch.Tensor, method: str = "ns", ns_iters: int = 6) -> torch.Tensor:
    if method == "svd":
        return _matrix_sign_via_svd(X)
    return _polar_newton_schulz(X, iters=ns_iters)


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
    # <sigma u v^T, S> = sigma * u^T S v
    if not Cs:
        return torch.zeros(0, device=S_mat_fp32.device, dtype=torch.float32)
    vals = []
    for c in Cs:
        vals.append(c.sigma * (c.u @ (S_mat_fp32 @ c.v)))
    return torch.stack(vals, dim=0)


def _add_rank1_shift_(H_fp32: torch.Tensor, Cs: List[Rank1C], lam_fp32: torch.Tensor) -> torch.Tensor:
    # H += sum_i lam_i * (sigma_i u_i v_i^T)
    # This is O(k*m*n) but avoids allocating (k,m,n).
    for i, c in enumerate(Cs):
        alpha = lam_fp32[i] * c.sigma
        # outer product: (m,1)@(1,n)
        H_fp32.add_(alpha * (c.u.unsqueeze(1) @ c.v.unsqueeze(0)))
    return H_fp32


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
        inner = _rank1_inner_products(Cs, S)  # (k,)
        lam = lam - eta_dual * inner

    H_final = G_fp32.clone()
    _add_rank1_shift_(H_final, Cs, lam)
    S_final = _msgn(H_final, method=msign_method, ns_iters=ns_iters).to(torch.float32)
    Delta = (-eta) * S_final

    if time_profile:
        print(f"[muon] shape={tuple(W.shape)} k={k} T={T} time={time.perf_counter()-t0:.3f}s")
    return Delta, lam


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
    print(f"Training examples: {len(dataset)}")

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

    # ---- Model ----
    print(f"Loading model {args.model_id}...")
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (
        torch.float16 if use_cuda else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, torch_dtype=dtype)
    model.to(device)
    model.train()

    # ---- Muon-OGD setup ----
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

        # --- Build Ci either from reference weights (fast) or from replay gradients (better for forgetting) ---
        if args.ci_from_grads:
            # Build a tiny replay set
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
                # Fallback: use GSM8K itself as replay
                print("ci_from_grads enabled but no ci_replay_dataset provided; using GSM8K train as replay.")
                replay_ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)

            n_replay = min(args.ci_replay_examples, len(replay_ds))
            replay_ds = replay_ds.shuffle(seed=args.ci_replay_seed).select(range(n_replay))

            # Tokenize replay with the SAME tok (assumes replay has question/answer if GSM8K; otherwise user must adapt)
            # If your old task isn't GSM8K-like, you need to adapt this part.
            def replay_tok(example):
                # Expect GSM8K-style keys; if not present, raise a clear error
                if "question" not in example or "answer" not in example:
                    raise ValueError(
                        "Replay dataset examples must have 'question' and 'answer' fields "
                        "(GSM8K format). Adapt replay_tok() for your dataset."
                    )
                return tok(example)

            print("Tokenizing replay dataset for Ci-from-grads...")
            replay_tokenized = replay_ds.map(replay_tok, remove_columns=replay_ds.column_names)

            def collate(batch):
                return {
                    "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
                    "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
                    "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
                }

            replay_loader = DataLoader(
                replay_tokenized,
                batch_size=1,
                shuffle=False,
                collate_fn=collate,
                num_workers=0,
            )

            print(f"Accumulating old-task gradients over {n_replay} replay examples...")
            # Accumulate grads per muon module
            G_acc: Dict[str, torch.Tensor] = {}
            for name, mod in muon_targets.items():
                G_acc[name] = torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device)

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

            print(f"Extracting protected directions Ci from accumulated replay gradients (k={args.muon_k})...")
            for name, mod in muon_targets.items():
                muon_C_map[name] = _extract_rank1_factors_from_matrix(G_acc[name], k=args.muon_k, device=mod.weight.device)
                if args.muon_warm_start:
                    muon_lambda_map[name] = torch.zeros(len(muon_C_map[name]), dtype=torch.float32, device=mod.weight.device)

            print("Ci-from-grads ready.")
        else:
            # Weight-based Ci (your original behavior, but stored factorized for speed)
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
                if use_cuda:
                    torch.cuda.empty_cache()
                print(f"Reference model freed. Extracted weight snapshots for {len(ci_weight_map)} modules.")
            else:
                print("Snapshotting current weights for Ci (ci_model_id == model_id).")
                ci_weight_map = {name: mod.weight.data.detach().clone().cpu() for name, mod in muon_targets.items()}

            print(f"Extracting protected directions Ci from reference weights (factorized SVD, k={args.muon_k})...")
            for name, mod in muon_targets.items():
                src_weight = ci_weight_map.get(name, mod.weight.data)
                muon_C_map[name] = _extract_rank1_factors_from_matrix(src_weight, k=args.muon_k, device=mod.weight.device)
                if args.muon_warm_start:
                    muon_lambda_map[name] = torch.zeros(len(muon_C_map[name]), dtype=torch.float32, device=mod.weight.device)

            print(f"Extracted Ci for {len(muon_C_map)} modules (source: {ci_model_id})")

        print(f"Muon-OGD: k={args.muon_k}, T={args.muon_T}, eta={args.muon_eta}, eta_dual={args.muon_eta_dual}, method={args.muon_msign_method}")

    # ---- Optimizer (exclude Muon weight tensors from AdamW) ----
    muon_weight_ids = set()
    if args.muon_ogd:
        for _, module in muon_targets.items():
            muon_weight_ids.add(id(module.weight))

    opt_params = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_weight_ids]
    print(f"AdamW params: {len(opt_params)} tensors (excluded {len(muon_weight_ids)} muon weight tensors)")
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=max(0, args.num_workers),
        pin_memory=bool(args.pin_memory and use_cuda),
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

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

    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")
    global_step = 0
    total_loss = 0.0
    _step_time_accum = 0.0
    _step_time_count = 0

    # autocast settings
    use_autocast = bool(args.use_autocast and use_cuda)
    autocast_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
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
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if has_bad_grad:
                    print(f"[skip] Step {global_step}: NaN/Inf in gradients, skipping.", flush=True)
                    model.zero_grad(set_to_none=True)
                    continue

                # Clip ONLY AdamW params (Muon uses sign/polar, magnitude clipping is less meaningful there)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)

                # AdamW step (non-Muon params)
                opt.step()
                scheduler.step()

                # Muon-OGD step
                if muon_opt is not None:
                    muon_opt.step()
                    muon_scheduler.step()
                elif args.muon_ogd and muon_targets:
                    try:
                        for name, module in muon_targets.items():
                            if module.weight.grad is None:
                                continue

                            # Weight decay for Muon weights (since excluded from AdamW)
                            if args.weight_decay > 0:
                                module.weight.data.mul_(1.0 - args.muon_eta * args.weight_decay)

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
                            module.weight.data.add_(Delta.to(module.weight.data.dtype))
                    except Exception as e:
                        print(f"Muon-OGD failure at step {global_step}: {e}", flush=True)

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

    pbar.close()
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()