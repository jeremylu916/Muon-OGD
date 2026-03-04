"""
train_huatuo_sft_svd.py
SFT on HuatuoGPT-o1 style medical QA with Muon-OGD to mitigate catastrophic forgetting.

Muon-OGD algorithm (Spectral-Norm Constrained Projection via Dual Iterations):
  1. Precompute protected directions {Ci} from a pretrained/previous-task model (--ci_model_id).
  2. Each optimizer step: compute gradient G.
  3. Inner loop T times: form H = G + sum_i lambda_i * Ci -> compute polar factor S = msgn(H) -> update lambda.
  4. Primal update: theta <- theta + Delta, Delta = -eta * msgn(H_final).
  Muon-targeted weights are EXCLUDED from AdamW to avoid double updates.
"""

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

# ---- Defaults ----
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_huatuo_muon"
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
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOG_EVERY = 10


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    p = argparse.ArgumentParser(description="SFT on HuatuoGPT-o1 medical QA with Muon-OGD.")
    # Dataset / model
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--train_split", type=str, default=DEFAULT_TRAIN_SPLIT)
    p.add_argument("--question_field", type=str, default=DEFAULT_QUESTION_FIELD)
    p.add_argument("--answer_field", type=str, default=DEFAULT_ANSWER_FIELD)
    p.add_argument("--language_field", type=str, default=DEFAULT_LANGUAGE_FIELD)
    p.add_argument("--english_only", action="store_true", default=True)
    # Training
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_train_examples", type=int, default=DEFAULT_NUM_TRAIN_EXAMPLES)
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    # Checkpointing
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    # Muon-OGD
    p.add_argument("--muon_ogd", action="store_true", help="Enable Muon-OGD constrained updates")
    p.add_argument("--muon_T", type=int, default=1, help="Inner dual iterations T")
    p.add_argument("--muon_eta", type=float, default=1e-4, help="Primal step size eta")
    p.add_argument("--muon_eta_dual", type=float, default=1e-4, help="Dual step size eta_lambda")
    p.add_argument("--muon_k", type=int, default=1, help="Number of protected directions k per layer")
    p.add_argument("--muon_layers", type=str, default="", help="Comma substrings to match module names (default=all 2D weights)")
    p.add_argument("--muon_warm_start", action="store_true", help="Warm-start dual variables lambda across steps")
    p.add_argument("--muon_use_optimizer_class", action="store_true", help="Use MuonOGDOptimizer class instead of manual per-layer Muon loop")
    p.add_argument("--muon_momentum", type=float, default=0.95, help="Momentum EMA coefficient for Muon optimizer class")
    p.add_argument("--muon_dynamic_scale", action="store_true", help="Enable dynamic per-layer scaling in Muon optimizer class")
    p.add_argument("--muon_msign_method", type=str, default="ns", choices=["svd", "ns"], help="Polar factor: 'ns' (Newton-Schulz, fast) or 'svd' (exact)")
    p.add_argument("--muon_ns_iters", type=int, default=6, help="Newton-Schulz iterations")
    p.add_argument("--ci_model_id", type=str, default="", help="Model to extract protected Ci from (pretrained-task model). Defaults to --model_id.")
    p.add_argument("--ci_model_ids", type=str, default="", help="Comma-separated list of models to extract Ci from and accumulate (e.g., gsm8k,bigcodebench checkpoints)")
    p.add_argument("--ci_k_per_source", type=int, default=0, help="If >0, use this k per Ci source; otherwise use --muon_k per source")
    p.add_argument("--ci_from_grads", action="store_true", help="Build Ci from replay gradients instead of weight SVD")
    p.add_argument("--ci_replay_dataset", type=str, default="", help="HF dataset name for old-task replay; if empty and ci_from_grads, fallback is Huatuo dataset")
    p.add_argument("--ci_replay_config", type=str, default="", help="HF dataset config name for replay dataset (optional)")
    p.add_argument("--ci_replay_split", type=str, default="train", help="Replay split name")
    p.add_argument("--ci_replay_examples", type=int, default=128, help="Replay examples used to build gradient Ci")
    p.add_argument("--ci_replay_seed", type=int, default=123, help="Replay sampling seed")
    # Profiling
    p.add_argument("--time_profile", action="store_true", help="Print per-module Muon timing")
    return p.parse_args()


# ============================================================
# SVD / polar factor helpers
# ============================================================

def _compute_svd(X: torch.Tensor, k: int = None, method: str = "rand_gpu",
                 n_oversamples: int = 8, n_iter: int = 1, time_profile: bool = False):
    """Compute top-k SVD of X on its device. method: 'exact_gpu' or 'rand_gpu'."""
    t0 = time.perf_counter()
    dev = X.device
    Xf = X.detach().to(torch.float32).to(dev)

    if method == "exact_gpu" or k is None:
        U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
        if time_profile:
            print(f"[svd exact_gpu] shape={tuple(Xf.shape)} time={time.perf_counter()-t0:.3f}s")
        return U, S, Vh

    # Randomized SVD on device
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
    U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    if time_profile:
        print(f"[svd rand_gpu] shape={tuple(Xf.shape)} k={k} time={time.perf_counter()-t0:.3f}s")
    return U, S, Vh


def _matrix_sign_via_svd(X: torch.Tensor) -> torch.Tensor:
    """Polar factor via exact SVD: P = U @ Vh."""
    dev, orig_dtype = X.device, X.dtype
    Xf = X.detach().to(torch.float32).to(dev)
    U, _, Vh = torch.linalg.svd(Xf, full_matrices=False)
    return (U @ Vh).to(dev).to(orig_dtype)


@torch.no_grad()
def _polar_newton_schulz(X: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """Fast polar factor via Newton-Schulz iterations (matmul only)."""
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
            T = 0.5 * (3.0 * I - Z @ Y.transpose(0, 1) @ Y)
            Y = Y @ T
            Z = T @ Z
        P = Y
    else:
        At = A.transpose(0, 1)
        Y = At
        I = torch.eye(m, device=dev, dtype=torch.float32)
        Z = I.clone()
        for _ in range(iters):
            T = 0.5 * (3.0 * I - Z @ Y.transpose(0, 1) @ Y)
            Y = Y @ T
            Z = T @ Z
        P = Y.transpose(0, 1)
    return P.to(dev).to(orig_dtype)


def _msgn(X: torch.Tensor, method: str = "ns", ns_iters: int = 6) -> torch.Tensor:
    """Matrix sign / polar factor dispatcher."""
    if method == "svd":
        return _matrix_sign_via_svd(X)
    return _polar_newton_schulz(X, iters=ns_iters)


@dataclass
class Rank1C:
    sigma: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


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


def _muon_ogd_apply_on_weight_factorized(
    W: torch.Tensor,
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
    """Muon-OGD inner loop for a single 2D weight matrix.

    Returns (Delta, lam_final).
    Delta = -eta * msgn(H_final) to be added to W.
    """
    t0 = time.perf_counter()
    dev = W.device
    G_fp32 = G.detach().to(torch.float32).to(dev)
    k = len(Cs)

    # Initialize / warm-start lambda
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

    # Inner dual loop (T iterations)
    for _ in range(T):
        H = G_fp32.clone()
        _add_rank1_shift_(H, Cs, lam)
        S = _msgn(H, method=msign_method, ns_iters=ns_iters).to(torch.float32)
        inner = _rank1_inner_products(Cs, S)
        lam = lam - eta_dual * inner

    # Final primal update
    H_final = G_fp32.clone()
    _add_rank1_shift_(H_final, Cs, lam)
    S_final = _msgn(H_final, method=msign_method, ns_iters=ns_iters).to(torch.float32)
    Delta = (-eta) * S_final

    if time_profile:
        print(f"[muon] shape={tuple(W.shape)} k={k} T={T} time={time.perf_counter()-t0:.3f}s")
    return Delta, lam


# ============================================================
# Dataset helpers
# ============================================================

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
        {"role": "system", "content": "You are a careful medical reasoning assistant. Provide concise, clinically grounded answers."},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Dataset ----
    print(f"Loading dataset {args.dataset_id}...")
    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.train_split, cache_dir=cache_dir)

    if args.english_only:
        if args.language_field in ds.column_names:
            ds = ds.filter(lambda ex: str(ex.get(args.language_field, "")).lower().startswith("en"))
        else:
            ds = ds.filter(lambda ex: looks_english(to_text(ex.get(args.question_field, ""))))

    if args.num_train_examples and args.num_train_examples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_train_examples))
    print(f"Training on {len(ds)} examples.")

    # ---- Tokenizer ----
    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(example):
        q = to_text(example.get(args.question_field, ""))
        a = to_text(example.get(args.answer_field, ""))

        prompt_only = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a careful medical reasoning assistant. Provide concise, clinically grounded answers."},
             {"role": "user", "content": q.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            build_messages(q, a),
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tok = tokenizer(prompt_only, truncation=True, max_length=args.max_length, add_special_tokens=False)
        full_tok = tokenizer(full, truncation=True, max_length=args.max_length, padding="max_length", add_special_tokens=False)

        labels = full_tok["input_ids"].copy()
        prompt_len = min(len(prompt_tok["input_ids"]), args.max_length)
        for i in range(prompt_len):
            labels[i] = -100
        for i, m in enumerate(full_tok["attention_mask"]):
            if m == 0:
                labels[i] = -100
        full_tok["labels"] = labels
        return full_tok

    print("Tokenizing dataset...")
    tokenized = ds.map(tok, remove_columns=ds.column_names)

    # ---- Model ----
    print(f"Loading model {args.model_id}...")
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, dtype=dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
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
                print("ci_from_grads enabled but no ci_replay_dataset provided; using current Huatuo dataset as replay.")
                replay_ds = ds

            n_replay = min(args.ci_replay_examples, len(replay_ds))
            replay_ds = replay_ds.shuffle(seed=args.ci_replay_seed).select(range(n_replay))

            print("Tokenizing replay dataset for Ci-from-grads...")
            replay_tokenized = replay_ds.map(tok, remove_columns=replay_ds.column_names)

            def replay_collate(batch):
                return {
                    "input_ids": torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long),
                    "attention_mask": torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long),
                    "labels": torch.tensor([ex["labels"] for ex in batch], dtype=torch.long),
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
            ci_sources = [s.strip() for s in args.ci_model_ids.split(",") if s.strip()]
            if not ci_sources:
                fallback_source = args.ci_model_id.strip() if args.ci_model_id.strip() else args.model_id
                ci_sources = [fallback_source]

            k_per_source = args.ci_k_per_source if args.ci_k_per_source > 0 else args.muon_k
            print(f"Accumulating Ci from {len(ci_sources)} source model(s), k_per_source={k_per_source}...")

            for name in muon_targets.keys():
                muon_C_map[name] = []

            for src_id in ci_sources:
                if src_id == args.model_id:
                    print(f"Snapshotting weights for Ci from: {src_id} (same as model_id)")
                    ci_weight_map = {name: module.weight.data.detach().clone().cpu() for name, module in muon_targets.items()}
                else:
                    print(f"Loading reference model for Ci: {src_id} (CPU, float32)...")
                    ci_ref_model = AutoModelForCausalLM.from_pretrained(src_id, cache_dir=cache_dir, torch_dtype=torch.float32)
                    ci_ref_model.eval()
                    ci_weight_map: Dict[str, torch.Tensor] = {}
                    for ref_name, ref_module in ci_ref_model.named_modules():
                        if ref_name in muon_targets and hasattr(ref_module, "weight"):
                            ci_weight_map[ref_name] = ref_module.weight.data.detach().clone().cpu()
                    del ci_ref_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Reference model freed. Weight snapshots for {len(ci_weight_map)} modules.")

                print(f"Extracting protected directions Ci from source: {src_id} ...")
                for name, module in muon_targets.items():
                    src_weight = ci_weight_map.get(name, module.weight.data)
                    Cs = _extract_rank1_factors_from_matrix(src_weight, k=k_per_source, device=module.weight.device)
                    muon_C_map[name].extend(Cs)

            if args.muon_warm_start:
                for name, module in muon_targets.items():
                    muon_lambda_map[name] = torch.zeros(len(muon_C_map[name]), dtype=torch.float32, device=module.weight.device)

            print(f"Extracted accumulated Ci for {len(muon_C_map)} modules from sources: {ci_sources}")

    # ---- Optimizer (exclude Muon-targeted weights from AdamW) ----
    muon_weight_ids = {id(m.weight) for m in muon_targets.values()} if args.muon_ogd else set()
    opt_params = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_weight_ids]
    print(f"AdamW params: {len(opt_params)} tensors (excluded {len(muon_weight_ids)} Muon weight tensors)")
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Scheduler ----
    def collate(batch):
        return {
            "input_ids": torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long),
            "attention_mask": torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long),
            "labels": torch.tensor([ex["labels"] for ex in batch], dtype=torch.long),
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
    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")
    if args.muon_ogd:
        print(f"Muon-OGD: k={args.muon_k}, T={args.muon_T}, eta={args.muon_eta}, eta_dual={args.muon_eta_dual}, method={args.muon_msign_method}")

    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")
    global_step = 0
    total_loss = 0.0
    _step_time_accum = 0.0
    _step_time_count = 0

    # ---- Training loop ----
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(loader):
            batch_start_time = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum

            # Skip NaN/Inf loss batches
            if not torch.isfinite(outputs.loss):
                print(f"[skip] Step {global_step} batch {step}: non-finite loss={outputs.loss.item():.4f}, skipping.", flush=True)
                model.zero_grad(set_to_none=True)
                continue

            loss.backward()
            total_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                # Check for NaN/Inf gradients
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if has_bad_grad:
                    print(f"[skip] Step {global_step}: NaN/Inf in gradients, skipping optimizer step.", flush=True)
                    model.zero_grad(set_to_none=True)
                    continue

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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

                # ---- Save Intermediate Checkpoints for Forgetting Curve ----
                if args.save_strategy == "steps" and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"\nSaving intermediate checkpoint at step {global_step} to {checkpoint_dir} ...", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                # Timing
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

    try:
        pbar.close()
    except Exception:
        pass

    print(f"Saving final model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()