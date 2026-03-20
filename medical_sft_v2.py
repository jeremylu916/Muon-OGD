import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_codealpaca_qwen0.5b_muon_ogd_algo2"
DEFAULT_DATASET_ID = "sahil2801/CodeAlpaca-20k"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 8
DEFAULT_LR = 1e-5
DEFAULT_SEED = 42
DEFAULT_NUM_TRAIN_EXAMPLES = 2000
DEFAULT_MAX_STEPS = 1000
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10
DEFAULT_PROBE_EVERY = 100
DEFAULT_PROBE_MAX_NEW_TOKENS = 96
DEFAULT_VAL_RATIO = 0.05
DEFAULT_VAL_EVERY = 100
DEFAULT_VAL_MAX_BATCHES = 16

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


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def parse_args():
    p = argparse.ArgumentParser(description="SFT with AdamW + Muon-OGD (Algorithm 2 Bilinear Subspace).")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--ci_model_ids", type=str, default="", help="Comma-separated list of reference models for Ci extraction")
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
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"])
    p.add_argument("--save_steps", type=int, default=500)

    p.add_argument("--muon_ogd", action="store_true")
    p.add_argument("--muon_use_optimizer_class", action="store_true", help="IGNORED: Algorithm 2 is fully inlined for performance.")
    p.add_argument("--muon_layers", type=str, default="o_proj,down_proj")
    p.add_argument("--muon_k", type=int, default=3)
    p.add_argument("--muon_T", type=int, default=1)
    p.add_argument("--muon_eta", type=float, default=1e-5)
    p.add_argument("--muon_eta_dual", type=float, default=1e-4)
    p.add_argument("--muon_msign_method", type=str, default="ns", choices=["svd", "ns"])
    p.add_argument("--muon_ns_iters", type=int, default=6)
    p.add_argument("--muon_warm_start", action="store_true")
    p.add_argument("--muon_dynamic_scale", action="store_true")
    
    p.add_argument("--ci_from_grads", action="store_true")
    p.add_argument("--ci_replay_examples", type=int, default=128)
    return p.parse_args()


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

def build_prompt_solution(user_prompt: str, solution_text: str):
    messages_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Write Python code to solve the task. Output ONLY valid Python code, no markdown, no explanation.\n\n{user_prompt.rstrip()}"},
    ]
    return messages_prompt, solution_text.rstrip() + "\n"

def collect_muon_targets(model: torch.nn.Module, layer_filters: List[str]) -> Dict[str, torch.nn.Module]:
    out = {}
    filters = [f.strip() for f in layer_filters if f.strip()]
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight")
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if filters and not any(f in name for f in filters):
            continue
        out[name] = module
    return out

def _load_replay_dataset(dataset_id: str, split: str, cache_dir=None, config: Optional[str] = None):
    if config:
        return load_dataset(dataset_id, config, split=split, cache_dir=cache_dir)
    return load_dataset(dataset_id, split=split, cache_dir=cache_dir)

def _build_supervised_single_batch(tokenizer, question_text: str, answer_text: str, max_length: int, device: torch.device, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    solution_ids = tokenizer(str(answer_text) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + solution_ids)[:max_length]
    labels = (([-100] * len(prompt_ids)) + solution_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }

def _accumulate_task_replay_gradients(model, tokenizer, muon_targets, device, cache_dir, seed, replay_examples, dataset_id, split, question_col, answer_col, system_prompt, dataset_config=None):
    print(f"  -> Replaying {dataset_id}...")
    ds = _load_replay_dataset(dataset_id=dataset_id, split=split, cache_dir=cache_dir, config=dataset_config)
    if replay_examples > 0 and replay_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(replay_examples))

    task_grad = {name: torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device) for name, mod in muon_targets.items()}
    used = 0
    model.train()
    
    for ex in ds:
        q = str(ex.get(question_col, "")).strip()
        a = str(ex.get(answer_col, "")).strip()
        if not q or not a: continue

        batch = _build_supervised_single_batch(tokenizer, q, a, 2048, device, system_prompt)
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss): continue

        loss.backward()
        with torch.no_grad():
            for name, mod in muon_targets.items():
                if mod.weight.grad is not None:
                    task_grad[name].add_(mod.weight.grad.detach().to(torch.float32))
        used += 1

    if used > 0:
        with torch.no_grad():
            for name in task_grad.keys():
                task_grad[name].div_(float(used))
                grad_norm = torch.linalg.norm(task_grad[name])
                if torch.isfinite(grad_norm) and grad_norm > 0:
                    task_grad[name].div_(grad_norm)
    return task_grad, used


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=cache_dir)
    if args.num_train_examples and args.num_train_examples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)

    def tokenize_fn(example):
        instruction = str(example.get("instruction", "")).strip() if "instruction" in example else str(example.get("prompt", ""))
        extra_input = str(example.get("input", "")).strip() if "input" in example else ""
        user_prompt = instruction if not extra_input else f"{instruction}\n\nInput:\n{extra_input}"
        solution = str(example.get("output", "")) if "output" in example else str(example.get("canonical_solution", ""))

        prompt_messages, solution = build_prompt_solution(user_prompt, solution)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer(solution + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + solution_ids)[:args.max_length]
        labels = (([-100] * len(prompt_ids)) + solution_ids)[:args.max_length]
        pad_len = args.max_length - len(input_ids)
        
        return {
            "input_ids": input_ids + [tokenizer.pad_token_id] * pad_len,
            "attention_mask": [1] * len(input_ids) + [0] * pad_len,
            "labels": labels + [-100] * pad_len,
        }

    print("Tokenizing CodeAlpaca...")
    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    if args.val_ratio > 0 and len(tokenized) > 1:
        shuffled = tokenized.shuffle(seed=args.seed)
        val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * args.val_ratio)))
        val_tokenized = shuffled.select(range(val_count))
        train_tokenized = shuffled.select(range(val_count, len(shuffled)))
    else:
        train_tokenized = tokenized
        val_tokenized = None

    use_cuda = torch.cuda.is_available()
    model_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache_dir, dtype=model_dtype)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    if not use_cuda: model = model.float()
    model.train()

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    loader = DataLoader(train_tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # =========================================================================
    # Algorithm 2 Subspace Extraction (U and V)
    # =========================================================================
    muon_targets = {}
    muon_C_map = {}
    muon_lambda_map = {} # Store dual matrix Lambda

    if args.muon_ogd:
        muon_targets = collect_muon_targets(model, args.muon_layers.split(","))
        
        if args.ci_from_grads:
            print(f"Accumulating replay gradients for Algorithm 2 Subspace (k={args.muon_k})...")
            g_acc = {name: torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device) for name, mod in muon_targets.items()}

            replay_specs = [
                {"dataset_id": "gsm8k", "split": "train", "dataset_config": "main", "question_col": "question", "answer_col": "answer", "system_prompt": "You are a helpful assistant."},
                {"dataset_id": "FreedomIntelligence/medical-o1-reasoning-SFT", "split": "train", "dataset_config": "en", "question_col": "Question", "answer_col": "Response", "system_prompt": "You are a careful medical reasoning assistant. Provide concise, clinically grounded answers."},
            ]

            used_tasks = 0
            for spec in replay_specs:
                try:
                    task_grad, used = _accumulate_task_replay_gradients(model, tokenizer, muon_targets, device, cache_dir, args.seed, args.ci_replay_examples, **spec)
                    if task_grad and used > 0:
                        with torch.no_grad():
                            for name in g_acc.keys(): g_acc[name].add_(task_grad[name])
                        used_tasks += 1
                except Exception as e:
                    print(f"  -> Replay source failed ({spec['dataset_id']}): {e}")

            if used_tasks == 0: raise RuntimeError("Failed to accumulate any replay gradients.")

            with torch.no_grad():
                for name in g_acc.keys():
                    g_acc[name].div_(float(used_tasks))
                    # Extract (U, V) matrices per Algorithm 2
                    U_k, V_k = extract_subspace_matrices(g_acc[name], args.muon_k, device)
                    muon_C_map[name] = (U_k, V_k)
            print(f"Algorithm 2: Extracted Orthogonal U/V constraint matrices for {len(muon_C_map)} modules.")
            model.zero_grad(set_to_none=True)

    # Exclude Muon targets from AdamW
    muon_weight_ids = set(id(m.weight) for m in muon_targets.values()) if args.muon_ogd else set()
    adam_params = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_weight_ids]
    adam_opt = torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=args.weight_decay)

    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(loader) // args.grad_accum)
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    adam_scheduler = get_linear_schedule_with_warmup(adam_opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    seen_steps = 0
    global_step = 0
    total_loss = 0.0
    pbar = tqdm(total=max_train_steps, desc="Training")

    # =========================================================================
    # Training Loop
    # =========================================================================
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            seen_steps += 1
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            
            if not torch.isfinite(loss): continue
            loss.backward()
            total_loss += outputs.loss.item()

            if seen_steps % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(adam_params, args.max_grad_norm)

                adam_opt.step()
                adam_scheduler.step()

                # =============================================================
                # Algorithm 2: Inline Muon-OGD Matrix Projection Step
                # =============================================================
                if args.muon_ogd:
                    with torch.no_grad():
                        for name, module in muon_targets.items():
                            if module.weight.grad is None: continue
                            
                            G_fp32 = module.weight.grad.to(torch.float32)
                            U, V = muon_C_map.get(name, (None, None))
                            
                            if U is not None and V is not None:
                                k = U.shape[1]
                                # Init dual matrix Lambda (k x k)
                                lam = muon_lambda_map.get(name, torch.zeros((k, k), dtype=torch.float32, device=device))
                                
                                # Inner dual loop
                                for _ in range(args.muon_T):
                                    H = G_fp32 + U @ lam @ V.transpose(0, 1)
                                    S = _msgn(H, iters=args.muon_ns_iters)
                                    grad_lam = U.transpose(0, 1) @ S @ V
                                    lam = lam - args.muon_eta_dual * grad_lam
                                
                                if args.muon_warm_start:
                                    muon_lambda_map[name] = lam
                                    
                                H = G_fp32 + U @ lam @ V.transpose(0, 1)
                            else:
                                H = G_fp32
                            
                            # Final Primal Step
                            S = _msgn(H, iters=args.muon_ns_iters)
                            
                            # Dynamic Scale (Gradient Norm Matching)
                            scale = 1.0
                            if args.muon_dynamic_scale:
                                scale = torch.linalg.norm(G_fp32).item() / max(1e-8, torch.linalg.norm(S).item())
                            
                            Delta = -args.muon_eta * scale * S
                            module.weight.data.add_(Delta.to(module.weight.dtype))
                # =============================================================

                model.zero_grad(set_to_none=True)
                global_step += 1
                
                if global_step % args.log_every == 0:
                    avg_loss = total_loss / (args.log_every * args.grad_accum)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    total_loss = 0.0
                    
                pbar.update(1)
                if global_step >= max_train_steps: break
        if global_step >= max_train_steps: break

    pbar.close()
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")

if __name__ == "__main__":
    main()