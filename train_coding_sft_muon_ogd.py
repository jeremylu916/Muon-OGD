import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from muon_ogd_optimizer import MuonOGDOptimizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_mbpp_qwen0.5b_muon_ogd"
DEFAULT_DATASET_ID = "mbpp"
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


@dataclass
class Rank1C:
    sigma: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


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
    p = argparse.ArgumentParser(description="SFT on MBPP coding data with AdamW + Muon-OGD.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--ci_model_id", type=str, default="", help="Reference model for Ci extraction. Defaults to --model_id")
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
    p.add_argument("--probe_every", type=int, default=DEFAULT_PROBE_EVERY, help="Run quick inference probe every N optimizer steps; <=0 disables")
    p.add_argument("--probe_max_new_tokens", type=int, default=DEFAULT_PROBE_MAX_NEW_TOKENS, help="Max new tokens for each probe inference")
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--val_every", type=int, default=DEFAULT_VAL_EVERY)
    p.add_argument("--val_max_batches", type=int, default=DEFAULT_VAL_MAX_BATCHES)
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")

    p.add_argument("--muon_ogd", action="store_true")
    p.add_argument("--muon_use_optimizer_class", action="store_true")
    p.add_argument("--muon_layers", type=str, default="o_proj,down_proj", help="Comma-separated substrings for 2D weight module names")
    p.add_argument("--muon_k", type=int, default=3)
    p.add_argument("--muon_T", type=int, default=1)
    p.add_argument("--muon_eta", type=float, default=1e-4)
    p.add_argument("--muon_eta_dual", type=float, default=1e-4)
    p.add_argument("--muon_msign_method", type=str, default="ns", choices=["svd", "ns"])
    p.add_argument("--muon_ns_iters", type=int, default=6)
    p.add_argument("--muon_warm_start", action="store_true")
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_dynamic_scale", action="store_true")
    p.add_argument("--ci_from_grads", action="store_true", help="Build Ci from replay gradients (math+medical) instead of static weight SVD")
    p.add_argument("--ci_replay_examples", type=int, default=128, help="Replay examples per prior task when --ci_from_grads is enabled")
    return p.parse_args()


def build_prompt_solution(user_prompt: str, solution_text: str):
    messages_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Write Python code to solve the task. "
                "Output ONLY valid Python code, no markdown, no explanation.\n\n"
                f"{user_prompt.rstrip()}"
            ),
        },
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


def extract_rank1_factors_from_matrix(mat: torch.Tensor, k: int, device: torch.device) -> List[Rank1C]:
    if mat.ndim != 2:
        return []
    m = mat.detach().to(torch.float32).to(device)
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    top = min(k, S.shape[0])
    factors = []
    for i in range(top):
        factors.append(Rank1C(sigma=S[i], u=U[:, i].contiguous(), v=Vh[i, :].contiguous()))
    return factors


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


def _accumulate_task_replay_gradients(
    model,
    tokenizer,
    muon_targets: Dict[str, torch.nn.Module],
    device: torch.device,
    cache_dir,
    seed: int,
    replay_examples: int,
    dataset_id: str,
    split: str,
    question_col: str,
    answer_col: str,
    system_prompt: str,
    dataset_config: Optional[str] = None,
):
    print(f"  -> Replaying {dataset_id} ({split}){f' config={dataset_config}' if dataset_config else ''}...")
    ds = _load_replay_dataset(dataset_id=dataset_id, split=split, cache_dir=cache_dir, config=dataset_config)
    if replay_examples > 0 and replay_examples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(replay_examples))

    task_grad = {
        name: torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device)
        for name, mod in muon_targets.items()
    }

    used = 0
    model.train()
    for ex in ds:
        q = str(ex.get(question_col, "")).strip()
        a = str(ex.get(answer_col, "")).strip()
        if not q or not a:
            continue

        batch = _build_supervised_single_batch(
            tokenizer=tokenizer,
            question_text=q,
            answer_text=a,
            max_length=2048,
            device=device,
            system_prompt=system_prompt,
        )

        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            continue

        loss.backward()
        with torch.no_grad():
            for name, mod in muon_targets.items():
                if mod.weight.grad is not None:
                    task_grad[name].add_(mod.weight.grad.detach().to(torch.float32))
        used += 1

    if used == 0:
        return None, 0

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
        if "instruction" in example and "output" in example:
            instruction = str(example.get("instruction", "")).strip()
            extra_input = str(example.get("input", "")).strip()
            user_prompt = instruction if not extra_input else f"{instruction}\n\nInput:\n{extra_input}"
            solution = str(example.get("output", ""))
        elif "text" in example and "code" in example:
            user_prompt = str(example.get("text", "")).strip()
            solution = str(example.get("code", ""))
        else:
            user_prompt = str(example.get("prompt", ""))
            solution = str(example.get("canonical_solution", ""))

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

    use_cuda = torch.cuda.is_available()
    model_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)

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
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            pred_ids = out[0, enc["input_ids"].shape[1]:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            print(f"[probe] step={step_idx} | {probe_id} | q={probe_q[:120]!r} | ****pred={pred[:180]!r} | ****target={probe_t[:180]!r}", flush=True)
        model.train()

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

    muon_targets = {}
    if args.muon_ogd:
        layer_filters = [x.strip() for x in args.muon_layers.split(",")] if args.muon_layers else []
        print("Collecting Muon target modules...")
        muon_targets = collect_muon_targets(model, layer_filters)
        print(f"Muon targets: {len(muon_targets)} modules")

    muon_C_map = {}
    if args.muon_ogd and muon_targets:
        if args.ci_from_grads:
            print(f"Accumulating replay gradients for Ci (examples per task={args.ci_replay_examples})...")
            g_acc = {
                name: torch.zeros_like(mod.weight.data, dtype=torch.float32, device=device)
                for name, mod in muon_targets.items()
            }

            replay_specs = [
                {
                    "dataset_id": "gsm8k",
                    "split": "train",
                    "dataset_config": "main",
                    "question_col": "question",
                    "answer_col": "answer",
                    "system_prompt": "You are a helpful assistant.",
                },
                {
                    "dataset_id": "FreedomIntelligence/medical-o1-reasoning-SFT",
                    "split": "train",
                    "dataset_config": "en",
                    "question_col": "Question",
                    "answer_col": "Response",
                    "system_prompt": "You are a careful medical reasoning assistant. Provide concise, clinically grounded answers.",
                },
            ]

            used_tasks = 0
            for spec in replay_specs:
                try:
                    task_grad, used = _accumulate_task_replay_gradients(
                        model=model,
                        tokenizer=tokenizer,
                        muon_targets=muon_targets,
                        device=device,
                        cache_dir=cache_dir,
                        seed=args.seed,
                        replay_examples=args.ci_replay_examples,
                        dataset_id=spec["dataset_id"],
                        split=spec["split"],
                        question_col=spec["question_col"],
                        answer_col=spec["answer_col"],
                        system_prompt=spec["system_prompt"],
                        dataset_config=spec["dataset_config"],
                    )
                except Exception as e:
                    print(f"  -> Replay source failed ({spec['dataset_id']}): {e}")
                    task_grad, used = None, 0

                if task_grad is None or used == 0:
                    continue

                with torch.no_grad():
                    for name in g_acc.keys():
                        g_acc[name].add_(task_grad[name])
                used_tasks += 1

            if used_tasks == 0:
                raise RuntimeError("ci_from_grads enabled, but no replay gradients were accumulated.")

            with torch.no_grad():
                for name in g_acc.keys():
                    g_acc[name].div_(float(used_tasks))

            model.zero_grad(set_to_none=True)
            for name in muon_targets.keys():
                muon_C_map[name] = extract_rank1_factors_from_matrix(g_acc[name], args.muon_k, device)

            print(f"Extracted replay-gradient Ci for {len(muon_C_map)} modules from {used_tasks} task source(s)")
        else:
            ci_model_id = args.ci_model_id.strip() if args.ci_model_id.strip() else args.model_id
            if ci_model_id != args.model_id:
                print(f"Loading reference model for Ci extraction: {ci_model_id} (CPU, float32)...")
                ci_ref_model = AutoModelForCausalLM.from_pretrained(ci_model_id, cache_dir=cache_dir, dtype=torch.float32)
                ci_ref_model.to("cpu")
                ref_state = {name: module.weight.detach().to(torch.float32).cpu() for name, module in ci_ref_model.named_modules() if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2}
                del ci_ref_model
            else:
                print("Snapshotting current weights for Ci (ci_model_id == model_id).")
                ref_state = {name: module.weight.detach().to(torch.float32).cpu() for name, module in model.named_modules() if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.ndim == 2}

            for name in muon_targets.keys():
                if name not in ref_state:
                    continue
                muon_C_map[name] = extract_rank1_factors_from_matrix(ref_state[name], args.muon_k, device)

            print(f"Extracted Ci for {len(muon_C_map)} modules (source: {ci_model_id})")

    muon_weight_ids = set(id(module.weight) for module in muon_targets.values()) if args.muon_ogd else set()
    adam_params = [p for p in model.parameters() if p.requires_grad and id(p) not in muon_weight_ids]
    print(f"AdamW params: {len(adam_params)} tensors (excluded {len(muon_weight_ids)} Muon weight tensors)")
    adam_opt = torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    adam_scheduler = get_linear_schedule_with_warmup(adam_opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)

    muon_opt = None
    muon_scheduler = None
    if args.muon_ogd and args.muon_use_optimizer_class and muon_targets:
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
        muon_scheduler = get_linear_schedule_with_warmup(
            muon_opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    seen_steps = 0
    global_step = 0
    total_loss = 0.0
    _step_time_accum = 0.0
    _step_time_count = 0
    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for batch in loader:
            batch_start = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}
            seen_steps += 1

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            if not torch.isfinite(outputs.loss):
                print(f"[skip] step={global_step}: non-finite loss={outputs.loss.item():.4f}", flush=True)
                model.zero_grad(set_to_none=True)
                continue

            loss.backward()
            total_loss += outputs.loss.item()

            if seen_steps % args.grad_accum == 0:
                has_bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
                if has_bad_grad:
                    print(f"[skip] step={global_step}: NaN/Inf gradients", flush=True)
                    model.zero_grad(set_to_none=True)
                    continue

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(adam_params, args.max_grad_norm)

                adam_opt.step()
                adam_scheduler.step()

                if muon_opt is not None:
                    muon_opt.step()
                    muon_scheduler.step()

                model.zero_grad(set_to_none=True)
                global_step += 1

                if args.save_strategy == "steps" and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"Saving checkpoint at step {global_step} to {checkpoint_dir}", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                iter_dt = time.perf_counter() - batch_start
                _step_time_accum += iter_dt
                _step_time_count += 1

                if global_step % args.log_every == 0:
                    avg_loss = total_loss / (args.log_every * args.grad_accum)
                    lr = adam_scheduler.get_last_lr()[0]
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"})
                    val_msg = ""
                    if args.val_every > 0 and global_step % args.val_every == 0:
                        val_loss = compute_val_loss()
                        if val_loss is not None:
                            val_msg = f" | ValLoss: {val_loss:.4f}"
                    print(f"Step {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s{val_msg}")
                    total_loss = 0.0

                if args.probe_every > 0 and global_step % args.probe_every == 0:
                    run_probe(global_step)

                pbar.update(1)

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

    pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
