import argparse
import json
import os
# Default-disable torch's cuDNN SDPA backend; on this GH200 stack it raises
# "cuDNN Frontend error: No valid execution plans built".
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")
import random
import re
import time
from typing import Dict, List

import torch
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from chat_template_utils import apply_chat_template_safe
from tqdm.auto import tqdm
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_huatuo"
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
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_PROBE_EVERY = 100
DEFAULT_PROBE_MAX_NEW_TOKENS = 64
DEFAULT_VAL_RATIO = 0.02
DEFAULT_VAL_EVERY = 100
DEFAULT_VAL_MAX_BATCHES = 32
DEFAULT_USE_OLORA = False
DEFAULT_OLORA_R = 16
DEFAULT_OLORA_ALPHA = 32
DEFAULT_OLORA_DROPOUT = 0.05
DEFAULT_OLORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_USE_SCULPT_SUBSPACE = False
DEFAULT_SUBSPACE_TOP_FRACTION = 0.5
DEFAULT_SUBSPACE_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"

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
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    p = argparse.ArgumentParser(description="SFT on HuatuoGPT-o1 style medical QA.")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--base_model_id", type=str, default="", help="Base model id/path when --model_id points to a PEFT adapter checkpoint")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--train_split", type=str, default=DEFAULT_TRAIN_SPLIT)
    p.add_argument("--question_field", type=str, default=DEFAULT_QUESTION_FIELD)
    p.add_argument("--answer_field", type=str, default=DEFAULT_ANSWER_FIELD)
    p.add_argument("--language_field", type=str, default=DEFAULT_LANGUAGE_FIELD)
    p.add_argument("--english_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require_verifiable", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_train_examples", type=int, default=DEFAULT_NUM_TRAIN_EXAMPLES)
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--probe_every", type=int, default=DEFAULT_PROBE_EVERY, help="Run quick inference probe every N optimizer steps; <=0 disables")
    p.add_argument("--probe_max_new_tokens", type=int, default=DEFAULT_PROBE_MAX_NEW_TOKENS, help="Max new tokens for each probe inference")
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio from tokenized train set")
    p.add_argument("--val_every", type=int, default=DEFAULT_VAL_EVERY, help="Run validation loss every N optimizer steps; <=0 disables")
    p.add_argument("--val_max_batches", type=int, default=DEFAULT_VAL_MAX_BATCHES, help="Max validation batches per validation run; <=0 uses full validation set")
    p.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")
    p.add_argument(
        "--answer_after_cot_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, train only on the final answer span after CoT/reasoning markers.",
    )
    p.add_argument("--use_olora", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_OLORA)
    p.add_argument("--olora_r", type=int, default=DEFAULT_OLORA_R)
    p.add_argument("--olora_alpha", type=int, default=DEFAULT_OLORA_ALPHA)
    p.add_argument("--olora_dropout", type=float, default=DEFAULT_OLORA_DROPOUT)
    p.add_argument("--olora_target_modules", type=str, default=DEFAULT_OLORA_TARGET_MODULES)
    p.add_argument("--use_sculpt_subspace", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_SCULPT_SUBSPACE)
    p.add_argument("--subspace_top_fraction", type=float, default=DEFAULT_SUBSPACE_TOP_FRACTION)
    p.add_argument("--subspace_target_modules", type=str, default=DEFAULT_SUBSPACE_TARGET_MODULES)
    return p.parse_args()


@torch.no_grad()
def build_high_subspace_bases(model, target_modules, top_fraction):
    subspaces = {}
    frac = max(0.0, min(1.0, top_fraction))
    for name, p in model.named_parameters():
        if p.ndim != 2 or not name.endswith("weight"):
            continue
        if target_modules and not any(tok in name for tok in target_modules):
            continue
        rank = min(p.shape)
        if rank <= 0:
            continue
        top_k = max(1, int(rank * frac))
        if rank > 1:
            top_k = min(top_k, rank - 1)
        w = p.detach().to(torch.float32)
        U, _, Vh = torch.linalg.svd(w, full_matrices=False)
        subspaces[name] = (U[:, :top_k].contiguous(), Vh[:top_k, :].contiguous())
    return subspaces


@torch.no_grad()
def project_grads_to_low_subspace(model, subspaces):
    for name, p in model.named_parameters():
        if p.grad is None or name not in subspaces:
            continue
        U_high, Vh_high = subspaces[name]
        g = p.grad.detach().to(torch.float32)
        g = g - U_high @ (U_high.transpose(0, 1) @ g)
        g = g - (g @ Vh_high.transpose(0, 1)) @ Vh_high
        p.grad.copy_(g.to(p.grad.dtype))


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
        {
            "role": "system",
            "content": "You are a careful medical assistant. Answer the medical question directly and concisely. Give the final answer first. Do not include unnecessary explanation.",
        },
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]

def format_target_answer(answer: str, final_only: bool = True) -> str:
    a = extract_final_answer_only(answer) if final_only else answer
    a = (a or "").strip()
    if not a:
        a = "Unknown."
    return a



def extract_final_answer_only(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    # Common CoT format: <think> ... </think> final answer
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail

    # Common final-answer delimiters used in reasoning traces.
    for pat in [
        r"(?is)final\s*answer\s*[:：]\s*(.+)$",
        r"(?is)answer\s*[:：]\s*(.+)$",
        r"(?is)####\s*(.+)$",
    ]:
        m = re.search(pat, text)
        if m:
            tail = m.group(1).strip()
            if tail:
                return tail

    return text

def is_trueish(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).lower() in {"true", "yes", "1", "verifiable"}


def resolve_column_name(ds, preferred_name: str, candidates: List[str], role: str) -> str:
    if preferred_name in ds.column_names:
        return preferred_name
    for name in candidates:
        if name in ds.column_names:
            print(
                f"[info] Requested {role}_field='{preferred_name}' not found; using '{name}' instead.",
                flush=True,
            )
            return name
    available = ", ".join(ds.column_names)
    raise ValueError(
        f"Could not resolve {role} field. Requested '{preferred_name}'. Available columns: {available}"
    )


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset_config:
        try:
            ds = load_dataset(args.dataset_id, args.dataset_config, split=args.train_split, cache_dir=cache_dir)
        except Exception as e:
            print(
                f"[warn] Failed to load dataset with config '{args.dataset_config}': {e}. Falling back to no config.",
                flush=True,
            )
            ds = load_dataset(args.dataset_id, split=args.train_split, cache_dir=cache_dir)
    else:
        ds = load_dataset(args.dataset_id, split=args.train_split, cache_dir=cache_dir)
    args.question_field = resolve_column_name(
        ds,
        args.question_field,
        ["Question", "question", "prompt", "instruction", "Open-ended Verifiable Question"],
        role="question",
    )
    args.answer_field = resolve_column_name(
        ds,
        args.answer_field,
        ["Response", "response", "answer", "output", "Ground-True Answer"],
        role="answer",
    )
    if args.answer_field == "Complex_CoT" and "Response" in ds.column_names:
        print("[info] answer_field resolved to 'Complex_CoT'. Switching to 'Response' to train without CoT.", flush=True)
        args.answer_field = "Response"
    print("*****************" + str(ds.column_names) + "*****************", flush=True)
    if args.english_only:
        ds_before_filter = ds
        if args.language_field in ds.column_names:
            ds = ds.filter(lambda ex: str(ex.get(args.language_field, "")).lower().startswith("en"))
        elif args.question_field in ds.column_names:
            ds = ds.filter(lambda ex: looks_english(to_text(ex.get(args.question_field, ""))))
        else:
            print(
                f"[warn] english_only enabled but neither '{args.language_field}' nor '{args.question_field}' exists. Skipping language filter.",
                flush=True,
            )

        if len(ds) == 0:
            print("[warn] English filtering removed all examples. Reverting to unfiltered dataset.", flush=True)
            ds = ds_before_filter
    


    if args.require_verifiable:
        if "verifiable" in ds.column_names:
            ds = ds.filter(lambda ex: is_trueish(ex.get("verifiable", False)))
        else:
            print("[info] require_verifiable enabled but 'verifiable' field not found. Skipping for this dataset.", flush=True)

    ds = ds.filter(
        lambda ex: to_text(ex.get(args.question_field, "")).strip() != ""
        and to_text(ex.get(args.answer_field, "")).strip() != ""
    )
    
    if args.num_train_examples and args.num_train_examples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_train_examples))

    cot_trimmed_examples = 0

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(example):
        q = to_text(example.get(args.question_field, ""))
        nonlocal cot_trimmed_examples
        a_raw = to_text(example.get(args.answer_field, ""))
        a = format_target_answer(a_raw, final_only=args.answer_after_cot_only)
        if args.answer_after_cot_only and a.strip() != (a_raw or "").strip():
            cot_trimmed_examples += 1

        prompt_only = apply_chat_template_safe(tokenizer, 
            [
                {
                    "role": "system",
                    "content": "You are a careful medical assistant. Answer the medical question directly and concisely. Give the final answer first. Do not include unnecessary explanation.",
                },
                {"role": "user", "content": q.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        full = apply_chat_template_safe(tokenizer, 
            build_messages(q, a),
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tok = tokenizer(prompt_only, truncation=True, max_length=args.max_length, add_special_tokens=False)
        full_tok = tokenizer(
            full,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            add_special_tokens=False,
        )

        labels = full_tok["input_ids"].copy()
        prompt_len = min(len(prompt_tok["input_ids"]), args.max_length)
        for i in range(prompt_len):
            labels[i] = -100
        for i, m in enumerate(full_tok["attention_mask"]):
            if m == 0:
                labels[i] = -100

        full_tok["labels"] = labels
        return full_tok

    tokenized = ds.map(tok, remove_columns=ds.column_names)

    if args.answer_after_cot_only:
        print(f"CoT-trimmed answers: {cot_trimmed_examples}/{len(tokenized)} examples", flush=True)

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
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if use_cuda else torch.float32

    print(f"[model] loading model from {args.model_id} ...", flush=True)
    adapter_cfg_path = os.path.join(args.model_id, "adapter_config.json")
    adapter_path_detected = os.path.isfile(adapter_cfg_path)
    base_ref = ""
    if adapter_path_detected:
        try:
            with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_ref = adapter_cfg.get("base_model_name_or_path", "")
            print(
                f"[model] detected PEFT adapter path; base model will be loaded from: {base_ref}",
                flush=True,
            )
        except Exception as e:
            print(f"[model] warning: failed to read adapter config ({e})", flush=True)

    _load_t0 = time.perf_counter()
    if adapter_path_detected:
        if PeftModel is None:
            raise ImportError("PEFT is required to load adapter checkpoints. Install with: pip install peft")
        base_model_id = args.base_model_id.strip() or os.environ.get("BASE_MODEL_ID", "").strip() or base_ref.strip()
        if not base_model_id:
            raise ValueError("Could not resolve base model for adapter checkpoint. Set --base_model_id.")
        print(f"[model] loading base model first: {base_model_id}", flush=True)
        _base_kwargs = {"cache_dir": cache_dir, "dtype": dtype}
        if use_cuda:
            _base_kwargs["device_map"] = {"": "cuda:0"}
            _base_kwargs["low_cpu_mem_usage"] = True
        _base_kwargs.setdefault("attn_implementation", os.environ.get("TRAIN_ATTN_IMPL", "sdpa"))
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **_base_kwargs)
        print(f"[model] base model loaded in {time.perf_counter() - _load_t0:.1f}s", flush=True)
        _adapter_t0 = time.perf_counter()
        print(f"[model] attaching adapter weights from: {args.model_id}", flush=True)
        model = PeftModel.from_pretrained(base_model, args.model_id, is_trainable=True)
        print(f"[model] adapter weights attached in {time.perf_counter() - _adapter_t0:.1f}s", flush=True)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, cache_dir=cache_dir, dtype=dtype,
            attn_implementation=os.environ.get("TRAIN_ATTN_IMPL", "sdpa"),
        )
        print(f"[model] base model loaded in {time.perf_counter() - _load_t0:.1f}s", flush=True)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[model] moving base model to device={device} before adapter init...", flush=True)
    model.to(device)
    if args.use_olora and not adapter_path_detected:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("--use_olora requires PEFT. Install with: pip install peft")
        target_modules = [m.strip() for m in args.olora_target_modules.split(",") if m.strip()]
        print(f"[model] applying O-LoRA adapters on modules: {target_modules}", flush=True)
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.olora_r,
            lora_alpha=args.olora_alpha,
            lora_dropout=args.olora_dropout,
            target_modules=target_modules,
            bias="none",
            init_lora_weights="olora",
        )
        _olora_t0 = time.perf_counter()
        model = get_peft_model(model, lora_cfg)
        print(f"[model] O-LoRA adapters applied in {time.perf_counter() - _olora_t0:.1f}s", flush=True)
        model.print_trainable_parameters()

    model.train()

    subspace_bases = {}
    if args.use_sculpt_subspace:
        target_modules = [m.strip() for m in args.subspace_target_modules.split(",") if m.strip()]
        print(f"[subspace] building high-subspace SVD bases for modules: {target_modules}", flush=True)
        _sub_t0 = time.perf_counter()
        subspace_bases = build_high_subspace_bases(model, target_modules, args.subspace_top_fraction)
        print(f"[subspace] prepared {len(subspace_bases)} matrices in {time.perf_counter() - _sub_t0:.1f}s", flush=True)

    def run_probe(step_idx: int):
        if args.probe_every <= 0:
            return
        model.eval()
        for probe_id, probe_q, probe_t in FIXED_PROBES:
            prompt_only = apply_chat_template_safe(tokenizer, 
                [
                    {
                        "role": "system",
                        "content": "You are a careful medical assistant. Answer the medical question directly and concisely. Give the final answer first. Do not include unnecessary explanation.",
                    },
                    {"role": "user", "content": probe_q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=args.max_length)
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
            print(f"[probe] step={step_idx} | {probe_id} | q={probe_q[:120]!r} | ****pred={pred[:160]!r} | ****target={probe_t[:160]!r}", flush=True)
        model.train()

    def collate(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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
                val_out = model(**batch)
                if torch.isfinite(val_out.loss):
                    val_loss_sum += val_out.loss.item()
                    val_batches += 1
                if args.val_max_batches > 0 and val_batches >= args.val_max_batches:
                    break
        model.train()
        if val_batches == 0:
            return None
        return val_loss_sum / val_batches
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)

    max_train_steps = args.max_steps if args.max_steps else (args.epochs * len(loader) // args.grad_accum)
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)
    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    opt.zero_grad(set_to_none=True)
    seen = 0
    opt_step = 0
    accum_loss = 0.0

    pbar = tqdm(total=max_train_steps, unit="step")
    _step_time_accum = 0.0
    _step_time_count = 0
    _t0 = time.time()

    for _epoch in range(args.epochs):
        print(f"Epoch {_epoch + 1}/{args.epochs}")
        for batch in loader:
            seen += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                print(f"[skip] opt_step={opt_step} non-finite loss={loss.item():.4f}, skipping.", flush=True)
                opt.zero_grad(set_to_none=True)
                accum_loss = 0.0
                seen -= 1
                continue
            accum_loss += loss.item()
            (loss / args.grad_accum).backward()

            if seen % args.grad_accum == 0:
                if args.use_sculpt_subspace and subspace_bases:
                    project_grads_to_low_subspace(model, subspace_bases)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                # if args.save_strategy == "steps" and args.save_steps > 0 and opt_step % args.save_steps == 0:
                #     checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{opt_step}")
                #     print(f"\nSaving intermediate checkpoint at step {opt_step} to {checkpoint_dir} ...", flush=True)
                #     os.makedirs(checkpoint_dir, exist_ok=True)
                #     model.save_pretrained(checkpoint_dir)
                #     tokenizer.save_pretrained(checkpoint_dir)

                _step_s = time.time() - _t0
                _t0 = time.time()
                _step_time_accum += _step_s
                _step_time_count += 1
                avg_step_s = _step_time_accum / _step_time_count
                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step_s=f"{avg_step_s:.2f}s")
                pbar.update(1)

                if opt_step % 10 == 0:
                    val_msg = ""
                    if args.val_every > 0 and opt_step % args.val_every == 0:
                        val_loss = compute_val_loss()
                        if val_loss is not None:
                            val_msg = f" | ValLoss: {val_loss:.4f}"
                    print(f"Step {opt_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s{val_msg}", flush=True)

                if args.probe_every > 0 and opt_step % args.probe_every == 0:
                    run_probe(opt_step)

                if args.max_steps and opt_step >= args.max_steps:
                    break
        if args.max_steps and opt_step >= args.max_steps:
            break

    pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    if getattr(args, "use_olora", False) and hasattr(model, "merge_and_unload"):
        print("Merging LoRA adapters into base weights before saving full model...", flush=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir, safe_serialization=True)
    else:
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

