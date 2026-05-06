import argparse
import json
import os
# Default-disable torch's cuDNN SDPA backend; on this GH200 stack it raises
# "cuDNN Frontend error: No valid execution plans built".
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")
import time
from datasets import load_dataset
import torch
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from chat_template_utils import apply_chat_template_safe
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_gsm8k"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_TRAIN_EXAMPLES = 512
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LOG_EVERY = 10
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
DEFAULT_BASE_MODEL_ID = ""

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
    """Load tokenizer with best-effort compatibility flags.

    Some Transformers versions emit a warning about an incorrect regex pattern
    for certain tokenizers. Newer versions support fix_mistral_regex=True.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on GSM8K train split.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--base_model_id",
        type=str,
        default=DEFAULT_BASE_MODEL_ID,
        help="Optional base model path/id when --model_id points to a PEFT adapter directory.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=DEFAULT_NUM_TRAIN_EXAMPLES,
        help="Subsample GSM8K train for a quick pilot.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max optimizer steps (after grad accumulation).",
    )
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--log_every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--probe_every", type=int, default=DEFAULT_PROBE_EVERY, help="Run quick inference probe every N optimizer steps; <=0 disables")
    parser.add_argument("--probe_max_new_tokens", type=int, default=DEFAULT_PROBE_MAX_NEW_TOKENS, help="Max new tokens for each probe inference")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio from tokenized train set")
    parser.add_argument("--val_every", type=int, default=DEFAULT_VAL_EVERY, help="Run validation loss every N optimizer steps; <=0 disables")
    parser.add_argument("--val_max_batches", type=int, default=DEFAULT_VAL_MAX_BATCHES, help="Max validation batches per validation run; <=0 uses full validation set")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps"], help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X optimizer steps")
    parser.add_argument("--use_olora", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_OLORA)
    parser.add_argument("--olora_r", type=int, default=DEFAULT_OLORA_R)
    parser.add_argument("--olora_alpha", type=int, default=DEFAULT_OLORA_ALPHA)
    parser.add_argument("--olora_dropout", type=float, default=DEFAULT_OLORA_DROPOUT)
    parser.add_argument("--olora_target_modules", type=str, default=DEFAULT_OLORA_TARGET_MODULES)
    parser.add_argument("--use_sculpt_subspace", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_SCULPT_SUBSPACE)
    parser.add_argument("--subspace_top_fraction", type=float, default=DEFAULT_SUBSPACE_TOP_FRACTION)
    parser.add_argument("--subspace_target_modules", type=str, default=DEFAULT_SUBSPACE_TARGET_MODULES)
    return parser.parse_args()


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


def build_prompt(tokenizer, question: str, answer: str):
    # Keep this consistent with `eval_gsm8k.py` to reduce prompt/extraction mismatch.
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {
            "role": "user",
            "content": f"Solve the following math word problem. End your response with the final numeric answer.\n\n{question}",
        },
        {"role": "assistant", "content": answer},
    ]
    return apply_chat_template_safe(tokenizer, messages, tokenize=False, add_generation_prompt=False)


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    torch.manual_seed(args.seed)

    dataset = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
    if args.num_train_examples and args.num_train_examples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        # Build prompt/answer separately so we can mask prompt tokens from the loss
        # (otherwise the model learns to "continue" the whole conversation including padding).
        prompt_only_messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {
                "role": "user",
                "content": f"Solve the following math word problem. End your response with the final numeric answer.\n\n{example['question']}",
            },
        ]
        prompt_text = apply_chat_template_safe(tokenizer, 
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Ensure the assistant content is appended directly after the generation prompt.
        # This guarantees `prompt_ids` is a prefix of `full_ids`, matching our masking.
        full_text = prompt_text + example["answer"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Loss only on the assistant answer portion.
        full_labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

        if len(full_ids) > args.max_length:
            # Preserve the *end* of the sequence (the final numeric answer is at the end)
            # instead of truncating away the end.
            input_ids = full_ids[-args.max_length :]
            labels = full_labels[-args.max_length :]
            attention_mask = [1] * args.max_length
        else:
            input_ids = full_ids
            labels = full_labels
            attention_mask = [1] * len(full_ids)

            pad_len = args.max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        # Sanity: all sequences must be exactly `max_length` for batching.
        assert len(input_ids) == args.max_length
        assert len(attention_mask) == args.max_length
        assert len(labels) == args.max_length

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16 if use_cuda else torch.float32

    print(f"[model] loading model from {args.model_id} ...", flush=True)
    adapter_cfg_path = os.path.join(args.model_id, "adapter_config.json")
    adapter_path_detected = os.path.isfile(adapter_cfg_path)
    base_ref = ""
    if os.path.isfile(adapter_cfg_path):
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
        base_model_id = (
            args.base_model_id.strip()
            or os.environ.get("BASE_MODEL_ID", "").strip()
            or base_ref.strip()
        )
        if not base_model_id:
            raise ValueError("Could not resolve base model for adapter checkpoint. Set --base_model_id.")
        print(f"[model] loading base model first: {base_model_id}", flush=True)
        _base_kwargs = {"cache_dir": cache_dir, "dtype": model_dtype}
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
            args.model_id, cache_dir=cache_dir, dtype=model_dtype,
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

    if not use_cuda:
        model = model.float()

    print(f"Using device={device} dtype={next(model.parameters()).dtype}")

    model.train()

    subspace_bases = {}
    if args.use_sculpt_subspace:
        target_modules = [m.strip() for m in args.subspace_target_modules.split(",") if m.strip()]
        print(f"[subspace] building high-subspace SVD bases for modules: {target_modules}", flush=True)
        _sub_t0 = time.perf_counter()
        subspace_bases = build_high_subspace_bases(model, target_modules, args.subspace_top_fraction)
        print(f"[subspace] prepared {len(subspace_bases)} matrices in {time.perf_counter() - _sub_t0:.1f}s", flush=True)

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

    def run_probe(step_idx: int):
        if args.probe_every <= 0:
            return
        model.eval()
        for probe_id, probe_q, probe_t in FIXED_PROBES:
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": probe_q},
            ]
            prompt_text = apply_chat_template_safe(tokenizer, prompt_messages, tokenize=False, add_generation_prompt=True)
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
            print(f"[probe] step={step_idx} | {probe_id} | q={probe_q[:120]!r} | ****pred={pred[:160]!r} | ****target={probe_t[:160]!r}", flush=True)
        model.train()
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps)
    print(f"Total optimization steps: {max_train_steps} | Warmup: {num_warmup_steps}")

    optimizer.zero_grad(set_to_none=True)
    seen_steps = 0
    opt_step = 0
    total_loss = 0.0
    _step_time_accum = 0.0
    _step_time_count = 0

    pbar = tqdm(total=max_train_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for batch in loader:
            batch_start_time = time.perf_counter()
            seen_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum

            if not torch.isfinite(outputs.loss):
                print(f"[skip] opt_step={opt_step}: non-finite loss={outputs.loss.item():.4f}, skipping.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                seen_steps -= 1
                continue

            loss.backward()
            total_loss += loss.item()

            if seen_steps % args.grad_accum == 0:
                has_bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if has_bad_grad:
                    print(f"[skip] opt_step={opt_step}: NaN/Inf in gradients, skipping.", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if args.use_sculpt_subspace and subspace_bases:
                    project_grads_to_low_subspace(model, subspace_bases)

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if args.save_strategy == "steps" and args.save_steps > 0 and opt_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{opt_step}")
                    print(f"\nSaving intermediate checkpoint at step {opt_step} to {checkpoint_dir} ...", flush=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                iter_dt = time.perf_counter() - batch_start_time
                _step_time_accum += iter_dt
                _step_time_count += 1

                if opt_step % args.log_every == 0:
                    avg_loss = total_loss / args.log_every
                    lr = scheduler.get_last_lr()[0]
                    avg_step_s = _step_time_accum / max(1, _step_time_count)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step_s": f"{avg_step_s:.2f}s"})
                    val_msg = ""
                    if args.val_every > 0 and opt_step % args.val_every == 0:
                        val_loss = compute_val_loss()
                        if val_loss is not None:
                            val_msg = f" | ValLoss: {val_loss:.4f}"
                    print(f"Step {opt_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | avg_step_s={avg_step_s:.2f}s{val_msg}")
                    total_loss = 0.0

                if args.probe_every > 0 and opt_step % args.probe_every == 0:
                    run_probe(opt_step)

                pbar.update(1)

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