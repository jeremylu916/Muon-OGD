import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import evaluate
except Exception:
    evaluate = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET_ID = "keivalya/MedQuad-MedicalQnADataset"
DEFAULT_DATASET_CONFIG = ""
DEFAULT_SPLIT = "train"

DEFAULT_QUESTION_FIELD = "Question"
DEFAULT_ANSWER_FIELD = "Answer"
DEFAULT_QTYPE_FIELD = "qtype"

DEFAULT_QTYPE = ""
DEFAULT_NUM_EXAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_OUT_FILE = "results/medical/medquad_eval.json"
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_BERTSCORE_BATCH_SIZE = 1
DEFAULT_BERTSCORE_DEVICE = ""
DEFAULT_BERTSCORE_MAX_CHARS = 1200

# Optional local held-out index files.
# These should contain one integer dataset index per line, or a JSON/JSONL list of ints.
DEFAULT_TRAIN_INDICES = ""
DEFAULT_VAL_INDICES = ""
DEFAULT_TEST_INDICES = ""
DEFAULT_EVAL_SPLIT_NAME = "test"  # train / val / test


def get_hf_cache_dir():
    cache_dir = os.environ.get("HF_CACHE_DIR", "").strip()
    return cache_dir or None


def load_tokenizer(model_id: str, cache_dir=None) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            fix_mistral_regex=True,
            trust_remote_code=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # CRITICAL FIX FOR BATCHED GENERATION:
    tokenizer.padding_side = "left"
    
    return tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a model on MedQuad with held-out split support.")

    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)

    p.add_argument("--question_field", type=str, default=DEFAULT_QUESTION_FIELD)
    p.add_argument("--answer_field", type=str, default=DEFAULT_ANSWER_FIELD)
    p.add_argument("--qtype_field", type=str, default=DEFAULT_QTYPE_FIELD)
    p.add_argument("--qtype", type=str, default=DEFAULT_QTYPE, help="Optional qtype filter")

    p.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY)

    p.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE)
    p.add_argument("--save_all_predictions", action="store_true")

    # Held-out split support
    p.add_argument("--eval_split_name", type=str, default=DEFAULT_EVAL_SPLIT_NAME,
                   help="Which logical split to evaluate: train / val / test")
    p.add_argument("--train_indices_file", type=str, default=DEFAULT_TRAIN_INDICES)
    p.add_argument("--val_indices_file", type=str, default=DEFAULT_VAL_INDICES)
    p.add_argument("--test_indices_file", type=str, default=DEFAULT_TEST_INDICES)

    # Metrics
    p.add_argument("--compute_rouge", action="store_true", help="Compute ROUGE-L via evaluate")
    p.add_argument("--compute_bertscore", action="store_true", help="Compute BERTScore via evaluate")
    p.add_argument("--bertscore_model_type", type=str, default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--bertscore_lang", type=str, default="en")
    p.add_argument("--bertscore_batch_size", type=int, default=DEFAULT_BERTSCORE_BATCH_SIZE,
                   help="Batch size used by BERTScore. Lower this to reduce memory usage.")
    p.add_argument("--bertscore_device", type=str, default=DEFAULT_BERTSCORE_DEVICE,
                   help="Device for BERTScore, e.g. 'cpu' or 'cuda:0'. Empty means auto.")
    p.add_argument("--bertscore_max_chars", type=int, default=DEFAULT_BERTSCORE_MAX_CHARS,
                   help="Truncate predictions/references to this many chars for ROUGE/BERTScore.")

    # Generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)

    # Optional batch generation
    p.add_argument("--batch_size", type=int, default=1)

    return p.parse_args()


def to_text(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join([str(x) for x in v])
    if isinstance(v, dict):
        return "\n".join([f"{k}: {v[k]}" for k in sorted(v.keys())])
    return str(v)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    from collections import Counter
    pred_c = Counter(pred_toks)
    gold_c = Counter(gold_toks)
    common = sum((pred_c & gold_c).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def build_prompt(tokenizer, question: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical assistant. "
                "Answer the user's medical question clearly, accurately, and concisely. "
                "Do not add unnecessary disclaimers or extra unrelated information."
            ),
        },
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _find_field(columns, preferred, candidates):
    if preferred in columns:
        return preferred
    for c in candidates:
        if c in columns:
            return c
    return preferred


def choose_torch_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_index_file(path: str) -> List[int]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")

    # Supports:
    # - one int per line
    # - JSON list of ints
    # - JSONL objects like {"idx": 123}
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # JSON list
    if raw.startswith("["):
        data = json.loads(raw)
        return [int(x) for x in data]

    # JSONL or line-separated ints
    indices = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            if "idx" not in obj:
                raise ValueError(f"JSONL line missing 'idx': {line}")
            indices.append(int(obj["idx"]))
        else:
            indices.append(int(line))
    return indices


def apply_heldout_split(
    ds: Dataset,
    eval_split_name: str,
    train_indices_file: str,
    val_indices_file: str,
    test_indices_file: str,
) -> Dataset:
    split_name = eval_split_name.lower().strip()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"eval_split_name must be one of train/val/test, got: {eval_split_name}")

    idx_map = {
        "train": load_index_file(train_indices_file) if train_indices_file else [],
        "val": load_index_file(val_indices_file) if val_indices_file else [],
        "test": load_index_file(test_indices_file) if test_indices_file else [],
    }

    # If no index files are provided, return ds as-is.
    # Warn when user asked for val/test logical split but nothing is enforcing it.
    if not any(idx_map.values()):
        if split_name != "train":
            print(
                f"[warn] eval_split_name='{split_name}' requested but no index files were provided. "
                "Evaluating on the raw source split as-is.",
                flush=True,
            )
        return ds

    selected = idx_map[split_name]
    if not selected:
        raise ValueError(
            f"No indices found for eval split '{split_name}'. "
            f"Provide the corresponding *_indices_file."
        )

    max_idx = len(ds) - 1
    bad = [i for i in selected if i < 0 or i > max_idx]
    if bad:
        raise ValueError(f"Found out-of-range indices for split '{split_name}', examples: {bad[:10]}")

    return ds.select(selected)


def maybe_subsample(ds: Dataset, num_examples: int, seed: int) -> Dataset:
    if num_examples <= 0 or num_examples >= len(ds):
        return ds
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    return ds.select(indices[:num_examples])


def compute_overlap_metrics(
    predictions: List[str],
    references: List[str],
    compute_rouge: bool,
    compute_bertscore: bool,
    bertscore_model_type: str,
    bertscore_lang: str,
) -> Dict[str, Any]:
    metrics = {}

    if (compute_rouge or compute_bertscore) and evaluate is None:
        raise ImportError(
            "The 'evaluate' package is required for ROUGE/BERTScore. "
            "Install it with: pip install evaluate rouge_score bert_score"
        )

    if compute_rouge:
        rouge = evaluate.load("rouge")
        rouge_out = rouge.compute(predictions=predictions, references=references)
        # Keep the common fields explicitly
        metrics["rouge1"] = float(rouge_out.get("rouge1", 0.0))
        metrics["rouge2"] = float(rouge_out.get("rouge2", 0.0))
        metrics["rougeL"] = float(rouge_out.get("rougeL", 0.0))
        metrics["rougeLsum"] = float(rouge_out.get("rougeLsum", 0.0))

    if compute_bertscore:
        bertscore = evaluate.load("bertscore")
        bs = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type=bertscore_model_type,
            lang=bertscore_lang,
        )
        metrics["bertscore_precision"] = float(sum(bs["precision"]) / len(bs["precision"])) if bs["precision"] else 0.0
        metrics["bertscore_recall"] = float(sum(bs["recall"]) / len(bs["recall"])) if bs["recall"] else 0.0
        metrics["bertscore_f1"] = float(sum(bs["f1"]) / len(bs["f1"])) if bs["f1"] else 0.0

    return metrics


def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def truncate_text_for_overlap(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()

    cfg = args.dataset_config.strip() or None
    ds = load_dataset(args.dataset_id, cfg, split=args.split, cache_dir=cache_dir)

    # Preserve original dataset row ids for traceability in outputs.
    ds = ds.add_column("__orig_idx", list(range(len(ds))))

    # Apply held-out split first so index files always map to source dataset rows.
    ds = apply_heldout_split(
        ds=ds,
        eval_split_name=args.eval_split_name,
        train_indices_file=args.train_indices_file,
        val_indices_file=args.val_indices_file,
        test_indices_file=args.test_indices_file,
    )

    args.question_field = _find_field(ds.column_names, args.question_field, ["Question", "question", "query"])
    args.answer_field = _find_field(ds.column_names, args.answer_field, ["Answer", "answer", "response"])
    args.qtype_field = _find_field(ds.column_names, args.qtype_field, ["qtype", "QType", "type", "question_type"])

    if args.question_field not in ds.column_names or args.answer_field not in ds.column_names:
        raise ValueError(
            f"Could not find question/answer fields. "
            f"question_field='{args.question_field}', answer_field='{args.answer_field}', columns={ds.column_names}"
        )

    # Optional qtype filter
    if args.qtype.strip():
        if args.qtype_field not in ds.column_names:
            raise ValueError(f"qtype field '{args.qtype_field}' not found. Columns: {ds.column_names}")
        wanted = args.qtype.strip().lower()
        ds = ds.filter(lambda ex: str(ex.get(args.qtype_field, "")).lower() == wanted)
        if len(ds) == 0:
            raise ValueError(f"No examples left after qtype filter '{args.qtype}'.")

    # Remove empty rows
    ds = ds.filter(
        lambda ex: to_text(ex.get(args.question_field, "")).strip() != ""
        and to_text(ex.get(args.answer_field, "")).strip() != ""
    )

    # Optional subsample
    ds = maybe_subsample(ds, args.num_examples, args.seed)

    tokenizer = load_tokenizer(args.model_id, cache_dir=cache_dir)
    torch_dtype = choose_torch_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_preds: List[str] = []
    all_refs: List[str] = []
    all_qtypes: List[str] = []
    rows: List[Dict[str, Any]] = []

    em_correct = 0
    f1_sum = 0.0

    examples = []
    for idx in range(len(ds)):
        ex = ds[idx]
        q = to_text(ex.get(args.question_field, ""))
        gold = to_text(ex.get(args.answer_field, ""))
        qtype = str(ex.get(args.qtype_field, "unknown")) if args.qtype_field in ds.column_names else "unknown"
        examples.append(
            {
                "dataset_idx": int(ex.get("__orig_idx", idx)),
                "question": q,
                "gold": gold,
                "qtype": qtype,
                "prompt": build_prompt(tokenizer, q),
            }
        )

    total = len(examples)
    pbar = tqdm(batch_iter(examples, args.batch_size), total=(total + args.batch_size - 1) // args.batch_size,
                desc="Evaluating", unit="batch")

    seen = 0
    for batch in pbar:
        prompts = [x["prompt"] for x in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0.0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
            gen_kwargs["top_k"] = args.top_k

        with torch.no_grad():
            outputs = model.generate(**encoded, **gen_kwargs)

        # With left padding enabled, all rows share the same prompt tensor length.
        # Slice generated tokens from this common length; using per-row non-pad lengths
        # can accidentally keep pieces of the prompt in decoded predictions.
        prompt_len = encoded["input_ids"].shape[1]

        for j, item in enumerate(batch):
            pred_ids = outputs[j][prompt_len:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            gold = item["gold"]

            em = exact_match_score(pred, gold)
            f1 = token_f1(pred, gold)

            em_correct += em
            f1_sum += f1

            all_preds.append(pred)
            all_refs.append(gold)
            all_qtypes.append(item["qtype"])

            # Always keep a compact preview. Optionally keep every prediction.
            if args.save_all_predictions or len(rows) < 30:
                rows.append(
                    {
                        "dataset_idx": item["dataset_idx"],
                        "question": item["question"],
                        "gold": gold,
                        "pred": pred,
                        "qtype": item["qtype"],
                        "exact_match": bool(em),
                        "token_f1": f1,
                    }
                )

        seen += len(batch)
        if args.progress_every > 0 and (seen % args.progress_every == 0 or seen == total):
            print(f"Processed {seen}/{total}")

    exact_match = em_correct / total if total else 0.0
    avg_token_f1 = f1_sum / total if total else 0.0

    overall = {
        "exact_match": exact_match,
        "avg_token_f1": avg_token_f1,
    }

    overlap_preds = [truncate_text_for_overlap(x, args.bertscore_max_chars) for x in all_preds]
    overlap_refs = [truncate_text_for_overlap(x, args.bertscore_max_chars) for x in all_refs]

    overall.update(
        compute_overlap_metrics(
            predictions=overlap_preds,
            references=overlap_refs,
            compute_rouge=args.compute_rouge,
            compute_bertscore=False,
            bertscore_model_type=args.bertscore_model_type,
            bertscore_lang=args.bertscore_lang,
        )
    )

    if args.compute_bertscore:
        bs_device = args.bertscore_device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            bertscore = evaluate.load("bertscore")
            bs = bertscore.compute(
                predictions=overlap_preds,
                references=overlap_refs,
                model_type=args.bertscore_model_type,
                lang=args.bertscore_lang,
                device=bs_device,
                batch_size=max(1, args.bertscore_batch_size),
            )
        except RuntimeError as e:
            # OOM fallback: clear CUDA cache and retry on CPU.
            if torch.cuda.is_available() and "out of memory" in str(e).lower():
                print("[warn] BERTScore CUDA OOM. Retrying on CPU...", flush=True)
                torch.cuda.empty_cache()
                bertscore = evaluate.load("bertscore")
                bs = bertscore.compute(
                    predictions=overlap_preds,
                    references=overlap_refs,
                    model_type=args.bertscore_model_type,
                    lang=args.bertscore_lang,
                    device="cpu",
                    batch_size=1,
                )
            else:
                raise

        overall["bertscore_precision"] = float(sum(bs["precision"]) / len(bs["precision"])) if bs["precision"] else 0.0
        overall["bertscore_recall"] = float(sum(bs["recall"]) / len(bs["recall"])) if bs["recall"] else 0.0
        overall["bertscore_f1"] = float(sum(bs["f1"]) / len(bs["f1"])) if bs["f1"] else 0.0

    # Per-qtype breakdown
    by_qtype = {}
    grouped_indices = defaultdict(list)
    for i, qt in enumerate(all_qtypes):
        grouped_indices[qt].append(i)

    for qt, idxs in grouped_indices.items():
        preds_q = [all_preds[i] for i in idxs]
        refs_q = [all_refs[i] for i in idxs]
        overlap_preds_q = [truncate_text_for_overlap(x, args.bertscore_max_chars) for x in preds_q]
        overlap_refs_q = [truncate_text_for_overlap(x, args.bertscore_max_chars) for x in refs_q]

        em_q = sum(exact_match_score(p, r) for p, r in zip(preds_q, refs_q)) / len(idxs)
        f1_q = sum(token_f1(p, r) for p, r in zip(preds_q, refs_q)) / len(idxs)

        q_metrics = {
            "count": len(idxs),
            "exact_match": em_q,
            "avg_token_f1": f1_q,
        }
        q_metrics.update(
            compute_overlap_metrics(
                predictions=overlap_preds_q,
                references=overlap_refs_q,
                compute_rouge=args.compute_rouge,
                compute_bertscore=False,
                bertscore_model_type=args.bertscore_model_type,
                bertscore_lang=args.bertscore_lang,
            )
        )

        if args.compute_bertscore:
            bs_device = args.bertscore_device.strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
            try:
                bertscore = evaluate.load("bertscore")
                bs = bertscore.compute(
                    predictions=overlap_preds_q,
                    references=overlap_refs_q,
                    model_type=args.bertscore_model_type,
                    lang=args.bertscore_lang,
                    device=bs_device,
                    batch_size=max(1, args.bertscore_batch_size),
                )
            except RuntimeError as e:
                if torch.cuda.is_available() and "out of memory" in str(e).lower():
                    print(f"[warn] BERTScore CUDA OOM on qtype={qt}. Retrying on CPU...", flush=True)
                    torch.cuda.empty_cache()
                    bertscore = evaluate.load("bertscore")
                    bs = bertscore.compute(
                        predictions=overlap_preds_q,
                        references=overlap_refs_q,
                        model_type=args.bertscore_model_type,
                        lang=args.bertscore_lang,
                        device="cpu",
                        batch_size=1,
                    )
                else:
                    raise

            q_metrics["bertscore_precision"] = float(sum(bs["precision"]) / len(bs["precision"])) if bs["precision"] else 0.0
            q_metrics["bertscore_recall"] = float(sum(bs["recall"]) / len(bs["recall"])) if bs["recall"] else 0.0
            q_metrics["bertscore_f1"] = float(sum(bs["f1"]) / len(bs["f1"])) if bs["f1"] else 0.0

        by_qtype[qt] = q_metrics

    payload = {
        "task": "medquad",
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "source_split": args.split,
        "eval_split_name": args.eval_split_name,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
        "qtype_field": args.qtype_field,
        "qtype_filter": args.qtype,
        "num_examples": total,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "bertscore_max_chars": args.bertscore_max_chars,
        "bertscore_batch_size": args.bertscore_batch_size,
        "bertscore_device": args.bertscore_device,
        "metrics": overall,
        "per_qtype": by_qtype,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_predictions": rows,
    }

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload["metrics"], indent=2))
    print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()