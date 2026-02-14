import argparse
import os
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_gsm8k"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_TRAIN_EXAMPLES = 512


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with best-effort compatibility flags.

    Some Transformers versions emit a warning about an incorrect regex pattern
    for certain tokenizers. Newer versions support fix_mistral_regex=True.
    """
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on GSM8K train split.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
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
    return parser.parse_args()


def build_prompt(tokenizer, question: str, answer: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Solve the problem and give the final answer.\n\n{question}"},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    dataset = load_dataset("gsm8k", "main", split="train")
    if args.num_train_examples and args.num_train_examples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_train_examples))

    tokenizer = load_tokenizer(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        text = build_prompt(tokenizer, example["question"], example["answer"])
        out = tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    use_cuda = torch.cuda.is_available()
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=model_dtype)

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    if not use_cuda:
        model = model.float()

    print(f"Using device={device} dtype={next(model.parameters()).dtype}")

    model.train()

    def collate(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    optimizer.zero_grad(set_to_none=True)
    seen_steps = 0
    opt_step = 0

    for _epoch in range(args.epochs):
        for batch in loader:
            seen_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = model(**batch).loss
            (loss / args.grad_accum).backward()

            if seen_steps % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % 10 == 0:
                    print(f"opt_step={opt_step} loss={loss.item():.4f}")

                if args.max_steps and opt_step >= args.max_steps:
                    break
        if args.max_steps and opt_step >= args.max_steps:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
