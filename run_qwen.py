import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer with best-effort compatibility flags."""
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)

tokenizer = load_tokenizer(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Write a short greeting."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
