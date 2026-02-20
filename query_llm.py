import os
from typing import List
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# HF cache: use HF_CACHE_DIR if set (e.g. llm-exp-hf-cache on Kubeflow), else repo root hf_cache
_env_cache = os.getenv("HF_CACHE_DIR")
if _env_cache:
    CACHE_DIR = os.path.abspath(_env_cache) if os.path.isabs(_env_cache) else os.path.join(_REPO_ROOT, _env_cache)
else:
    CACHE_DIR = os.path.join(_REPO_ROOT, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SYS_PROMPT = "You are a helpful assistant."


# ========== 1.  Base class (keeps common boilerplate) =====================
class _BaseGenerator:
    """
    Load model/tokenizer/pipeline once.
    Child classes implement `_build_prompt()`.

    Public method:
        generate(list[str], max_new_tokens=64, batch_size=4) -> list[str]
    """
    def __init__(self, model_name: str, system_prompt: str = SYS_PROMPT):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                token=self.hf_token,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=self.hf_token,
                trust_remote_code=True,
            )
        except OSError as e:
            raise OSError(
                "Failed to load model/tokenizer from Hugging Face. "
                "Check (1) model_name is valid (for Qwen3 use Qwen/Qwen3-8B), "
                "and (2) your token is set via HF_TOKEN or HUGGINGFACE_HUB_TOKEN "
                "for private/gated repos."
            ) from e
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer,)

        # ensure the tokenizer can pad
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    # ------------- public -------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompts: List[str], max_new_tokens: int = 64, batch_size: int = 1, temperature: float = 0.0) -> List[str]:
        eos_id = self.tokenizer.eos_token_id
        results: List[str] = []

        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            prompt_texts = [self._build_prompt(p) for p in chunk]
            gen_kw = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": eos_id,
                "pad_token_id": eos_id,
                "batch_size": len(prompt_texts),
                "return_full_text": False,
                "repetition_penalty": 1.15,
            }

            if temperature == 0.0:
                outputs = self.pipe(
                    prompt_texts,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    **gen_kw,
                )
            else:
                outputs = self.pipe(
                    prompt_texts,
                    do_sample=True,
                    temperature=temperature,
                    **gen_kw,
                )

            for output in outputs:
                results.append(output[0]["generated_text"])
        return results
    
    @torch.inference_mode()
    def generate_multiple(self, prompts: List[str], num_samples: int = 3, max_new_tokens: int = 64, batch_size: int = 1, temperature: float = 0.7) -> List[List[str]]:
        """
        Generate multiple samples for each prompt.
        
        Args:
            prompts: List of input prompts
            num_samples: Number of samples to generate per prompt
            max_new_tokens: Maximum new tokens to generate
            batch_size: Batch size for processing
            temperature: Sampling temperature (should be > 0 for diversity)
            
        Returns:
            List of lists, where each inner list contains num_samples responses for the corresponding prompt
        """
        if temperature == 0.0:
            raise ValueError("Temperature must be > 0 for multiple sampling.")
            
        eos_id = self.tokenizer.eos_token_id
        results: List[List[str]] = []

        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            prompt_texts = [self._build_prompt(p) for p in chunk]
            
            outputs = self.pipe(
                prompt_texts,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_samples,  # This generates multiple samples per prompt
                batch_size=len(prompt_texts),
                return_full_text=False,
            )
            
            # outputs is now structured as [prompt1_samples, prompt2_samples, ...]
            # where each prompt_samples contains num_samples outputs
            for output in outputs:
                prompt_samples = [sample["generated_text"] for sample in output]
                results.append(prompt_samples)
                
        return results

    # ------------- to be overridden --------------------------------------
    def _build_prompt(self, user_prompt: str) -> str:
        """Return the concrete prompt string fed to the model."""
        raise NotImplementedError


# ========== 2.  Llama-family =============================================
class LlamaGenerator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

# ========== 3.  Qwen-family ==============================================
class QwenGenerator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

# ========== 3.1  Qwen3==============================================
class Qwen3ThinkGenerator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )

# ========== 3.2  Qwen3==============================================
# For Qwen3, the non-thinking mode is preferred for standard generation tasks.
class Qwen3Generator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )

# ========== 4.  Mistral/Mixtral ==========================================
class MistralGenerator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        return self.system_prompt.rstrip() + "\n\n" + user_prompt

# ========== 5.  Gemma-family ==========================================
class GemmaGenerator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        # Gemma models don't support system roles in chat templates
        # Build prompt manually instead
        return f"{self.system_prompt}\n\n{user_prompt}"


# ========== 6.  GLM-Z1 (Z.ai) ==========================================
# GLM-Z1's built-in chat_template uses visible = content.split(' ')[-1] for user, so only
# the last "word" is shown and the actual question is dropped. Build prompt manually so
# the full user content is sent: [gMASK]<|user|>\n{content}\n<|assistant|>
class GLMZ1Generator(_BaseGenerator):
    def _build_prompt(self, user_prompt: str) -> str:
        return "[gMASK]<|user|>\n" + user_prompt.rstrip() + "\n<|assistant|>"
