"""Drop-in vLLM backend used by eval_*.py when USE_VLLM=1.

Mimics the subset of `transformers.AutoModelForCausalLM` surface that the eval
scripts touch: `.eval()`, `.device`, `.generation_config`, and `.generate(**inputs,
max_new_tokens=..., do_sample=..., temperature=..., top_p=..., pad_token_id=...,
eos_token_id=...)`. Returns a 2D tensor `[1, prompt_len + gen_len]` to match HF's
output shape so the existing `out[0][prompt_len:]` decoding pattern keeps working.

Speed wins come from vLLM's faster kernels and (optionally) tensor parallelism.
For maximum throughput refactor the eval loops to submit all prompts at once;
this wrapper keeps the single-prompt-at-a-time API for drop-in compatibility.

Knobs (env):
  USE_VLLM=1                      enable this backend
  VLLM_TP=<int>                   tensor_parallel_size (default 1)
  VLLM_GPU_MEM_UTIL=<float>       gpu_memory_utilization (default 0.85)
  VLLM_MAX_MODEL_LEN=<int>        cap context length (helps with long-context configs)
  VLLM_DTYPE=auto|bfloat16|float16
"""
from __future__ import annotations

import os
import types
from typing import List, Optional, Union

import torch


def vllm_enabled() -> bool:
    return os.environ.get("USE_VLLM", "").strip() in ("1", "true", "TRUE", "yes")


# vLLM 0.20.x probes DeepGEMM (FP8) kernels at warmup even on bf16 models. On
# nodes without `deep_gemm` installed (e.g. GH200 stacks here), that probe
# raises RuntimeError. We default-disable it so eval runs survive; users who
# do have deep_gemm installed and want FP8 can override with VLLM_USE_DEEP_GEMM=1.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")


class VLLMCausalLM:
    def __init__(
        self,
        model_id: str,
        dtype: Optional[Union[str, torch.dtype]] = None,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        trust_remote_code: bool = True,
    ):
        # Patch vLLM's MultimodalContext.get_hf_config to accept transformers'
        # Qwen3_5TextConfig where vLLM expects its own Qwen3_5Config. vLLM 0.20.x
        # routes the text-only qwen3_5_text model_type through the multimodal
        # Qwen3-VL handler, which then refuses the text config class.
        # Combined with the vision_config stub injected via hf_overrides below,
        # this lets the data-parser code path complete on text-only checkpoints.
        try:
            import vllm.multimodal.processing.context as _ctx_mod  # type: ignore
            _orig_get_hf_config = _ctx_mod.InputProcessingContext.get_hf_config

            def _coerce_subconfigs(cfg):
                # vLLM accesses vision_config / text_config with attribute
                # syntax. If hf_overrides delivered them as plain dicts, wrap
                # them in SimpleNamespace so dotted access works. Recurse into
                # nested dicts because qwen3_5_vision has a few nested fields.
                def _to_ns(x):
                    if isinstance(x, dict):
                        return types.SimpleNamespace(
                            **{k: _to_ns(v) for k, v in x.items()}
                        )
                    if isinstance(x, list):
                        return [_to_ns(v) for v in x]
                    return x
                for sub in ("vision_config", "text_config", "audio_config"):
                    v = getattr(cfg, sub, None)
                    if isinstance(v, dict):
                        setattr(cfg, sub, _to_ns(v))
                return cfg

            def _patched_get_hf_config(self, hf_config_type=None):
                cfg = _coerce_subconfigs(self.model_config.hf_config)
                if hf_config_type is None or isinstance(cfg, hf_config_type):
                    return cfg
                # Permit class mismatch for *text* sub-configs of multimodal
                # families (e.g. Qwen3_5TextConfig where Qwen3_5Config is
                # expected). The model code falls back to attribute access,
                # which our stub vision_config covers.
                cls_name = type(cfg).__name__
                expected_name = getattr(hf_config_type, "__name__", str(hf_config_type))
                if cls_name.endswith("TextConfig") and expected_name in cls_name.replace("TextConfig", "Config"):
                    return cfg
                return _orig_get_hf_config(self, hf_config_type)

            _ctx_mod.InputProcessingContext.get_hf_config = _patched_get_hf_config
            print("[vllm] monkeypatched InputProcessingContext.get_hf_config "
                  "to tolerate text-only sub-configs.", flush=True)
        except Exception as _e:
            print(f"[vllm] context patch skipped: {_e}", flush=True)

        from vllm import LLM  # imported lazily so HF-only paths don't require vllm

        if isinstance(dtype, torch.dtype):
            dtype = {torch.bfloat16: "bfloat16", torch.float16: "float16",
                     torch.float32: "float32"}.get(dtype, "auto")
        # Read both new (EVAL_VLLM_*) and legacy (VLLM_*) names for backwards compat.
        # vLLM 0.20+ warns on unknown VLLM_* env vars; the EVAL_VLLM_* names suppress that.
        dtype   = os.environ.get("EVAL_VLLM_DTYPE",         os.environ.get("VLLM_DTYPE",         dtype or "auto"))
        tp      = int(os.environ.get("EVAL_VLLM_TP",        os.environ.get("VLLM_TP",            tensor_parallel_size or 1)))
        mem     = float(os.environ.get("EVAL_VLLM_GPU_MEM_UTIL", os.environ.get("VLLM_GPU_MEM_UTIL", gpu_memory_utilization or 0.85)))
        _ml     = os.environ.get("EVAL_VLLM_MAX_MODEL_LEN", os.environ.get("VLLM_MAX_MODEL_LEN", max_model_len or 0))
        max_len = int(_ml) or None if _ml is not None else None

        kwargs = dict(
            model=model_id,
            dtype=dtype,
            tensor_parallel_size=tp,
            gpu_memory_utilization=mem,
            trust_remote_code=trust_remote_code,
        )
        if max_len is not None:
            kwargs["max_model_len"] = max_len

        # Workaround for vLLM 0.20.x: the qwen3_5_text (text-only Qwen3.5) config
        # is mistakenly routed through the multimodal Qwen3-VL handler. We
        #   (a) inject a stub vision_config via hf_overrides,
        #   (b) drop a stub preprocessor_config.json into the checkpoint dir
        #       so AutoImageProcessor.from_pretrained() finds something to load.
        try:
            import json as _json
            cfg_path = os.path.join(model_id, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as _f:
                    _cfg = _json.load(_f)
                if (_cfg.get("model_type") == "qwen3_5_text"
                        and "vision_config" not in _cfg):
                    print("[vllm] qwen3_5_text detected — injecting stub vision_config "
                          "via hf_overrides to satisfy vLLM's multimodal handler.",
                          flush=True)
                    kwargs["hf_overrides"] = {
                        "vision_config": {
                            "model_type": "qwen3_5_vision",
                            "depth": 1,
                            "hidden_size": 32,
                            "num_heads": 1,
                            "in_chans": 3,
                            "patch_size": 14,
                            "spatial_merge_size": 2,
                            "spatial_patch_size": 14,
                            "temporal_patch_size": 2,
                            "tokens_per_second": 2,
                            "window_size": 64,
                            "out_hidden_size": 32,
                            "intermediate_size": 32,
                            "fullatt_block_indexes": [0],
                            "hidden_act": "silu",
                        },
                    }

                    # Drop a stub preprocessor_config.json so AutoImageProcessor
                    # has something to load. We never feed images, so the
                    # processor object only has to *exist*.
                    pp_path = os.path.join(model_id, "preprocessor_config.json")
                    if not os.path.exists(pp_path) and os.access(model_id, os.W_OK):
                        stub = {
                            "image_processor_type": "Qwen2VLImageProcessor",
                            "processor_class": "Qwen2_5_VLProcessor",
                            "patch_size": 14,
                            "merge_size": 2,
                            "min_pixels": 3136,
                            "max_pixels": 12845056,
                            "do_convert_rgb": True,
                            "do_normalize": True,
                            "do_rescale": True,
                            "do_resize": True,
                            "image_mean": [0.48145466, 0.4578275, 0.40821073],
                            "image_std":  [0.26862954, 0.26130258, 0.27577711],
                            "rescale_factor": 1 / 255.0,
                            "resample": 3,
                            "size": {"longest_edge": 12845056, "shortest_edge": 3136},
                            "temporal_patch_size": 2,
                            "vision_token_id": 151654,
                            "_note": "Stub written by vllm_eval_backend.py to satisfy AutoImageProcessor.",
                        }
                        with open(pp_path, "w") as _f:
                            _json.dump(stub, _f, indent=2)
                        print(f"[vllm] wrote stub preprocessor_config.json -> {pp_path}", flush=True)
        except Exception as _e:
            print(f"[vllm] hf_overrides / preprocessor stub skipped: {_e}", flush=True)

        print(f"[vllm] init model={model_id} dtype={dtype} tp={tp} "
              f"gpu_mem_util={mem} max_model_len={max_len}", flush=True)
        self._llm = LLM(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_config = types.SimpleNamespace(temperature=1.0, top_p=1.0, top_k=50)

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **_unused,
    ) -> torch.Tensor:
        from vllm import SamplingParams, TokensPrompt

        if input_ids.dim() != 2:
            raise ValueError(f"VLLMCausalLM.generate expects 2D input_ids, got {input_ids.shape}")
        prompts = [row.tolist() for row in input_ids]

        if eos_token_id is None:
            stop_ids = None
        elif isinstance(eos_token_id, int):
            stop_ids = [eos_token_id]
        else:
            stop_ids = list(eos_token_id)

        if not do_sample or (temperature is None) or (temperature == 0):
            sp = SamplingParams(
                temperature=0.0,
                max_tokens=int(max_new_tokens),
                stop_token_ids=stop_ids,
            )
        else:
            sp = SamplingParams(
                temperature=float(temperature),
                top_p=float(top_p) if top_p is not None else 1.0,
                top_k=int(top_k) if top_k else -1,
                max_tokens=int(max_new_tokens),
                stop_token_ids=stop_ids,
            )

        outs = self._llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            sp,
            use_tqdm=False,
        )

        rows = []
        max_total = 0
        for prompt_ids, out in zip(prompts, outs):
            gen = list(out.outputs[0].token_ids)
            row = list(prompt_ids) + gen
            rows.append(row)
            if len(row) > max_total:
                max_total = len(row)

        pad = int(pad_token_id) if pad_token_id is not None else 0
        for r in rows:
            if len(r) < max_total:
                r.extend([pad] * (max_total - len(r)))
        return torch.tensor(rows, dtype=torch.long)


def maybe_load_vllm(model_id: str, **kwargs) -> Optional[VLLMCausalLM]:
    if not vllm_enabled():
        return None
    return VLLMCausalLM(model_id=model_id, **kwargs)
