# Muon-OGD

**Muon-OGD for LLM Continual Learning.**

This repository contains training and evaluation scripts to run a **3-stage continual learning** pipeline on an instruction-tuned Llama model, with evaluations after each stage.

- **Stages**: **A (coding)** → **B (math)** → **C (medical)**
- **Methods supported**:
  - **AdamW** (sequential SFT baseline)
  - **O-LoRA**
  - **Sculpting Subspace**
  - **Muon-OGD**
- **Evaluation** is **vLLM-accelerated** (enabled by default) via `vllm_eval_backend.py`.

---

## Repository layout

Key entrypoints at the repo root:

- `llama-3B.sh` — unified runner for the full continual-learning pipeline (train + eval + summary aggregation)
- `train_coding_bigcodebench_sft.py` — Stage A baseline training (SeqSFT/AdamW)
- `train_coding_bigcodebench_muon_ogd.py` — Stage A Muon-OGD training
- `train_math_sft.py` — Stage B baseline training
- `train_math_muon_ogd.py` — Stage B Muon-OGD training
- `train_medical_sft.py` — Stage C baseline training
- `train_medical_muon_ogd.py` — Stage C Muon-OGD training
- `vllm_eval_backend.py` — vLLM backend glue for evaluation scripts

> Note: the `llama-3B.sh` runner expects additional scripts to exist (e.g. `eval_gsm8k.py`, `eval_medical.py`, `eval_bigcodebench_remote.py`, `train_math_sft_svd.py`). If they’re not in your checkout yet, the runner will exit with a “missing file” error.

---

## Quickstart

### 1) Create an environment

Use whichever workflow you prefer (conda/venv). A minimal starting point:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Then install core dependencies (exact versions may vary depending on your stack):

```bash
pip install torch transformers datasets accelerate
```

Optional dependencies depending on the methods you enable:

- **O-LoRA** requires `peft`:
  ```bash
  pip install peft
  ```
- **vLLM eval** requires `vllm`:
  ```bash
  pip install vllm
  ```

---

### 2) Run the full pipeline (recommended)

The main entrypoint is:

```bash
bash llama-3B.sh
```

By default, this runs **all methods** and performs eval after each stage.

Outputs are written under a run root like:

- `${RUN_ROOT}/adamw_run*/...`
- `${RUN_ROOT}/olora_run*/...`
- `${RUN_ROOT}/subspace_run*/...`
- `${RUN_ROOT}/muon_run*/...`

At the end, the script writes an aggregated summary:

- `${RUN_ROOT}/summary_mean_std_all.json`

---

## Configuration (environment variables)

`llama-3B.sh` is designed to be configured via environment variables.

### Paths

- `USER_NAME` (default: `$USER`, fallback: `unknown`)
- `PROJECT_ROOT` (default: current directory, `$PWD`)
- `ROOT_DIR` (default: `${PROJECT_ROOT}`)
- `RUN_ROOT` (default: `${ROOT_DIR}/runs`, i.e. `./runs` when using defaults)

Example:

```bash
USER_NAME=$USER \
PROJECT_ROOT=$PWD \
ROOT_DIR=$PWD \
RUN_ROOT=$PWD/runs \
bash llama-3B.sh
```

### Model

- `MODEL_ID` (default: `unsloth/Llama-3.2-3B-Instruct`)
- `BASE_MODEL_ID` (default: `${MODEL_ID}`)

If your model is gated or you want to run from a local snapshot:

```bash
MODEL_ID=/path/to/local/Llama-3.2-3B-Instruct bash llama-3B.sh
```

The script performs a small **preflight check** to verify it can load the config + tokenizer/chat template.

### Enable/disable methods

- `RUN_ADAMW=1`
- `RUN_OLORA=1`
- `RUN_SUBSPACE=1`
- `RUN_MUON=1`

Example: run **only Muon-OGD**:

```bash
RUN_ADAMW=0 RUN_OLORA=0 RUN_SUBSPACE=0 RUN_MUON=1 bash llama-3B.sh
```

### Repeats / seeds

- `NUM_REPEATS` (default: `1`) — repeats for AdamW/O-LoRA/Subspace
- `MUON_NUM_REPEATS` (default: `3`) — repeats for Muon-OGD (to report mean ± std)
- `BASE_SEED` (default: `42`)

Example:

```bash
NUM_REPEATS=3 MUON_NUM_REPEATS=5 BASE_SEED=123 bash llama-3B.sh
```

### vLLM evaluation

Enabled by default:

- `USE_VLLM=1`
- `EVAL_VLLM_TP=1`
- `EVAL_VLLM_MAX_MODEL_LEN=4096`
- `EVAL_VLLM_GPU_MEM_UTIL=0.85`

If you want to disable vLLM:

```bash
USE_VLLM=0 bash llama-3B.sh
```

---

## What gets evaluated?

From the runner script, evaluation is performed after each stage on:

- **Coding**: BigCodeBench (remote eval script)
- **Math**: GSM8K
- **Medical**: a verifier-based evaluation using:
  - verifier model: `FreedomIntelligence/medical_o1_verifier_3B`
  - dataset: `FreedomIntelligence/medical-o1-verifiable-problem`

The script saves logs under each run directory and writes JSON outputs for metrics.

---

## Tips / troubleshooting

- If you see model access errors, try:
  - `huggingface-cli login`
  - setting `MODEL_ID` to a local checkpoint path
- If you enable O-LoRA and it fails immediately, ensure `peft` is installed and importable.
- If medical eval OOMs, reduce GPU memory util for medical eval:
  ```bash
  EVAL_VLLM_GPU_MEM_UTIL_MEDICAL=0.45 bash llama-3B.sh
  ```

---

## Citation / acknowledgment

If you use Muon-OGD in academic work, consider adding a citation section here (paper / arXiv / bibtex).

---

## License

No license file detected yet. If you intend this to be reused, consider adding a `LICENSE` (e.g., Apache-2.0 / MIT) and updating this section.
