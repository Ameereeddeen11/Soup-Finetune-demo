# Soup Fine-Tune Demo

Fine-tuning a small LLM (Qwen2.5-1.5B-Instruct) with [Soup](https://github.com/MakazhanAlpamys/Soup) on Apple Silicon (M4 Pro) — a hands-on learning project exploring LoRA fine-tuning end-to-end: data preparation, training, evaluation, and export.

## Motivation

This project was built as a practical introduction to LLM fine-tuning as part of my AI Engineering learning path. Rather than reading about fine-tuning, the goal was to run the full pipeline myself — from a public dataset to a working, chat-testable model — using a lightweight, reproducible toolchain that runs entirely on a laptop GPU.

## Tech Stack

- **[Soup CLI](https://github.com/MakazhanAlpamys/Soup)** — fine-tuning orchestration (config-driven, handles LoRA/quantization/batching automatically)
- **Base model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Method:** LoRA (Low-Rank Adaptation) via SFT (Supervised Fine-Tuning)
- **Backend:** MLX (Apple Silicon)
- **Dataset:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (52k instruction-following examples; 2,000-example subset used here)
- **Hardware:** MacBook Pro, Apple M4 Pro, 24 GB unified memory

## Project Structure

```
soup-finetune-demo/
├── README.md
├── soup.yaml              # training configuration
├── requirements.txt
├── data/
│   └── train.jsonl         # generated locally, not committed (see below)
├── prepare_data/
│   └── main.py     # downloads and formats the Alpaca subset
└── output/                 # trained LoRA adapter, not committed (see below)
```

> **Note:** `data/*.jsonl` and `output/` are excluded via `.gitignore`. The dataset is public and reproducible via `prepare_data/main.py`; the trained weights are a build artifact, not source — regenerate them by running the pipeline below.

## Setup

```bash
git clone <your-repo-url>
cd soup-finetune-demo

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Verify your environment is ready (checks Python version, Apple Silicon/MPS detection, and dependencies):

```bash
soup doctor
```

## Reproducing the Training Run

**1. Prepare the dataset**

Downloads a 2,000-example subset of Alpaca and converts it to the JSONL format Soup expects:

```bash
python prepare_data/main.py
```

**2. Train**

```bash
soup train --config soup.yaml
```

**3. Chat with the fine-tuned model**

```bash
soup chat --model output
```

**4. (Optional) Merge and export**

Merge the LoRA adapter into the base model:

```bash
soup merge --adapter output
```

Export to GGUF for use with Ollama / llama.cpp:

```bash
soup export --model output --format gguf --quant q4_k_m
```

## Configuration

See [`soup.yaml`](./soup.yaml) for the full config. Key choices:

| Setting | Value | Why |
|---|---|---|
| `base` | Qwen2.5-1.5B-Instruct | Small enough to fine-tune quickly on 24 GB unified memory |
| `backend` | mlx | Uses Apple Silicon efficiently |
| `task` | sft | Supervised fine-tuning on instruction/response pairs |
| `training.epochs` | 2 | Small dataset (2k examples) — more epochs risked overfitting |
| `training.lr` | 2e-5 | Standard fine-tuning learning rate; low enough to avoid catastrophic forgetting |
| `training.lora.r` | 16 | Reasonable quality/speed tradeoff for a first run |

## Results

| Metric | Value |
|---|---|
| Training loss | 1.7964 → 1.3564 |
| Duration | 14 minutes |
| Hardware | MacBook Pro M4 Pro (24 GB) |
| Training examples | 1,800 (90% split) |
| Validation examples | 200 (10% split) |

Training loss decreased steadily across both epochs, and the resulting model reliably follows instruction-style prompts in manual testing (e.g., explanations, list generation, short creative writing) — noticeably more consistent in format and completeness than the base model on the same prompts.

Full run metadata is available via:

```bash
soup runs show run_20260916_233719_3eda3450
```

## Limitations

- Trained on a small subset (1,000 of 52,000 available Alpaca examples) — quality would likely improve with the full dataset or more epochs.
- No automated evaluation benchmark was run; results are based on manual, qualitative testing via `soup chat`.
- Single training run — no hyperparameter sweep was performed.

## Acknowledgments

- [Soup](https://github.com/MakazhanAlpamys/Soup) by Alpamys Makazhan — the fine-tuning toolchain used throughout this project.
- [Qwen2.5](https://huggingface.co/Qwen) by Alibaba Cloud.
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.