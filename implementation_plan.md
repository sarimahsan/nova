# Modern Transformer from Scratch — Kaggle 2×T4 Training (~295M params)

Build a complete, self-contained ~295M parameter transformer language model from scratch with modern architecture choices (SwiGLU, RMSNorm, RoPE, GQA), train on TinyStories via HuggingFace, produce training plots, and pass pytest — all ready to `git push` → `git pull` on Kaggle and run.

## Context

You have an existing research framework at `e:\transformer` (package name `transformer_research`) that already implements all the core components. We'll **adapt and restructure** this proven codebase into a clean, standalone project at `e:\mistral` with a flat import structure, Kaggle-optimized configs, and a single-file training entrypoint.

## User Review Required

> [!IMPORTANT]
> **Dataset**: Using **TinyStories** (`roneneldan/TinyStories`) — 500K samples, clean English text, trains well at this scale, produces readable output for qualitative evaluation.

> [!IMPORTANT]
> **Model Scale**: Targeting **~295M parameters** — `hidden_dim=1024`, 16 layers, 16 heads (GQA with 4 KV heads), `ffn_dim=4096`, `seq_len=512`. Memory-safe on 2×T4 (32GB total) with bf16. Estimated training: **~6 hours** for 3 epochs on 500K samples.

> [!WARNING]
> **Multi-GPU Strategy**: Using `nn.DataParallel` for simplicity — wraps the model transparently, no launch script changes. GPU 0 memory: ~8.4GB/16GB, GPU 1: ~5.4GB/16GB. If you prefer DDP (better scaling), let me know.

## Open Questions

> [!IMPORTANT]
> **1. Checkpoint saving**: Your `.gitignore` excludes `*.pt` files. The training script will save the best checkpoint **locally on Kaggle** (won't be pushed to git). OK?

> [!IMPORTANT]
> **2. Custom AdamW vs PyTorch AdamW**: Defaulting to **PyTorch fused AdamW** for GPU speed. The custom from-scratch AdamW is available as a flag. OK?

---

## Model Architecture — 295M Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_dim` | 1024 | Fills T4 tensor cores efficiently |
| `num_layers` | 16 | Deep enough for strong language modeling |
| `num_heads` | 16 | 64-dim heads (optimal for RoPE) |
| `num_kv_heads` | 4 | 4:1 GQA ratio — saves 40% KV memory |
| `head_dim` | 64 | Standard, SDPA-optimized |
| `ffn_dim` | 4096 | 4× hidden_dim (SwiGLU standard) |
| `max_seq_len` | 512 | Good context, fits in memory |
| `vocab_size` | 50257 | GPT-2 tokenizer |
| `batch_size` | 48 | 24 per GPU with DataParallel |
| `precision` | bf16 | Native on T4, no GradScaler needed |
| **Total params** | **~295M** | |

### Memory Budget (DataParallel + bf16)

| Component | GPU 0 | GPU 1 |
|-----------|-------|-------|
| Model params (bf16) | 590 MB | 590 MB |
| Optimizer states (fp32) | 2.4 GB | — |
| Gradients (bf16) | 590 MB | 590 MB |
| Activations (batch=24/gpu) | ~4.8 GB | ~4.8 GB |
| **Total** | **~8.4 GB / 16 GB** ✅ | **~6.0 GB / 16 GB** ✅ |

### Training Time Estimate

```
500K samples × 512 tokens × 3 epochs = 768M training tokens
FLOPs ≈ 6 × 295M × 768M = 1.36 × 10¹⁸
2×T4 effective throughput ≈ 65 TFLOPS
Time ≈ 1.36e18 / 65e12 ≈ 20,900 seconds ≈ 5.8 hours
```

---

## Proposed Changes

### Project Structure at `e:\mistral`

```
e:\mistral/
├── components/
│   ├── __init__.py
│   ├── activations.py       # SiLU (for SwiGLU), GELU, ReLU²
│   ├── attention.py          # GQA with QK-norm + RoPE
│   ├── block.py              # Pre-norm transformer block
│   ├── feedforward.py        # SwiGLU FFN
│   ├── norms.py              # RMSNorm, LayerNorm (from scratch)
│   └── rope.py               # Rotary Position Embeddings
├── models/
│   ├── __init__.py
│   └── transformer.py        # Full causal LM
├── trainer/
│   ├── __init__.py
│   ├── trainer.py            # Training loop (AMP, grad accum, LR schedule)
│   └── callbacks.py          # Logger, Checkpoint, EarlyStopping, Plotter
├── utils/
│   ├── __init__.py
│   ├── data.py               # HF dataset loading + tokenization
│   ├── seed.py               # Reproducibility
│   └── config.py             # Dataclass-based config
├── tests/
│   ├── __init__.py
│   ├── test_environment.py   # 🆕 Library availability + version checks
│   └── test_model.py         # Model architecture + training tests
├── configs/
│   └── default.yaml          # Default 295M training config
├── train.py                  # 🚀 Single entrypoint for Kaggle
├── requirements.txt          # Dependencies
└── README.md                 # Setup + Kaggle instructions
```

---

### Components Layer

#### [NEW] [activations.py](file:///e:/mistral/components/activations.py)
- `SiLU`, `GELU`, `ReLUSquared` — each as `nn.Module`, from scratch using `torch.nn.functional`
- Adapted from [activations.py](file:///e:/transformer/components/activations.py)

#### [NEW] [norms.py](file:///e:/mistral/components/norms.py)
- `RMSNorm` — from scratch: `weight * x * rsqrt(mean(x²) + eps)`, no bias, no mean subtraction
- `LayerNorm` — from scratch: mean/variance normalization with learnable scale and bias
- Factory function `build_norm(norm_type, dim)` — clean signature, no config object
- Adapted from [norms.py](file:///e:/transformer/components/norms.py)

#### [NEW] [rope.py](file:///e:/mistral/components/rope.py)
- `build_rope_cache(seq_len, head_dim, base=10000)` — precomputes cos/sin frequency matrices
- `apply_rope(x, cos, sin)` — rotary embeddings on Q/K tensors
- Adapted from [rope.py](file:///e:/transformer/components/rope.py)

#### [NEW] [attention.py](file:///e:/mistral/components/attention.py)
- `GroupedQueryAttention` — configurable `num_heads` / `num_kv_heads`
- QK-normalization with per-head RMSNorm (toggleable via config)
- `F.scaled_dot_product_attention(is_causal=True)` for fused kernels on T4
- Attention entropy tracking (no-grad side computation)
- `repeat_kv()` for GQA key/value expansion
- Adapted from [attention.py](file:///e:/transformer/components/attention.py) with **clean dataclass config** instead of getattr gymnastics

#### [NEW] [feedforward.py](file:///e:/mistral/components/feedforward.py)
- `SwiGLUFFN` — `w_down(silu(w_gate(x)) * w_up(x))`, 3 linear projections, no bias
- Configurable `ffn_dim` (default: 4× hidden_dim)
- Adapted from [feedforward.py](file:///e:/transformer/components/feedforward.py)

#### [NEW] [block.py](file:///e:/mistral/components/block.py)
- `TransformerBlock` — Pre-norm: `x + attn(norm(x))`, `x + ffn(norm(x))`
- RMSNorm before both attention and FFN
- Hidden state RMS tracking for diagnostics
- Adapted from [block.py](file:///e:/transformer/components/block.py)

---

### Model Layer

#### [NEW] [transformer.py](file:///e:/mistral/models/transformer.py)
- `TransformerLM` class — full causal LM:
  - `nn.Embedding` → 16 × `TransformerBlock` → `RMSNorm` → `nn.Linear` (lm_head)
  - Tied word embeddings (lm_head shares weights with embedding)
  - Dynamic RoPE cache per forward pass
  - Weight init: normal (std=0.02) with residual scaling `1/√(2×num_layers)` for output projections
- Clean `__init__` using a dataclass config
- Adapted from [mini_qwen.py](file:///e:/transformer/models/mini_qwen.py) scaled up to 295M

---

### Trainer Layer

#### [NEW] [trainer.py](file:///e:/mistral/trainer/trainer.py)
- `Trainer` with:
  - Mixed precision via `torch.amp.autocast` (bf16)
  - Gradient accumulation (4 steps)
  - Cosine LR with linear warmup (500 steps)
  - Gradient clipping (max_norm=1.0)
  - Callback system
  - **Auto DataParallel** — wraps model when `torch.cuda.device_count() > 1`
  - Validation loop with `@torch.no_grad()`
- Adapted from [trainer.py](file:///e:/transformer/trainer/trainer.py) with DataParallel support

#### [NEW] [callbacks.py](file:///e:/mistral/trainer/callbacks.py)
- `Callback` base class
- `LoggerCallback` — stdout + CSV + JSON history
- `CheckpointCallback` — saves best model on val loss improvement
- `EarlyStoppingCallback` — stops on val loss plateau
- `PlotterCallback` (**NEW**) — generates matplotlib plots at training end:
  - Training loss curve
  - Validation loss + perplexity
  - Learning rate schedule
  - Combined 2×2 dashboard (saved as PNG)
- Adapted from [callbacks.py](file:///e:/transformer/trainer/callbacks.py) with PlotterCallback added

---

### Utils Layer

#### [NEW] [data.py](file:///e:/mistral/utils/data.py)
- `TextDataset` — loads HF dataset, tokenizes, chunks into `(x, y)` causal LM pairs
- `get_dataloaders()` — returns train + val DataLoaders
- Supports TinyStories, WikiText, FineWeb (auto-detects text column)
- Handles missing validation splits (slices 10% from train)
- GPT-2 tokenizer (50257 vocab), `num_workers=2, pin_memory=True`
- Adapted from [data.py](file:///e:/transformer/utils/data.py)

#### [NEW] [seed.py](file:///e:/mistral/utils/seed.py)
- `set_seed(seed)` — Python, NumPy, PyTorch, CUDA seeds + deterministic flags
- From [seed.py](file:///e:/transformer/utils/seed.py)

#### [NEW] [config.py](file:///e:/mistral/utils/config.py)
- `ModelConfig` dataclass — all model + training hyperparams with defaults for the 295M config
- `load_config(yaml_path)` → `ModelConfig`
- Eliminates the `getattr`/`isinstance(config, dict)` anti-pattern from the existing codebase

---

### Training Entrypoint

#### [NEW] [train.py](file:///e:/mistral/train.py)
Single-file entrypoint:
1. Parses CLI args (config path, output dir, seed, `--quick_test` flag)
2. Loads YAML config → `ModelConfig` dataclass
3. Sets seed
4. Loads HF dataset + tokenizer (syncs vocab_size)
5. Builds model, wraps in `DataParallel` if multi-GPU
6. Creates fused AdamW optimizer
7. Registers callbacks (Logger, Checkpoint, EarlyStopping, Plotter)
8. Runs `Trainer.train()`
9. Prints final summary (best val loss, perplexity, runtime, tokens/sec)

**`--quick_test` mode**: Overrides config to `max_samples=200, epochs=1, hidden_dim=64, num_layers=2` for a 30-second pipeline smoke test.

**Kaggle usage:**
```bash
!git clone https://github.com/<your-repo>/mistral.git
%cd mistral
!pip install -r requirements.txt
!python -m pytest tests/test_environment.py -v  # Verify environment first
!python -m pytest tests/test_model.py -v         # Verify model architecture
!python train.py --config configs/default.yaml --output_dir /kaggle/working/results
```

---

### Config

#### [NEW] [default.yaml](file:///e:/mistral/configs/default.yaml)
```yaml
# Model — 295M parameters
hidden_dim: 1024
num_layers: 16
num_heads: 16
num_kv_heads: 4
max_seq_len: 512
vocab_size: 50257
ffn_dim: 4096
tie_word_embeddings: true
norm_type: rmsnorm
activation_type: swiglu
qk_norm: true
use_rope: true
dropout: 0.0

# Training
lr: 6.0e-4
min_lr: 6.0e-5
weight_decay: 0.01
warmup_steps: 500
batch_size: 48
gradient_accumulation_steps: 4
epochs: 3
val_interval: 500
clip_grad: 1.0
precision: bf16

# Data
dataset_name: roneneldan/TinyStories
dataset_config: null
tokenizer_name: gpt2
max_samples: 500000
val_max_samples: 5000

# System
seed: 42
early_stopping_patience: 5
output_dir: results
```

---

### Tests

#### [NEW] [test_environment.py](file:///e:/mistral/tests/test_environment.py) — 🆕 Library & Environment Smoke Tests

Run these **first** on Kaggle before anything else. They verify the runtime is sane.

| Test | What it verifies |
|------|-----------------|
| `test_python_version` | Python ≥ 3.9 |
| `test_torch_installed` | `torch` importable |
| `test_torch_version` | PyTorch ≥ 2.0.0 |
| `test_cuda_available` | `torch.cuda.is_available()` is True |
| `test_cuda_device_count` | At least 1 GPU, logs actual count |
| `test_cuda_device_name` | Logs GPU name(s) (confirms T4) |
| `test_cuda_memory` | Each GPU has ≥ 14GB free memory |
| `test_bf16_support` | `torch.cuda.is_bf16_supported()` is True |
| `test_sdpa_available` | `F.scaled_dot_product_attention` exists (PyTorch 2.0+) |
| `test_flash_attention_backend` | Checks if flash/math/efficient SDPA backends available |
| `test_transformers_installed` | `transformers` importable |
| `test_transformers_version` | transformers ≥ 4.30.0 |
| `test_datasets_installed` | `datasets` importable |
| `test_datasets_version` | datasets ≥ 2.14.0 |
| `test_tokenizers_installed` | `tokenizers` importable |
| `test_yaml_installed` | `yaml` (PyYAML) importable |
| `test_matplotlib_installed` | `matplotlib` importable |
| `test_numpy_installed` | `numpy` importable |
| `test_numpy_version` | numpy ≥ 1.24.0 |
| `test_disk_space` | Output directory has ≥ 1GB free space |
| `test_dataparallel_works` | Wrapping a small model in `nn.DataParallel` + forward pass succeeds |

#### [NEW] [test_model.py](file:///e:/mistral/tests/test_model.py) — Architecture & Training Tests

| Test | What it verifies |
|------|-----------------|
| `test_rmsnorm_output_shape_and_scale` | Output shape correct, RMS ≈ 1.0 |
| `test_rmsnorm_vs_pytorch` | Custom RMSNorm matches `torch.nn.RMSNorm` if available |
| `test_layernorm_output_shape_and_stats` | Mean ≈ 0, variance ≈ 1 |
| `test_swiglu_ffn_shape` | SwiGLU preserves input shape |
| `test_swiglu_gate_mechanism` | Gate projection actually gates (not just pass-through) |
| `test_rope_cache_shape` | cos/sin have `(seq_len, head_dim)` shape |
| `test_rope_equivariance` | Shifted positions produce shifted embeddings |
| `test_gqa_attention_shape` | Output shape correct with 4:1 GQA ratio |
| `test_gqa_kv_repeat` | `repeat_kv` correctly expands KV heads |
| `test_qk_norm_toggle` | QK-norm modules present/absent based on config |
| `test_transformer_block_shape` | Block preserves shape, tracks RMS diagnostics |
| `test_full_model_forward` | `(B, T)` → `(B, T, V)` logits, correct shapes |
| `test_full_model_backward` | All parameters receive gradients |
| `test_training_step_updates_weights` | One optimizer step changes weights |
| `test_model_parameter_count` | Params in 280M–310M range |
| `test_causal_masking` | Future tokens don't leak into past logits |
| `test_tied_embeddings` | `lm_head.weight is embed.weight` |
| `test_config_from_yaml` | YAML → dataclass round-trips correctly |
| `test_seed_reproducibility` | Same seed → bitwise identical outputs |
| `test_mixed_precision_forward` | bf16 autocast forward pass works without errors |
| `test_gradient_accumulation` | Accumulated grads match single large-batch grads |

---

### Dependencies

#### [NEW] [requirements.txt](file:///e:/mistral/requirements.txt)
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tokenizers>=0.13.0
pyyaml>=6.0
matplotlib>=3.7.0
numpy>=1.24.0
pytest>=7.0.0
```

---

## Verification Plan

### Automated Tests (run in order)
```bash
# 1. Environment & library checks (run FIRST on Kaggle)
python -m pytest tests/test_environment.py -v --tb=short

# 2. Model architecture & training tests
python -m pytest tests/test_model.py -v --tb=short

# 3. Quick end-to-end smoke test (CPU, tiny model, ~30 seconds)
python train.py --config configs/default.yaml --output_dir test_output --quick_test
```

### Manual Verification (Kaggle)
1. `git push` from local → `git pull` on Kaggle
2. Run `test_environment.py` — confirms CUDA, T4 GPUs, bf16, library versions
3. Run `test_model.py` — confirms model builds and trains correctly
4. Run full training: `python train.py --config configs/default.yaml --output_dir /kaggle/working/results`
5. Verify:
   - Loss decreases, LR schedule follows cosine warmup
   - Both T4 GPUs utilized (DataParallel)
   - Plots generated in output directory (loss curves, LR schedule, dashboard)
   - No OOM errors (bf16 + batch=48 fits in 2×16GB)
   - Checkpoint saved at best val loss
   - Training completes within ~6 hours
