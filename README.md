# Nova: Modern Transformer from Scratch (~141M parameters)

A complete, self-contained implementation of a modern transformer language model built from scratch using PyTorch. Nova features:
* **RMSNorm** and **LayerNorm** (implemented from scratch)
* **SwiGLU**, **GELU**, and **ReLU²** feedforward activations (implemented from scratch)
* **Rotary Position Embeddings (RoPE)** (implemented from scratch)
* **Grouped Query Attention (GQA)** with optional **QK-Normalization** and attention entropy tracking
* Tied word embeddings (shared weights between embeddings and language model head)
* Custom callback-driven training loop supporting automatic mixed precision (AMP) and DataParallel multi-GPU scaling
* Custom metric visualizer compiling a 2×2 dashboard of metrics at the end of training

---

## 📊 Model Architecture Diagram

```mermaid
graph TD
    Input[Token IDs: Batch, Seq] --> Embed[Embedding Layer]
    Embed --> Block1[Transformer Block 1]
    
    subgraph Block ["Transformer Block Layer (Pre-Norm)"]
        direction TB
        B_In[Input Hidden States] --> B_RMS1[RMSNorm]
        B_RMS1 --> B_GQA[Grouped Query Attention]
        B_GQA --> B_Add1[Residual Connection +]
        B_In --> B_Add1
        
        B_Add1 --> B_RMS2[RMSNorm]
        B_RMS2 --> B_FFN[SwiGLU FFN]
        B_FFN --> B_Add2[Residual Connection +]
        B_Add1 --> B_Add2
        B_Add2 --> B_Out[Output Hidden States]
    end
    
    Block1 --> BlockN[Transformer Block N]
    BlockN --> FinalNorm[Final RMSNorm]
    FinalNorm --> LMHead[LM Head]
    LMHead --> Output[Logits: Batch, Seq, Vocab]
    
    style Block fill:#f9f9f9,stroke:#333,stroke-width:1px
    style B_GQA fill:#d1e7dd,stroke:#0f5132,stroke-width:1px
    style B_FFN fill:#fff3cd,stroke:#664d03,stroke-width:1px
```

---

## 📂 Project Structure

```
.
├── components/
│   ├── activations.py      # SiLU (for SwiGLU), GELU, ReLU²
│   ├── attention.py        # GQA with QK-norm + RoPE
│   ├── block.py            # Pre-norm transformer block
│   ├── feedforward.py      # SwiGLU FFN
│   ├── norms.py            # RMSNorm, LayerNorm (from scratch)
│   └── rope.py             # Rotary Position Embeddings
├── models/
│   └── transformer.py      # Full causal LM
├── trainer/
│   ├── trainer.py          # Causal language model training loop
│   └── callbacks.py        # Logger, Checkpoint, EarlyStopping, Plotter
├── utils/
│   ├── data.py             # HF dataset loading + tokenization
│   ├── seed.py             # Reproducibility
│   └── config.py           # Dataclass-based config
├── tests/
│   ├── test_environment.py # Smoke tests for packages and hardware
│   └── test_model.py       # Comprehensive unit tests for model layers
├── configs/
│   └── default.yaml        # ~141M parameter model default configuration
├── train.py                # Single training entrypoint
└── requirements.txt        # Package dependencies
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/nova.git
cd nova

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Verification & Unit Testing

Always run tests to verify that your current hardware, versions, and layers are functioning correctly:

```bash
# 1. Run environment and library version smoke checks
python -m pytest tests/test_environment.py -v

# 2. Run model architecture and training process tests
python -m pytest tests/test_model.py -v

# 3. Run a quick local CPU pipeline test (takes ~30 seconds)
python train.py --config configs/default.yaml --output_dir test_output --quick_test
```

---

## 🚀 Training on Kaggle (2×T4 GPUs)

Under Kaggle notebook environment with **2×T4 GPUs** accelerator option enabled, execute the following commands in a code cell:

```bash
# 1. Clone your repository
!git clone https://github.com/<your-username>/nova.git
%cd nova

# 2. Install requirements
!pip install -r requirements.txt

# 3. Verify environment is configured correctly
!python -m pytest tests/test_environment.py -v
!python -m pytest tests/test_model.py -v

# 4. Start training on 2×T4 GPUs (DataParallel + bf16 autocast)
!python train.py --config configs/default.yaml --output_dir /kaggle/working/results
```

### Output Artifacts
Once training finishes, the directory specified in `--output_dir` will contain:
1. `checkpoint.pt`: Best model weights based on validation loss.
2. `metrics.csv`: Step-by-step log of training loss, learning rate, GPU memory, perplexity, and throughput.
3. `history.json`: Log file with history details.
4. `training_dashboard.png`: Training metrics dashboard containing loss, validation loss, perplexity, and learning rate curves.
