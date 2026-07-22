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

This repository implements the **Nova** architecture, a modern decoder-only transformer. Below are the visual diagrams of the global architecture pipeline and the inner details of each Transformer block.

### 1. Global Model Pipeline

```mermaid
graph TD
    %% Custom Styling
    classDef default fill:#1e1e2e,stroke:#cdd6f4,stroke-width:1px,color:#cdd6f4;
    classDef input fill:#f9e2af,stroke:#fab387,stroke-width:2px,color:#11111b;
    classDef block fill:#313244,stroke:#bac2de,stroke-dasharray: 5 5,color:#cdd6f4;
    classDef attention fill:#a6e3a1,stroke:#94e2d5,stroke-width:2px,color:#11111b;
    classDef ffn fill:#b4befe,stroke:#cba6f7,stroke-width:2px,color:#11111b;
    classDef layer fill:#45475a,stroke:#585b70,stroke-width:1px,color:#cdd6f4;
    classDef output fill:#f38ba8,stroke:#eba0ac,stroke-width:2px,color:#11111b;

    %% Nodes
    Input["Token IDs<br>Batch, Seq"]:::input --> Embed["Embedding Layer<br>Vocab Size -> Hidden Dim"]:::layer
    Embed --> Drop["Embedding Dropout"]:::layer
    Drop --> Block1["Transformer Block 1"]:::block
    Block1 --> BlockDots[ ... ]
    BlockDots --> BlockN["Transformer Block N"]:::block
    
    %% Dynamic RoPE Cache
    RoPECache["Dynamic RoPE Cache<br>cos, sin"]:::layer -.-> |Used by GQA| Block1
    RoPECache -.-> |Used by GQA| BlockN
    
    BlockN --> FinalNorm["Final Norm<br>RMSNorm / LayerNorm"]:::layer
    FinalNorm --> LMHead["LM Head<br>Linear: Hidden Dim -> Vocab"]:::layer
    LMHead --> Logits["Logits<br>Batch, Seq, Vocab"]:::output
    
    %% Tied Weights Link
    Embed -.->|Shared Weight Tying| LMHead
```

### 2. Transformer Block Internals (Pre-Norm)

```mermaid
graph TB
    %% Custom Styling
    classDef default fill:#1e1e2e,stroke:#cdd6f4,color:#cdd6f4;
    classDef residual fill:#f5c2e7,stroke:#cba6f7,stroke-width:2px,color:#11111b;
    classDef norm fill:#89b4fa,stroke:#74c7ec,stroke-width:1px,color:#11111b;
    classDef proj fill:#fab387,stroke:#f9e2af,stroke-width:1px,color:#11111b;
    classDef op fill:#94e2d5,stroke:#a6e3a1,stroke-width:1px,color:#11111b;
    classDef routing fill:#b4befe,stroke:#89b4fa,stroke-width:1px,color:#11111b;

    B_In[Input Hidden States]:::default
    
    %% Attention Branch
    B_In --> B_AttnNorm["Norm Layer<br>RMSNorm / LayerNorm"]:::norm
    
    subgraph GQA ["Grouped Query Attention (GQA) Sub-block"]
        direction TB
        B_AttnNorm --> Q_Proj["Query Projection<br>num_heads * head_dim"]:::proj
        B_AttnNorm --> K_Proj["Key Projection<br>num_kv_heads * head_dim"]:::proj
        B_AttnNorm --> V_Proj["Value Projection<br>num_kv_heads * head_dim"]:::proj
        
        Q_Proj --> Q_Norm["Q Norm<br>RMSNorm"]:::norm
        K_Proj --> K_Norm["K Norm<br>RMSNorm"]:::norm
        
        Q_Norm --> Q_RoPE["Apply RoPE<br>Rotary Embeddings"]:::op
        K_Norm --> K_RoPE["Apply RoPE<br>Rotary Embeddings"]:::op
        
        K_RoPE --> K_Rep["Repeat KV<br>n_rep = num_heads / num_kv_heads"]:::op
        V_Proj --> V_Rep["Repeat KV<br>n_rep = num_heads / num_kv_heads"]:::op
        
        Q_RoPE --> SDPA["Scaled Dot Product Attention<br>Causal Masking & Dropout"]:::op
        K_Rep --> SDPA
        V_Rep --> SDPA
        
        SDPA --> Attn_OutProj["Out Projection<br>Linear"]:::proj
        Attn_OutProj --> Attn_Drop["Residual Dropout"]:::op
    end
    
    %% Residual Connection 1
    B_In --> B_Add1((+)):::residual
    Attn_Drop --> B_Add1
    
    %% FFN Branch
    B_Add1 --> B_FFNNorm["Norm Layer<br>RMSNorm / LayerNorm"]:::norm
    
    subgraph FFN ["Feed-Forward Network (Gated / SwiGLU Default)"]
        direction TB
        B_FFNNorm --> Gate_Proj["Gate Proj<br>w_gate"]:::proj
        B_FFNNorm --> Up_Proj["Up Proj<br>w_up"]:::proj
        
        Gate_Proj --> Act["Activation<br>SiLU / GELU / ReLU²"]:::op
        
        Act --> Mul["Element-wise Product<br>act(gate) * up"]:::op
        Up_Proj --> Mul
        
        Mul --> FFN_Drop["FFN Dropout"]:::op
        FFN_Drop --> Down_Proj["Down Proj<br>w_down"]:::proj
    end
    
    %% Residual Connection 2
    B_Add1 --> B_Add2((+)):::residual
    Down_Proj --> B_Add2
    
    B_Add2 --> B_Out[Output Hidden States]:::default
```

---

## 🤗 Hugging Face Model

The trained weights and model card for **Nova 141M** are published on Hugging Face:  
👉 **[sarimahsan/nova-141m-tinystories](https://huggingface.co/sarimahsan/nova-141m-tinystories)**


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
git clone https://github.com/sarimahsan/nova.git
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
# 1. Clone repository
!git clone https://github.com/sarimahsan/nova.git
%cd nova

# 2. Install requirements
!pip install -r requirements.txt

# 3. Verify environment is configured correctly
!python -m pytest tests/test_environment.py -v
!python -m pytest tests/test_model.py -v

# 4. Start training on 2×T4 GPUs (DataParallel + bf16 autocast)
!python train.py --config configs/default.yaml --output_dir /kaggle/working/results
```