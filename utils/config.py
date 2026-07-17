import yaml
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    # Model architecture
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 3
    max_seq_len: int = 512
    vocab_size: int = 50257
    ffn_dim: int = 3072
    tie_word_embeddings: bool = True
    norm_type: str = "rmsnorm"
    activation_type: str = "swiglu"
    qk_norm: bool = True
    use_rope: bool = True
    dropout: float = 0.0

    # Training parameters
    lr: float = 6.0e-4
    min_lr: float = 6.0e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    batch_size: int = 48
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    val_interval: int = 500
    clip_grad: float = 1.0
    precision: str = "bf16"

    # Data parameters
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: str = None
    tokenizer_name: str = "gpt2"
    max_samples: int = 500000
    val_max_samples: int = 5000

    # System parameters
    seed: int = 42
    early_stopping_patience: int = 5
    output_dir: str = "results"

def load_config(yaml_path: str) -> ModelConfig:
    """
    Loads config from a YAML file, filtering out keys that are not valid fields of ModelConfig.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    
    # Filter dictionary to keep only keys that match ModelConfig fields
    valid_keys = ModelConfig.__dataclass_fields__.keys()
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
    return ModelConfig(**filtered_data)
