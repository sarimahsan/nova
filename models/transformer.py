import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components.block import TransformerBlock
from components.norms import build_norm
from components.rope import build_rope_cache

class TransformerLM(nn.Module):
    def __init__(self, config):
        """
        config must be a config object (e.g. ModelConfig) containing:
        - hidden_dim
        - max_seq_len
        - vocab_size
        - num_layers
        - num_heads
        - num_kv_heads
        - tie_word_embeddings
        - norm_type
        - activation_type
        - qk_norm
        - use_rope
        - ffn_dim
        """
        super().__init__()
        self.config = config
        
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len
        vocab_size = config.vocab_size
        num_layers = config.num_layers
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads
        tie_word_embeddings = config.tie_word_embeddings

        self.embed = nn.Embedding(vocab_size, self.hidden_dim)
        self.embed_dropout = nn.Dropout(getattr(config, 'dropout', 0.0))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                norm_type=config.norm_type,
                activation_type=config.activation_type,
                qk_norm=config.qk_norm,
                use_rope=config.use_rope,
                ffn_dim=config.ffn_dim,
                dropout=getattr(config, 'dropout', 0.0)
            )
            for _ in range(num_layers)
        ])

        self.norm = build_norm(config.norm_type, self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, vocab_size, bias=False)

        if tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.last_cosine_similarities = []

        # Initialize weights
        self.apply_initialization()

    def apply_initialization(self):
        initializer_range = 0.02
        num_layers = len(self.blocks)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
                # Special scaling for residual output projections (e.g. out_proj, w_down, w_out)
                if any(proj in name for proj in ["out_proj", "w_down", "w_out"]):
                    with torch.no_grad():
                        module.weight.data.copy_(module.weight.data / math.sqrt(2.0 * num_layers))
                        
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                
            elif hasattr(module, "weight") and module.weight is not None:
                # Custom Norm classes and PyTorch norm layers
                if "norm" in name or isinstance(module, (nn.LayerNorm, nn.modules.normalization.LayerNorm)):
                    nn.init.ones_(module.weight)
                    if hasattr(module, "bias") and module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: (batch, seq)
        """
        # Determine dynamic autocast precision
        precision = getattr(self.config, "precision", "fp32")
        amp_dtype = torch.float32
        if precision == "fp16":
            amp_dtype = torch.float16
        elif precision == "bf16":
            amp_dtype = torch.bfloat16
            
        device_type = "cuda" if x.is_cuda else "cpu"
        amp_enabled = (precision in ["fp16", "bf16"] and device_type == "cuda")

        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            b, s = x.shape
            x = self.embed(x)
            x = self.embed_dropout(x)

            # Dynamic RoPE Cache
            cos, sin = build_rope_cache(s, self.blocks[0].attn.head_dim, device=x.device)

            # Blocks forward pass
            cosine_sims = []
            for block in self.blocks:
                h_next = block(x, cos, sin)
                with torch.no_grad():
                    cos_val = F.cosine_similarity(x, h_next, dim=-1).mean().item()
                    cosine_sims.append(cos_val)
                x = h_next
            self.last_cosine_similarities = cosine_sims

            x = self.norm(x)
            logits = self.lm_head(x)

            return logits
