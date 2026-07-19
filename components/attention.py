import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from components.rope import apply_rope
from components.norms import RMSNorm

def repeat_kv(x, n_rep):
    """
    Repeats keys/values for Grouped Query Attention (GQA).
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    x = x[:, :, None, :, :]            # (b, h, 1, s, d)
    x = x.expand(b, h, n_rep, s, d)    # repeat
    x = x.reshape(b, h * n_rep, s, d)
    return x

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, qk_norm=True, use_rope=True, dropout=0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)

        self.qk_norm = qk_norm
        self.use_rope = use_rope

        # QK norm using learnable RMSNorm per head if enabled
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Attribute to track entropy for logging callbacks
        self.last_attn_entropy = 0.0

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for multi-head
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        if self.use_rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Repeat KV for GQA
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Attention entropy calculation is disabled to prevent OutOfMemory (OOM) errors on large batch sizes
        self.last_attn_entropy = 0.0

        # PyTorch native optimized causal SDPA
        # dropout_p is only applied during training by SDPA internally
        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(b, s, self.hidden_dim)
        return self.resid_dropout(self.out_proj(out))
