import torch
import torch.nn as nn
from components.attention import GroupedQueryAttention
from components.norms import build_norm
from components.feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, norm_type="rmsnorm", activation_type="swiglu", qk_norm=True, use_rope=True, ffn_dim=None, dropout=0.0):
        super().__init__()
        self.attn_norm = build_norm(norm_type, hidden_dim)
        self.attn = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            use_rope=use_rope,
            dropout=dropout
        )

        self.ffn_norm = build_norm(norm_type, hidden_dim)
        self.ffn = FeedForward(
            activation_type=activation_type,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        # Diagnostic attributes to track hidden state magnitudes (RMS)
        self.last_rms_attn_in = 0.0
        self.last_rms_attn_out = 0.0
        self.last_rms_ffn_in = 0.0
        self.last_rms_ffn_out = 0.0

    def forward(self, x, cos, sin):
        # x: (batch, seq, hidden_dim)
        
        # Calculate RMS before attention norm
        with torch.no_grad():
            self.last_rms_attn_in = torch.sqrt(x.pow(2).mean()).item()

        # Attention block
        residual = x
        x_normed = self.attn_norm(x)
        
        # Calculate RMS after attention norm
        with torch.no_grad():
            self.last_rms_attn_out = torch.sqrt(x_normed.pow(2).mean()).item()
            
        attn_out = self.attn(x_normed, cos, sin)
        x = residual + attn_out

        # Calculate RMS before FFN norm
        with torch.no_grad():
            self.last_rms_ffn_in = torch.sqrt(x.pow(2).mean()).item()

        # FFN block
        residual = x
        x_normed = self.ffn_norm(x)

        # Calculate RMS after FFN norm
        with torch.no_grad():
            self.last_rms_ffn_out = torch.sqrt(x_normed.pow(2).mean()).item()

        ffn_out = self.ffn(x_normed)
        x = residual + ffn_out

        return x
