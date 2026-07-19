import torch
import torch.nn as nn
from components.activations import GELU, ReLUSquared, SiLU

class GELUFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.0):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)
        self.w_in = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act = GELU()
        self.dropout = nn.Dropout(dropout)
        self.w_out = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_out(self.dropout(self.act(self.w_in(x))))

class ReLU2FFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.0):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)
        self.w_in = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act = ReLUSquared()
        self.dropout = nn.Dropout(dropout)
        self.w_out = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_out(self.dropout(self.act(self.w_in(x))))

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.0):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 8 // 3) # SwiGLU hidden dim is typically scaled differently, but we default to config-driven ffn_dim or 4/3 hidden_dim
            # But let's use the explicit ffn_dim or hidden_dim * 4 to preserve comparability.
            ffn_dim = int(hidden_dim * 4) if ffn_dim is None else ffn_dim
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w_down(self.dropout(self.act(self.w_gate(x)) * self.w_up(x)))

class FeedForward(nn.Module):
    """
    Unified FeedForward wrapper class.
    """
    def __init__(self, activation_type, hidden_dim, ffn_dim=None, dropout=0.0):
        super().__init__()
        act_type_lower = activation_type.lower()
        if act_type_lower == "swiglu":
            self.net = SwiGLUFFN(hidden_dim, ffn_dim, dropout=dropout)
        elif act_type_lower == "gelu":
            self.net = GELUFFN(hidden_dim, ffn_dim, dropout=dropout)
        elif act_type_lower == "relu2":
            self.net = ReLU2FFN(hidden_dim, ffn_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}")

    def forward(self, x):
        return self.net(x)

def build_ffn(activation_type, hidden_dim, ffn_dim=None, dropout=0.0):
    """
    Factory function to build the correct FeedForward module type.
    """
    return FeedForward(activation_type, hidden_dim, ffn_dim, dropout=dropout)

