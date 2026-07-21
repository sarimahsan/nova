import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) from scratch.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        # Compute variance in float32 for numerical stability, but apply the scaling
        # in the input's native precision to save activation memory during backprop
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm + self.eps).type_as(x))

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (LayerNorm) implemented from scratch for full transparency.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

def build_norm(norm_type, dim, eps=1e-6):
    """
    Factory function to build a normalization layer.
    """
    norm_type_lower = norm_type.lower()
    if norm_type_lower == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type_lower == "layernorm":
        return LayerNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
