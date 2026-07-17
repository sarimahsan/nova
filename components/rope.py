import torch

def build_rope_cache(seq_len, head_dim, base=10000, device="cpu"):
    """
    Returns cos and sin matrices for RoPE.
    Shape: (seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)

    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

    return cos, sin

def apply_rope(x, cos, sin):
    """
    x: (batch, heads, seq, head_dim)
    cos/sin: (seq, head_dim)
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out = torch.empty_like(x)

    out[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    out[..., 1::2] = x1 * sin[..., ::2] + x2 * cos[..., ::2]

    return out
