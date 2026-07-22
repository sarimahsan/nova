import os
import sys
import pytest
import torch
import torch.nn as nn
import yaml

# Ensure components and models are in the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_rmsnorm_output_shape_and_scale():
    from components.norms import RMSNorm
    x = torch.randn(4, 16, 32)
    norm = RMSNorm(32)
    out = norm(x)
    assert out.shape == x.shape
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

def test_rmsnorm_vs_pytorch():
    from components.norms import RMSNorm
    x = torch.randn(2, 4, 16)
    norm = RMSNorm(16)
    if hasattr(nn, "RMSNorm"):
        ref = nn.RMSNorm(16)
        ref.weight.data.copy_(norm.weight.data)
        out = norm(x)
        ref_out = ref(x)
        assert torch.allclose(out, ref_out, atol=1e-5)
    else:
        # Fallback comparison against custom implementation correctness
        norm = RMSNorm(16)
        out = norm(x)
        assert out.shape == (2, 4, 16)

def test_layernorm_output_shape_and_stats():
    from components.norms import LayerNorm
    x = torch.randn(4, 16, 32) * 5 + 10.0
    norm = LayerNorm(32)
    out = norm(x)
    assert out.shape == x.shape
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

def test_swiglu_ffn_shape():
    from components.feedforward import FeedForward
    x = torch.randn(2, 8, 32)
    ffn = FeedForward("swiglu", hidden_dim=32, ffn_dim=64)
    out = ffn(x)
    assert out.shape == x.shape

def test_swiglu_gate_mechanism():
    from components.feedforward import SwiGLUFFN
    ffn = SwiGLUFFN(hidden_dim=32, ffn_dim=64)
    x = torch.randn(2, 8, 32)
    
    # Zeroing w_gate should lead to zero output because silu(0)*x = 0
    ffn.w_gate.weight.data.zero_()
    out = ffn(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

def test_rope_cache_shape():
    from components.rope import build_rope_cache
    cos, sin = build_rope_cache(128, 64)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)

def test_rope_equivariance():
    from components.rope import build_rope_cache, apply_rope
    cos, sin = build_rope_cache(32, 16)
    x = torch.randn(1, 1, 32, 16)
    out = apply_rope(x, cos, sin)
    
    # Shift position slice
    out_sliced = apply_rope(x[:, :, 5:6], cos[5:6], sin[5:6])
    assert torch.allclose(out[:, :, 5:6], out_sliced, atol=1e-5)

def test_gqa_attention_shape():
    from components.attention import GroupedQueryAttention
    x = torch.randn(2, 16, 64)
    # 64 dim, 4 heads, 2 KV heads
    attn = GroupedQueryAttention(hidden_dim=64, num_heads=4, num_kv_heads=2)
    cos = torch.randn(16, 16)
    sin = torch.randn(16, 16)
    out = attn(x, cos, sin)
    assert out.shape == x.shape

def test_gqa_kv_repeat():
    from components.attention import repeat_kv
    x = torch.randn(2, 2, 8, 16)
    out = repeat_kv(x, n_rep=3)
    assert out.shape == (2, 6, 8, 16)
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.equal(out[:, 1], x[:, 0])
    assert torch.equal(out[:, 2], x[:, 0])
    assert torch.equal(out[:, 3], x[:, 1])

def test_qk_norm_toggle():
    from components.attention import GroupedQueryAttention
    attn_with = GroupedQueryAttention(hidden_dim=32, num_heads=4, num_kv_heads=2, qk_norm=True)
    attn_without = GroupedQueryAttention(hidden_dim=32, num_heads=4, num_kv_heads=2, qk_norm=False)
    assert hasattr(attn_with, "q_norm")
    assert hasattr(attn_with, "k_norm")
    assert not hasattr(attn_without, "q_norm")
    assert not hasattr(attn_without, "k_norm")

def test_transformer_block_shape():
    from components.block import TransformerBlock
    x = torch.randn(2, 8, 32)
    cos = torch.randn(8, 8)
    sin = torch.randn(8, 8)
    block = TransformerBlock(hidden_dim=32, num_heads=4, num_kv_heads=2)
    out = block(x, cos, sin)
    assert out.shape == x.shape
    assert block.last_rms_attn_in > 0
    assert block.last_rms_attn_out > 0
    assert block.last_rms_ffn_in > 0
    assert block.last_rms_ffn_out > 0
    assert block.attn.last_attn_entropy >= 0

def test_full_model_forward():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16
    )
    model = TransformerLM(config)
    x = torch.randint(0, 100, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 100)

def test_full_model_backward():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16
    )
    model = TransformerLM(config)
    x = torch.randint(0, 100, (2, 8))
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"

def test_training_step_updates_weights():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16
    )
    model = TransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 100, (2, 8))
    
    p_before = model.embed.weight.clone().detach()
    
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    optimizer.step()
    
    p_after = model.embed.weight.detach()
    assert not torch.equal(p_before, p_after)

def test_model_parameter_count():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16
    )
    model = TransformerLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0

def test_causal_masking():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16
    )
    model = TransformerLM(config)
    model.eval()
    
    x1 = torch.randint(0, 100, (1, 8))
    x2 = x1.clone()
    x2[0, -1] = (x1[0, -1] + 1) % 100
    
    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)
        
    assert torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-5)

def test_tied_embeddings():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        tie_word_embeddings=True
    )
    model = TransformerLM(config)
    assert model.embed.weight is model.lm_head.weight

def test_config_from_yaml(tmp_path):
    from utils.config import load_config
    
    yaml_content = {
        "hidden_dim": 256,
        "num_layers": 8,
        "norm_type": "layernorm"
    }
    
    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f)
        
    config = load_config(str(yaml_file))
    assert config.hidden_dim == 256
    assert config.num_layers == 8
    assert config.norm_type == "layernorm"
    assert config.activation_type == "swiglu"

def test_seed_reproducibility():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    from utils.seed import set_seed
    
    config = ModelConfig(vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4, num_kv_heads=2)
    
    set_seed(42)
    model1 = TransformerLM(config)
    x = torch.randint(0, 100, (2, 8))
    out1 = model1(x)
    
    set_seed(42)
    model2 = TransformerLM(config)
    out2 = model2(x)
    
    assert torch.equal(out1, out2)

def test_mixed_precision_forward():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4, num_kv_heads=2)
    model = TransformerLM(config)
    x = torch.randint(0, 100, (2, 8))
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=True):
        logits = model(x)
        assert logits.shape == (2, 8, 100)

def test_gradient_accumulation():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    from utils.seed import set_seed
    
    config = ModelConfig(vocab_size=10, hidden_dim=16, num_layers=1, num_heads=4, num_kv_heads=1)
    
    set_seed(42)
    model1 = TransformerLM(config)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=1.0)
    
    set_seed(42)
    model2 = TransformerLM(config)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1.0)
    
    x = torch.randint(0, 10, (2, 4))
    
    # Standard single batch loss computation
    optimizer1.zero_grad()
    loss1 = model1(x).sum()
    loss1.backward()
    optimizer1.step()
    
    # Gradient accumulation computation
    optimizer2.zero_grad()
    for i in range(2):
        loss_part = model2(x[i:i+1]).sum()
        loss_part.backward()
        
    optimizer2.step()
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5)

def test_trainer_dataparallel():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        from utils.config import ModelConfig
        from models.transformer import TransformerLM
        from trainer.trainer import Trainer
        from torch.utils.data import TensorDataset, DataLoader
        
        config = ModelConfig(
            vocab_size=10,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            epochs=1,
            val_interval=2,
            gradient_accumulation_steps=1
        )
        model = TransformerLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        x = torch.randint(0, 10, (8, 4))
        y = torch.randint(0, 10, (8, 4))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            val_loader=loader,
            config=config,
            callbacks=[]
        )
        
        assert isinstance(trainer.model, nn.DataParallel)
        
        trainer.train()
        assert trainer.global_step > 0


def test_kv_cache_equivalence():
    from utils.config import ModelConfig
    from models.transformer import TransformerLM
    config = ModelConfig(vocab_size=100, hidden_dim=32, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=16)
    model = TransformerLM(config)
    model.eval()

    x = torch.randint(0, 100, (1, 8))

    # Full forward without KV cache
    with torch.no_grad():
        full_logits = model(x)

    # Step-by-step forward with KV cache
    with torch.no_grad():
        # Prefill prompt (first 5 tokens)
        prompt_logits, past_kv = model(x[:, :5], use_cache=True, start_pos=0)
        
        # Step through remaining tokens one by one
        curr_kv = past_kv
        all_step_logits = [prompt_logits]
        for pos in range(5, 8):
            step_logits, curr_kv = model(x[:, pos:pos+1], past_key_values=curr_kv, start_pos=pos, use_cache=True)
            all_step_logits.append(step_logits)

        cached_logits = torch.cat(all_step_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5), "KV cached output must match full forward output"

