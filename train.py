import os
import argparse
import time
import math
import torch
import yaml

from utils.config import load_config, ModelConfig
from utils.seed import set_seed
from utils.data import get_dataloaders
from models.transformer import TransformerLM
from trainer.trainer import Trainer
from trainer.callbacks import (
    LoggerCallback,
    CheckpointCallback,
    DiagnosticCallback,
    EarlyStoppingCallback,
    PlotterCallback
)

def parse_args():
    parser = argparse.ArgumentParser(description="Modern Transformer Training Entrypoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints and metrics")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--quick_test", action="store_true", help="Run a small, fast subset sweep on CPU to test the pipeline")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config path '{args.config}' not found. Using default configurations.")
        config = ModelConfig()

    # 2. Apply command-line overrides
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed

    # 3. Apply Quick Test overrides if requested
    if args.quick_test:
        print("\n>>> Applying QUICK TEST overrides for rapid execution...")
        config.max_samples = 200
        config.val_max_samples = 50
        config.epochs = 1
        config.val_interval = 2
        config.hidden_dim = 64
        config.num_layers = 2
        config.num_heads = 4
        config.num_kv_heads = 1
        config.warmup_steps = 2
        config.batch_size = 4
        config.gradient_accumulation_steps = 1
        config.precision = "fp32"

    print(f"\n==========================================")
    print(f"Initializing Transformer training pipeline")
    print(f"Output directory: {config.output_dir}")
    print(f"Seed: {config.seed}")
    print(f"Model parameters: {config.hidden_dim} dim | {config.num_layers} layers | {config.num_heads} heads | {config.norm_type} norm | {config.activation_type} activation")
    print(f"==========================================")

    # 4. Set reproducibility seed
    set_seed(config.seed)

    # 5. Load dataloaders and tokenizer
    print("Loading Hugging Face dataset...")
    train_loader, val_loader, tokenizer = get_dataloaders(config)

    # Sync vocabulary size
    config.vocab_size = len(tokenizer)
    print(f"Tokenizer loaded. Vocabulary size: {config.vocab_size}")

    # 6. Instantiate model
    print("Building model...")
    model = TransformerLM(config)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built. Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # 7. Setup optimizer
    use_fused = (torch.cuda.is_available() and config.precision != "fp32")
    
    # Separate parameters into decay and no_decay groups to stabilize training
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D parameters (like LayerNorm/RMSNorm scale weights) should not be decayed
        if param.ndim < 2 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    print(f"Setting up AdamW optimizer (fused={use_fused}) with {len(decay_params)} decayed and {len(no_decay_params)} non-decayed parameters...")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.lr,
        betas=(0.9, 0.95),
        fused=use_fused
    )

    # 8. Set up callbacks
    callbacks = [
        LoggerCallback(config.output_dir, log_interval=2 if args.quick_test else 10),
        CheckpointCallback(config.output_dir),
        DiagnosticCallback(config.output_dir, log_interval=5 if args.quick_test else 100),
        EarlyStoppingCallback(patience=config.early_stopping_patience),
        PlotterCallback(config.output_dir)
    ]

    # 9. Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        callbacks=callbacks
    )

    # 10. Start training
    print("Starting training loop...")
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    print(f"\n==========================================")
    print(f"Training completed in {duration/60:.2f} minutes.")
    if trainer.last_val_loss is not None:
        print(f"Best validation loss: {trainer.last_val_loss:.4f}")
        print(f"Perplexity: {math.exp(min(trainer.last_val_loss, 100)):.2f}")
    print(f"Results and plots saved to: {config.output_dir}")
    print(f"==========================================")

    # 11. Post-training: load best model and export to HF format
    print("\nPreparing model and metrics for deployment...")
    
    # Paths for output folders
    hf_model_dir = os.path.join(config.output_dir, "model")
    metrics_dir = os.path.join(config.output_dir, "metrics")
    os.makedirs(hf_model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load best checkpoint if it exists, otherwise use the current weights in memory
    best_checkpoint_path = os.path.join(config.output_dir, "checkpoint.pt")
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from '{best_checkpoint_path}'...")
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        # Load weights into raw_model (unwrapped from DataParallel if necessary)
        raw_model = model.module if hasattr(model, "module") else model
        state_dict = checkpoint["model_state_dict"]
        # Strip DataParallel prefix if present
        from collections import OrderedDict
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            clean_state_dict[name] = v
        raw_model.load_state_dict(clean_state_dict)
    else:
        print("No checkpoint file found. Exporting the final weights in memory.")

    # Save to Safetensors format
    raw_model = model.module if hasattr(model, "module") else model
    try:
        from safetensors.torch import save_file
        safetensors_path = os.path.join(hf_model_dir, "model.safetensors")
        save_file(raw_model.state_dict(), safetensors_path)
        print(f"Saved model weights to Safetensors format at: '{safetensors_path}'")
    except ImportError:
        print("safetensors package is not installed. Saving standard PyTorch model.bin instead...")
        bin_path = os.path.join(hf_model_dir, "model.bin")
        torch.save(raw_model.state_dict(), bin_path)
        print(f"Saved model weights to standard PyTorch format at: '{bin_path}'")

    # Save config.json
    import json
    from dataclasses import asdict
    config_json_path = os.path.join(hf_model_dir, "config.json")
    with open(config_json_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved configuration to: '{config_json_path}'")

    # Save tokenizer files
    tokenizer.save_pretrained(hf_model_dir)
    print(f"Saved tokenizer files to: '{hf_model_dir}'")

    # 12. Organize metric files (CSVs, JSON logs, dashboard plots)
    metrics_files = [
        "metrics.csv", 
        "history.json", 
        "training_dashboard.png", 
        "model_statistics.json", 
        "loss_curve.png", 
        "learning_rate.png"
    ]
    import shutil
    moved_count = 0
    for fname in metrics_files:
        src_path = os.path.join(config.output_dir, fname)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(metrics_dir, fname))
            moved_count += 1
            
    print(f"Moved {moved_count} CSV/metrics/plot files to: '{metrics_dir}'")

if __name__ == "__main__":
    main()
