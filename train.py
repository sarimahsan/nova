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
    print(f"Setting up AdamW optimizer (fused={use_fused})...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
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

if __name__ == "__main__":
    main()
