import os
import csv
import json
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from components.norms import RMSNorm, LayerNorm
from components.attention import GroupedQueryAttention

class Callback:
    """
    Base class for training callbacks.
    """
    def on_train_start(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_start(self, trainer): pass
    def on_epoch_end(self, trainer): pass
    def on_step_start(self, trainer): pass
    def on_step_end(self, trainer, loss_val): pass
    def on_validation_start(self, trainer): pass
    def on_validation_end(self, trainer, val_loss): pass


class LoggerCallback(Callback):
    """
    Logs metrics per step to stdout, CSV, and JSON.
    """
    def __init__(self, output_dir, log_interval=10):
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.csv_path = os.path.join(output_dir, "metrics.csv")
        self.history_path = os.path.join(output_dir, "history.json")
        self.history = []
        self.step_start_time = None

    def on_train_start(self, trainer):
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "step", "train_loss", "val_loss", "perplexity",
                    "learning_rate", "tokens_per_second", "gpu_memory_mb", "elapsed_time"
                ])

    def on_step_start(self, trainer):
        self.step_start_time = time.time()

    def on_step_end(self, trainer, loss_val):
        if trainer.global_step % self.log_interval == 0 or trainer.global_step == 1 or trainer.global_step == trainer.total_steps:
            step_duration = time.time() - self.step_start_time if self.step_start_time else 0.0
            
            # Compute tokens per second
            tokens_processed = trainer.config.batch_size * trainer.config.max_seq_len
            tokens_per_sec = tokens_processed / step_duration if step_duration > 0 else 0.0
            
            # Compute GPU memory
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            
            # Current LR
            lr = trainer.get_current_lr()
            
            # Perplexity
            perplexity = math.exp(min(loss_val, 100))
            
            elapsed = time.time() - trainer.train_start_time if trainer.train_start_time else 0.0

            # Log to stdout
            print(
                f"Epoch {trainer.epoch} | Step {trainer.global_step} | "
                f"Loss {loss_val:.4f} | PPL {perplexity:.2f} | LR {lr:.2e} | "
                f"Tokens/s {tokens_per_sec:.1f} | GPU Mem {gpu_mem:.1f}MB"
            )

            # Record
            metrics = {
                "epoch": trainer.epoch,
                "step": trainer.global_step,
                "train_loss": loss_val,
                "val_loss": getattr(trainer, "last_val_loss", None),
                "perplexity": perplexity,
                "learning_rate": lr,
                "tokens_per_second": tokens_per_sec,
                "gpu_memory_mb": gpu_mem,
                "elapsed_time": elapsed
            }
            self.history.append(metrics)
            
            # Write to CSV
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics["epoch"], metrics["step"], f"{metrics['train_loss']:.6f}",
                    f"{metrics['val_loss']:.6f}" if metrics["val_loss"] is not None else "",
                    f"{metrics['perplexity']:.4f}", f"{metrics['learning_rate']:.2e}",
                    f"{metrics['tokens_per_second']:.2f}", f"{metrics['gpu_memory_mb']:.2f}",
                    f"{metrics['elapsed_time']:.2f}"
                ])
                
            # Save history JSON
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)


class CheckpointCallback(Callback):
    """
    Saves model checkpoints when validation loss improves, and logs model statistics.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
        self.stats_path = os.path.join(output_dir, "model_statistics.json")
        self.best_val_loss = float("inf")
        self.prev_param_weights = {}

    def on_train_start(self, trainer):
        self._capture_weights(trainer.model)

    def _capture_weights(self, model):
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        self.prev_param_weights = {
            name: p.clone().detach().cpu()
            for name, p in raw_model.named_parameters()
            if p.requires_grad
        }

    def on_validation_end(self, trainer, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            # 1. Save checkpoint
            checkpoint = {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "epoch": trainer.epoch,
                "global_step": trainer.global_step,
                "val_loss": val_loss
            }
            torch.save(checkpoint, self.checkpoint_path)
            print(f"--> Best model checkpoint saved to '{self.checkpoint_path}' (Val Loss: {val_loss:.4f})")
            
            # 2. Compute Model Statistics at checkpoint
            stats = self._compute_model_statistics(trainer)
            with open(self.stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"--> Saved checkpoint model statistics to '{self.stats_path}'")
            
            # Capture current weights as baseline for the next checkpoint
            self._capture_weights(trainer.model)

    def _compute_model_statistics(self, trainer):
        raw_model = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
        stats = {}
        
        # Parameter Norm
        param_norms = {}
        total_param_norm = 0.0
        for name, p in raw_model.named_parameters():
            if p.requires_grad:
                norm = p.norm().item()
                param_norms[name] = norm
                total_param_norm += norm ** 2
        total_param_norm = math.sqrt(total_param_norm)
        
        # Weight Update Ratio: ||W_t - W_{t_prev}|| / ||W_{t_prev}||
        update_ratios = {}
        avg_update_ratio = 0.0
        count = 0
        for name, p in raw_model.named_parameters():
            if p.requires_grad and name in self.prev_param_weights:
                curr_w = p.detach().cpu()
                prev_w = self.prev_param_weights[name]
                diff_norm = (curr_w - prev_w).norm().item()
                prev_norm = prev_w.norm().item()
                ratio = diff_norm / (prev_norm + 1e-8)
                update_ratios[name] = ratio
                avg_update_ratio += ratio
                count += 1
        avg_update_ratio = avg_update_ratio / count if count > 0 else 0.0
        
        # Gradient Norm
        grad_norms = {}
        total_grad_norm = 0.0
        for name, p in raw_model.named_parameters():
            if p.requires_grad and p.grad is not None:
                norm = p.grad.norm().item()
                grad_norms[name] = norm
                total_grad_norm += norm ** 2
        total_grad_norm = math.sqrt(total_grad_norm)
        
        # Activation Norm proxy (RMS across blocks)
        act_rms = []
        for i, block in enumerate(raw_model.blocks):
            act_rms.append({
                f"layer_{i}_rms_attn_in": block.last_rms_attn_in,
                f"layer_{i}_rms_attn_out": block.last_rms_attn_out,
                f"layer_{i}_rms_ffn_in": block.last_rms_ffn_in,
                f"layer_{i}_rms_ffn_out": block.last_rms_ffn_out
            })

        stats = {
            "step": trainer.global_step,
            "val_loss": self.best_val_loss,
            "total_parameter_norm": total_param_norm,
            "average_weight_update_ratio": avg_update_ratio,
            "total_gradient_norm": total_grad_norm,
            "parameter_norms": param_norms,
            "weight_update_ratios": update_ratios,
            "gradient_norms": grad_norms,
            "block_activation_rms": act_rms
        }
        return stats


class DiagnosticCallback(Callback):
    """
    Periodically hooks into activations and logs gradient statistics,
    attention entropy, and hidden state norms.
    """
    def __init__(self, output_dir, log_interval=100):
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.grad_csv_path = os.path.join(output_dir, "gradients.csv")
        self.act_csv_path = os.path.join(output_dir, "activations.csv")
        self.hooks = []
        self.activation_stats = {}
        self.pre_update_weights = {}

    def on_train_start(self, trainer):
        os.makedirs(self.output_dir, exist_ok=True)
        # Setup headers
        if not os.path.exists(self.grad_csv_path):
            with open(self.grad_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "layer_name", "grad_norm", "grad_mean", "grad_std",
                    "grad_max", "grad_min", "grad_rms", "grad_variance", "grad_gsnr",
                    "weight_update_ratio"
                ])
                
        if not os.path.exists(self.act_csv_path):
            with open(self.act_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "layer_name", "act_rms", "act_mean", "act_std",
                    "act_max", "act_min", "act_near_zero", "act_saturated"
                ])

        raw_model = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
        self._register_activation_hooks(raw_model)

    def _register_activation_hooks(self, model):
        from components.activations import GELU, ReLUSquared, SiLU
        
        def make_hook(name):
            def hook(module, input, output):
                if not self.should_log:
                    return
                if isinstance(output, tuple):
                    output = output[0]
                if not isinstance(output, torch.Tensor):
                    return
                
                t = output.detach().float()
                rms = torch.sqrt(t.pow(2).mean()).item()
                mean = t.mean().item()
                std = t.std().item()
                maximum = t.max().item()
                minimum = t.min().item()
                
                near_zero = ""
                saturated = ""
                
                if isinstance(module, (GELU, ReLUSquared, SiLU)):
                    near_zero = (t.abs() < 0.01).float().mean().item()
                    if isinstance(module, ReLUSquared):
                        saturated = (t <= 0.0).float().mean().item()
                    else:
                        if len(input) > 0 and isinstance(input[0], torch.Tensor):
                            x_in = input[0].detach().float()
                            saturated = (x_in < -3.0).float().mean().item()
                
                self.activation_stats[name] = {
                    "rms": rms,
                    "mean": mean,
                    "std": std,
                    "max": maximum,
                    "min": minimum,
                    "near_zero": near_zero,
                    "saturated": saturated
                }
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, RMSNorm, LayerNorm, GELU, ReLUSquared, SiLU)):
                if any(x in name for x in ["blocks"]):
                    self.hooks.append(module.register_forward_hook(make_hook(name)))

    def on_step_start(self, trainer):
        self.should_log = (trainer.global_step % self.log_interval == 0 or trainer.global_step == trainer.total_steps)
        self.activation_stats.clear()
        
        if self.should_log:
            raw_model = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
            self.pre_update_weights = {
                name: p.clone().detach()
                for name, p in raw_model.named_parameters()
                if p.requires_grad
            }

    def on_step_end(self, trainer, loss_val):
        if not self.should_log:
            return

        raw_model = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
        
        # 1. Log gradient and weight update stats
        with open(self.grad_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for name, p in raw_model.named_parameters():
                if p.requires_grad:
                    update_ratio = 0.0
                    if name in self.pre_update_weights:
                        old_w = self.pre_update_weights[name]
                        diff_norm = (p.detach() - old_w).norm().item()
                        old_norm = old_w.norm().item()
                        update_ratio = diff_norm / (old_norm + 1e-8)
                        
                    grad_norm, grad_mean, grad_std, grad_max, grad_min = 0.0, 0.0, 0.0, 0.0, 0.0
                    grad_rms, grad_variance, grad_gsnr = 0.0, 0.0, 0.0
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        grad_norm = g.norm().item()
                        grad_mean = g.mean().item()
                        grad_std = g.std().item()
                        grad_max = g.max().item()
                        grad_min = g.min().item()
                        grad_rms = torch.sqrt(g.pow(2).mean()).item()
                        grad_variance = g.var().item()
                        grad_gsnr = (grad_mean ** 2) / (grad_variance + 1e-8)
                        
                    writer.writerow([
                        trainer.global_step, name, f"{grad_norm:.6f}", f"{grad_mean:.6f}",
                        f"{grad_std:.6f}", f"{grad_max:.6f}", f"{grad_min:.6f}",
                        f"{grad_rms:.6f}", f"{grad_variance:.6f}", f"{grad_gsnr:.6e}",
                        f"{update_ratio:.6e}"
                    ])

        # 2. Log activation stats
        with open(self.act_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for name, stats in self.activation_stats.items():
                writer.writerow([
                    trainer.global_step, name, f"{stats['rms']:.6f}", f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}", f"{stats['max']:.6f}", f"{stats['min']:.6f}",
                    f"{stats['near_zero']}" if stats['near_zero'] != "" else "",
                    f"{stats['saturated']}" if stats['saturated'] != "" else ""
                ])
                
            for i, block in enumerate(raw_model.blocks):
                writer.writerow([trainer.global_step, f"blocks.{i}.rms_attn_in", f"{block.last_rms_attn_in:.6f}", "", "", "", "", "", ""])
                writer.writerow([trainer.global_step, f"blocks.{i}.rms_attn_out", f"{block.last_rms_attn_out:.6f}", "", "", "", "", "", ""])
                writer.writerow([trainer.global_step, f"blocks.{i}.rms_ffn_in", f"{block.last_rms_ffn_in:.6f}", "", "", "", "", "", ""])
                writer.writerow([trainer.global_step, f"blocks.{i}.rms_ffn_out", f"{block.last_rms_ffn_out:.6f}", "", "", "", "", "", ""])
                writer.writerow([trainer.global_step, f"blocks.{i}.attn.entropy", f"{block.attn.last_attn_entropy:.6f}", "", "", "", "", "", ""])
                
                if hasattr(raw_model, "last_cosine_similarities") and i < len(raw_model.last_cosine_similarities):
                    cos_sim = raw_model.last_cosine_similarities[i]
                    writer.writerow([trainer.global_step, f"blocks.{i}.cosine_similarity", f"{cos_sim:.6f}", "", "", "", "", "", ""])

        self.pre_update_weights.clear()

    def on_train_end(self, trainer):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class EarlyStoppingCallback(Callback):
    """
    Triggers early stopping if validation loss fails to improve for `patience` intervals.
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.epochs_no_improve = 0

    def on_validation_end(self, trainer, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"\n[Early Stopping] Triggered! Validation loss did not improve for {self.patience} validation intervals.")
                trainer.stop_training = True


class PlotterCallback(Callback):
    """
    Generates training curves and metrics dashboards at the end of training.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_train_end(self, trainer):
        history_path = os.path.join(self.output_dir, "history.json")
        if not os.path.exists(history_path):
            print(f"[PlotterCallback] History file not found at {history_path}. Skipping plot generation.")
            return

        try:
            with open(history_path, "r") as f:
                history = json.load(f)

            if not history:
                print("[PlotterCallback] History file is empty. Skipping plot generation.")
                return

            steps = [h["step"] for h in history]
            train_losses = [h["train_loss"] for h in history]
            lrs = [h["learning_rate"] for h in history]

            val_history = trainer.validation_history
            val_steps = [v["step"] for v in val_history]
            val_losses = [v["val_loss"] for v in val_history]
            val_ppls = [v["perplexity"] for v in val_history]

            # Set up matplotlib style
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Training Loss
            axs[0, 0].plot(steps, train_losses, label="Train Loss", color="#1f77b4", linewidth=1.5)
            axs[0, 0].set_title("Training Loss Curve", fontsize=14, fontweight="bold")
            axs[0, 0].set_xlabel("Steps", fontsize=12)
            axs[0, 0].set_ylabel("Loss", fontsize=12)
            axs[0, 0].grid(True, linestyle="--", alpha=0.6)
            axs[0, 0].legend()

            # 2. Learning Rate
            axs[0, 1].plot(steps, lrs, label="Learning Rate", color="#ff7f0e", linewidth=1.5)
            axs[0, 1].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            axs[0, 1].set_xlabel("Steps", fontsize=12)
            axs[0, 1].set_ylabel("LR", fontsize=12)
            axs[0, 1].grid(True, linestyle="--", alpha=0.6)
            axs[0, 1].legend()

            # 3. Validation Loss
            if val_steps:
                axs[1, 0].plot(val_steps, val_losses, label="Val Loss", color="#d62728", marker="o", linewidth=1.5)
                axs[1, 0].set_title("Validation Loss Curve", fontsize=14, fontweight="bold")
                axs[1, 0].set_xlabel("Steps", fontsize=12)
                axs[1, 0].set_ylabel("Loss", fontsize=12)
                axs[1, 0].grid(True, linestyle="--", alpha=0.6)
                axs[1, 0].legend()
            else:
                axs[1, 0].text(0.5, 0.5, "No Validation Data Recorded", ha="center", va="center", fontsize=12)

            # 4. Validation Perplexity
            if val_steps:
                axs[1, 1].plot(val_steps, val_ppls, label="Perplexity", color="#2ca02c", marker="s", linewidth=1.5)
                axs[1, 1].set_title("Validation Perplexity Curve", fontsize=14, fontweight="bold")
                axs[1, 1].set_xlabel("Steps", fontsize=12)
                axs[1, 1].set_ylabel("Perplexity", fontsize=12)
                axs[1, 1].grid(True, linestyle="--", alpha=0.6)
                axs[1, 1].legend()
            else:
                axs[1, 1].text(0.5, 0.5, "No Validation Data Recorded", ha="center", va="center", fontsize=12)

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "training_dashboard.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"[PlotterCallback] Training dashboard successfully saved to '{plot_path}'")

        except Exception as e:
            print(f"[PlotterCallback] Failed to generate plots: {e}")
