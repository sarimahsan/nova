import torch
import torch.nn as nn
import math
import time

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callbacks = callbacks or []

        self.epoch = 0
        self.global_step = 0
        self.stop_training = False
        self.train_start_time = None
        self.last_val_loss = None
        self.total_tokens_seen = 0
        self.validation_history = []

        # Precision options: "fp32", "fp16", "bf16"
        self.precision = getattr(config, "precision", "fp32")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = "cuda" if "cuda" in self.device else "cpu"

        # GradScaler is required for fp16 mixed precision to avoid underflow
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.precision == "fp16" and self.device_type == "cuda"))

        # Optimization & Scheduling Parameters
        self.max_lr = getattr(config, "lr", 6e-4)
        self.min_lr = getattr(config, "min_lr", 6e-5)
        self.warmup_steps = getattr(config, "warmup_steps", 500)
        self.clip_grad = getattr(config, "clip_grad", 1.0)
        self.grad_accum = getattr(config, "gradient_accumulation_steps", 4)
        self.epochs = getattr(config, "epochs", 3)
        self.val_interval = getattr(config, "val_interval", 500)

        # Total training steps (optimizer updates)
        self.total_steps = math.ceil(len(self.train_loader) / self.grad_accum) * self.epochs

        # Move model to device
        self.model.to(self.device)

        # Multi-GPU support using nn.DataParallel
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Detected {torch.cuda.device_count()} GPUs. Wrapping model in nn.DataParallel...")
            self.model = nn.DataParallel(self.model)

    def get_current_lr(self):
        """
        Cosine learning rate decay scheduler with linear warmup.
        """
        step = self.global_step
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.total_steps:
            return self.min_lr
            
        denom = self.total_steps - self.warmup_steps
        if denom <= 0:
            decay_ratio = 1.0
        else:
            decay_ratio = (step - self.warmup_steps) / denom
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def set_lr(self, lr):
        """
        Applies a new learning rate to the optimizer's parameter groups.
        """
        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train(self):
        """
        Runs the training and validation loops, invoking callbacks.
        """
        self.train_start_time = time.time()

        for callback in self.callbacks:
            callback.on_train_start(self)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.model.train()

            for callback in self.callbacks:
                callback.on_epoch_start(self)

            for step, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Track cumulative tokens seen
                self.total_tokens_seen += x.shape[0] * x.shape[1]

                # Initialize gradients, step-level metrics, update LR, and run callbacks
                if step % self.grad_accum == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step_loss_accumulator = 0.0
                    self.step_microbatch_count = 0
                    
                    # Update scheduler learning rate once per optimizer update
                    current_lr = self.get_current_lr()
                    self.set_lr(current_lr)
                    
                    for callback in self.callbacks:
                        callback.on_step_start(self)

                # Determine dynamic autocast precision
                amp_dtype = torch.float32
                if self.precision == "fp16":
                    amp_dtype = torch.float16
                elif self.precision == "bf16":
                    amp_dtype = torch.bfloat16
                    
                amp_enabled = (self.precision in ["fp16", "bf16"] and self.device_type == "cuda")

                # Mixed precision forward pass
                with torch.amp.autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = self.model(x)
                    B, T, V = logits.shape
                    raw_loss = criterion(logits.reshape(-1, V), y.reshape(-1))
                    loss = raw_loss / self.grad_accum

                self.step_loss_accumulator += raw_loss.item()
                self.step_microbatch_count += 1

                # Backward pass
                if self.device_type == "cuda" and self.precision == "fp16":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step optimizer after accumulating gradients
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                    if self.device_type == "cuda" and self.precision == "fp16":
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                        self.last_grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                        self.last_grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                        self.optimizer.step()

                    self.global_step += 1
                    loss_val = self.step_loss_accumulator / self.step_microbatch_count
                    
                    # Calculate parameter L2 norm
                    with torch.no_grad():
                        param_norm_sq = 0.0
                        for p in self.model.parameters():
                            if p.requires_grad:
                                param_norm_sq += p.norm().item() ** 2
                        self.last_param_norm = math.sqrt(param_norm_sq)

                    # Validation
                    if self.global_step % self.val_interval == 0 or self.global_step == self.total_steps:
                        val_loss = self.validate()
                        self.last_val_loss = val_loss

                        # Store in validation history
                        ppl = math.exp(min(val_loss, 100))
                        self.validation_history.append({
                            "epoch": self.epoch,
                            "step": self.global_step,
                            "val_loss": val_loss,
                            "perplexity": ppl
                        })

                        for callback in self.callbacks:
                            callback.on_validation_start(self)
                        for callback in self.callbacks:
                            callback.on_validation_end(self, val_loss)

                        self.model.train()

                    for callback in self.callbacks:
                        callback.on_step_end(self, loss_val)

                    if self.stop_training:
                        break

            for callback in self.callbacks:
                callback.on_epoch_end(self)

            if self.stop_training:
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

    @torch.no_grad()
    def validate(self):
        """
        Runs evaluation validation on the validation dataset.
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0

        amp_dtype = torch.float32
        if self.precision == "fp16":
            amp_dtype = torch.float16
        elif self.precision == "bf16":
            amp_dtype = torch.bfloat16
        amp_enabled = (self.precision in ["fp16", "bf16"] and self.device_type == "cuda")

        for val_x, val_y in self.val_loader:
            val_x = val_x.to(self.device)
            val_y = val_y.to(self.device)

            with torch.amp.autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                val_logits = self.model(val_x)
                B_v, T_v, V_v = val_logits.shape
                v_loss = criterion(val_logits.reshape(-1, V_v), val_y.reshape(-1))

            val_loss += v_loss.item()

        return val_loss / len(self.val_loader)
