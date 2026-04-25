"""
ForgeAI — Training Loop
=========================

Hardware-adaptive training loop for PyTorch (CUDA / MPS / CPU).

Features:
  - Mixed precision training (bfloat16 / float16 / float32)
  - Gradient accumulation for effective large batch sizes
  - Gradient clipping (max norm)
  - Cosine warmup LR schedule
  - Validation perplexity tracking
  - Periodic checkpoint saving
  - Gradient norm logging (catch instabilities early)
  - Real-time training metrics via JSON log file

MLX native training is planned but not yet implemented.
This trainer works on Apple Silicon via the MPS backend.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.checkpoints.manager import CheckpointMetadata, save_checkpoint, load_checkpoint
from app.training.scheduler import cosine_warmup_scheduler


@dataclass
class TrainConfig:
    """Configuration for a training run."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Batch / accumulation
    batch_size: int = 4
    grad_accumulation_steps: int = 1

    # Schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    min_lr_ratio: float = 0.1

    # Precision
    dtype: str = "bfloat16"  # bfloat16, float16, float32

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/run"
    save_every_steps: int = 1000

    # Validation
    val_every_steps: int = 200
    val_batches: int = 20  # number of batches for validation

    # Logging
    log_every_steps: int = 10
    log_file: str | None = None  # JSON lines log for web UI

    # Architecture info (for metadata)
    architecture: str = "transformer"
    backend: str = "cuda"

    # Gradient checkpointing (saves VRAM at cost of speed)
    gradient_checkpointing: bool = False


@dataclass
class TrainMetrics:
    """Accumulated metrics for a training step."""
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    step: int = 0
    epoch: int = 0
    total_tokens: int = 0


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_str, torch.float32)


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on transformer blocks."""
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()


def train(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: Optional[DataLoader[tuple[torch.Tensor, torch.Tensor]]],
    config: TrainConfig,
    model_config: dict[str, Any] | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """
    Run the training loop.

    Args:
        model:         The model to train
        train_loader:  Training DataLoader
        val_loader:    Validation DataLoader (optional)
        config:        Training configuration
        model_config:  Model config dict for checkpoint metadata
        resume_from:   Path to checkpoint to resume from

    Returns:
        Dict with final training metrics
    """
    device = config.backend
    if device == "mlx":
        # TODO(native-mlx-training): replace this fallback when MLX training backend is implemented.
        warnings.warn(
            "Backend 'mlx' requested, but native MLX training is not implemented yet. "
            "Falling back to CPU for this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        device = "cpu"

    model = model.to(device)
    dtype = _get_torch_dtype(config.dtype)

    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    # Optimizer: AdamW with weight decay on non-bias, non-norm parameters
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    # Learning rate scheduler
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        min_lr_ratio=config.min_lr_ratio,
    )

    # Mixed precision scaler (for float16 on CUDA)
    use_amp = dtype != torch.float32 and device != "cpu"
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and "cuda" in device))

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    total_tokens = 0
    if resume_from:
        meta = load_checkpoint(resume_from, model, optimizer, scheduler, device=device)
        start_step = meta.step
        start_epoch = meta.epoch
        total_tokens = meta.total_tokens_seen
        print(f"Resumed from step {start_step} (epoch {start_epoch})")

    # Logging
    log_file = None
    if config.log_file:
        Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(config.log_file, "a")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ────────────────────────────────────────────────────────

    if len(train_loader) == 0:
        raise ValueError(
            "Training DataLoader is empty. Check dataset size, context_length, batch_size, and split."
        )
    if val_loader is not None and len(val_loader) == 0:
        raise ValueError(
            "Validation DataLoader is empty. Reduce batch_size/val_split or use more data."
        )
    if config.max_steps <= start_step:
        raise ValueError(
            f"max_steps ({config.max_steps}) must be greater than resume step ({start_step})."
        )

    model.train()
    step = start_step
    epoch = start_epoch
    running_loss = 0.0
    best_val_loss = float("inf")
    accum_loss = 0.0

    train_iter = iter(train_loader)

    pbar = tqdm(
        total=config.max_steps - start_step,
        desc="Training",
        unit="step",
        initial=0,
    )

    t_start = time.time()
    tokens_this_interval = 0

    try:
        while step < config.max_steps:
            optimizer.zero_grad()
            accum_loss = 0.0

            for micro_step in range(config.grad_accumulation_steps):
                # Get next batch (restart epoch if exhausted)
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    epoch += 1
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(device)
                y = y.to(device)
                batch_tokens = x.numel()

                # Forward pass with mixed precision
                amp_device = "cuda" if "cuda" in device else "cpu"
                with torch.amp.autocast(device_type=amp_device, dtype=dtype, enabled=use_amp):
                    logits, loss = model(x, targets=y)
                    loss = loss / config.grad_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()
                accum_loss += loss.item()
                total_tokens += batch_tokens
                tokens_this_interval += batch_tokens

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            ).item()

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running_loss += accum_loss

            # ── Logging ──────────────────────────────────────────────────────

            if step % config.log_every_steps == 0:
                elapsed = time.time() - t_start
                tps = tokens_this_interval / max(elapsed, 1e-6)
                avg_loss = running_loss / config.log_every_steps
                lr = scheduler.get_last_lr()[0]

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tps:.0f}",
                    "gnorm": f"{grad_norm:.2f}",
                })
                pbar.update(config.log_every_steps)

                if log_file:
                    log_entry = {
                        "step": step,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": lr,
                        "tokens_per_second": tps,
                        "total_tokens": total_tokens,
                        "timestamp": time.time(),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()

                running_loss = 0.0
                tokens_this_interval = 0
                t_start = time.time()

            # ── Validation ───────────────────────────────────────────────────

            if val_loader is not None and step % config.val_every_steps == 0:
                val_loss = _validate(model, val_loader, config, device, dtype, use_amp)
                val_ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow

                tqdm.write(
                    f"  [val] step={step}  loss={val_loss:.4f}  ppl={val_ppl:.2f}"
                )

                if log_file:
                    log_entry = {
                        "step": step,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "timestamp": time.time(),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                model.train()

            # ── Checkpointing ────────────────────────────────────────────────

            if step % config.save_every_steps == 0:
                meta = CheckpointMetadata(
                    step=step,
                    epoch=epoch,
                    loss=accum_loss,
                    val_loss=best_val_loss if best_val_loss < float("inf") else None,
                    learning_rate=scheduler.get_last_lr()[0],
                    total_tokens_seen=total_tokens,
                    model_config=model_config or {},
                    architecture=config.architecture,
                    backend=config.backend,
                    dtype=config.dtype,
                )
                ckpt_path = save_checkpoint(
                    config.checkpoint_dir, model, optimizer, scheduler, meta
                )
                tqdm.write(f"  [ckpt] Saved checkpoint at step {step} → {ckpt_path}")

    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user. Saving checkpoint...")
        meta = CheckpointMetadata(
            step=step,
            epoch=epoch,
            loss=accum_loss if step > start_step else 0.0,
            val_loss=best_val_loss if best_val_loss < float("inf") else None,
            learning_rate=scheduler.get_last_lr()[0],
            total_tokens_seen=total_tokens,
            model_config=model_config or {},
            architecture=config.architecture,
            backend=config.backend,
            dtype=config.dtype,
        )
        save_checkpoint(config.checkpoint_dir, model, optimizer, scheduler, meta)
        tqdm.write("  Checkpoint saved.")

    finally:
        pbar.close()
        if log_file:
            log_file.close()

    return {
        "final_step": step,
        "final_epoch": epoch,
        "final_loss": accum_loss,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "total_tokens": total_tokens,
    }


def _validate(
    model: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig,
    device: str,
    dtype: torch.dtype,
    use_amp: bool,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.val_batches:
                break
            x = x.to(device)
            y = y.to(device)

            amp_device = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=amp_device, dtype=dtype, enabled=use_amp):
                _, loss = model(x, targets=y)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)
