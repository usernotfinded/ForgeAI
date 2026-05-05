"""
ForgeAI — Checkpoint Manager
==============================

Save, load, and resume model checkpoints with full training state.

Each checkpoint includes:
  - Model weights (state_dict)
  - Optimizer state
  - Scheduler state
  - Training metadata (step, epoch, loss, config)
  - A 'latest' symlink for easy resume

Checkpoint directory structure:
    checkpoints/my-run/
    ├── step_001000/
    │   ├── model.pt
    │   ├── optimizer.pt
    │   ├── scheduler.pt
    │   └── metadata.json
    ├── step_002000/
    │   └── ...
    └── latest -> step_002000
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint."""
    step: int
    epoch: int
    loss: float
    val_loss: Optional[float]
    learning_rate: float
    total_tokens_seen: int
    model_config: dict[str, Any]
    architecture: str
    backend: str
    dtype: str


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metadata: CheckpointMetadata,
) -> Path:
    """
    Save a training checkpoint.

    Creates a directory named step_XXXXXX containing model weights,
    optimizer state, scheduler state, and metadata. Updates the
    'latest' symlink.
    """
    checkpoint_dir = Path(checkpoint_dir)
    step_dir = checkpoint_dir / f"step_{metadata.step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), step_dir / "model.pt")

    # Save optimizer state
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")

    # Save scheduler state
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")

    # Save metadata
    meta_dict = asdict(metadata)
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=2)

    # Update 'latest' symlink
    latest_link = checkpoint_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(step_dir.name)

    return step_dir


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: str = "cpu",
) -> CheckpointMetadata:
    """
    Load a checkpoint into model (and optionally optimizer/scheduler).

    Args:
        checkpoint_path: Path to checkpoint dir (e.g. checkpoints/run/step_001000)
                         or parent dir with 'latest' symlink
        model:           Model to load weights into
        optimizer:       Optimizer to restore state (optional, for resume)
        scheduler:       Scheduler to restore state (optional, for resume)
        device:          Device to map tensors to

    Returns:
        CheckpointMetadata from the loaded checkpoint
    """
    path = Path(checkpoint_path)

    # Resolve 'latest' symlink
    if (path / "latest").exists():
        path = (path / "latest").resolve()
    elif not (path / "model.pt").exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Expected a directory with model.pt or a parent with 'latest' symlink."
        )

    # Load model weights
    model_path = path / "model.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Load optimizer state
    if optimizer is not None:
        opt_path = path / "optimizer.pt"
        if opt_path.exists():
            opt_state = torch.load(opt_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(opt_state)

    # Load scheduler state
    if scheduler is not None:
        sched_path = path / "scheduler.pt"
        if sched_path.exists():
            sched_state = torch.load(sched_path, map_location=device, weights_only=True)
            scheduler.load_state_dict(sched_state)

    # Load metadata
    meta_path = path / "metadata.json"
    with open(meta_path) as f:
        meta_dict = json.load(f)

    return CheckpointMetadata(**meta_dict)


def list_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    """List all checkpoints in a directory, sorted by step."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    for d in sorted(checkpoint_dir.iterdir()):
        if d.is_dir() and d.name.startswith("step_"):
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["path"] = str(d)
                checkpoints.append(meta)

    return checkpoints


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Return the path to the latest checkpoint, or None if no checkpoints exist."""
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest"
    if latest.exists():
        return latest.resolve()

    # Fall back to highest step number
    step_dirs = sorted(
        (d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    return step_dirs[-1] if step_dirs else None
