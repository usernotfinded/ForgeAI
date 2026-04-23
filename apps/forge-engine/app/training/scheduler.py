"""
ForgeAI — Learning Rate Scheduler
===================================

Cosine annealing with linear warmup — the standard LR schedule for
transformer pre-training (used by GPT-3, LLaMA, Chinchilla, etc.).

Schedule:
  1. Linear warmup from 0 to max_lr over warmup_steps
  2. Cosine decay from max_lr to min_lr over remaining steps
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a cosine annealing scheduler with linear warmup.

    Args:
        optimizer:      The optimizer to schedule
        warmup_steps:   Steps for linear warmup (typically 1-5% of total)
        total_steps:    Total training steps
        min_lr_ratio:   Final LR as a fraction of peak LR (default 0.1 = 10%)

    Returns:
        A LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
