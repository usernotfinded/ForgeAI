"""
ForgeAI Training Loop
=====================
Hardware-adaptive training with mixed precision, gradient accumulation,
checkpointing, and validation perplexity tracking.
"""

from __future__ import annotations

from .planner import estimate_training, TrainingPlan
from .trainer import train, TrainConfig, TrainMetrics
from .scheduler import cosine_warmup_scheduler

__all__ = [
    "estimate_training",
    "TrainingPlan",
    "train",
    "TrainConfig",
    "TrainMetrics",
    "cosine_warmup_scheduler",
]
