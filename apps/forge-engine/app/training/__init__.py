"""
ForgeAI Training Loop
=====================
Hardware-adaptive training with mixed precision, gradient accumulation,
checkpointing, and validation perplexity tracking.
"""

from __future__ import annotations

from typing import Any, Callable

from .planner import estimate_training, TrainingPlan

_missing_torch_error: ModuleNotFoundError | None = None
_runtime_train: Callable[..., Any]
_runtime_scheduler: Callable[..., Any]
_runtime_train_config: type[Any]
_runtime_train_metrics: type[Any]


def _missing_torch(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise ModuleNotFoundError(
        "PyTorch dependencies are required for ForgeAI training runtime components."
    ) from _missing_torch_error

try:
    from .trainer import train as _runtime_train
    from .trainer import TrainConfig as _runtime_train_config
    from .trainer import TrainMetrics as _runtime_train_metrics
    from .scheduler import cosine_warmup_scheduler as _runtime_scheduler
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime deps
    _missing_torch_error = exc

    _runtime_train = _missing_torch
    _runtime_scheduler = _missing_torch

    class _MissingTrainConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_torch(*args, **kwargs)

    class _MissingTrainMetrics:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_torch(*args, **kwargs)

    _runtime_train_config = _MissingTrainConfig
    _runtime_train_metrics = _MissingTrainMetrics

train = _runtime_train
cosine_warmup_scheduler = _runtime_scheduler
TrainConfig = _runtime_train_config
TrainMetrics = _runtime_train_metrics

__all__ = [
    "estimate_training",
    "TrainingPlan",
    "train",
    "TrainConfig",
    "TrainMetrics",
    "cosine_warmup_scheduler",
]
