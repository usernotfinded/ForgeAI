"""Tiny optional MLX smoke path.

The smoke intentionally avoids ForgeAI datasets, checkpoints, pretrained models,
and the stable PyTorch trainer. It is only a Phase 0/1 backend foundation check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.experimental.mlx_backend.availability import (
    MlxAvailability,
    check_availability,
    require_attr,
    require_mlx,
)


@dataclass(frozen=True)
class MlxSmokeResult:
    availability: MlxAvailability
    tensor_shape: tuple[int, ...]
    tensor_sum: float
    forward_shape: tuple[int, ...]
    loss: float | None
    train_step_ran: bool


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    return tuple(int(dim) for dim in shape)


def _to_float(value: Any) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


def run_smoke(*, train_step: bool = True) -> MlxSmokeResult:
    """Run a tiny deterministic MLX smoke test with no downloads."""
    availability = check_availability()
    mx, nn, optimizers = require_mlx()

    random = getattr(mx, "random", None)
    seed = getattr(random, "seed", None)
    if callable(seed):
        seed(0)

    array = require_attr(mx, "array")
    sum_fn = require_attr(mx, "sum")
    mean_fn = require_attr(mx, "mean")
    eval_fn = require_attr(mx, "eval")
    linear_cls = require_attr(nn, "Linear")

    x = array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    tiny_op = sum_fn((x + 1.0) * 0.5)
    eval_fn(tiny_op)

    model = linear_cls(4, 2)
    outputs = model(x)
    eval_fn(outputs)

    loss_value: float | None = None
    if train_step:
        target = array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        sgd_cls = require_attr(optimizers, "SGD")
        value_and_grad = require_attr(nn, "value_and_grad")
        optimizer = sgd_cls(learning_rate=0.01)

        def loss_fn(model_arg: Any, inputs: Any, expected: Any) -> Any:
            prediction = model_arg(inputs)
            diff = prediction - expected
            return mean_fn(diff * diff)

        loss_and_grad = value_and_grad(model, loss_fn)
        loss, gradients = loss_and_grad(model, x, target)
        optimizer.update(model, gradients)

        parameters = model.parameters()
        optimizer_state = getattr(optimizer, "state", None)
        if optimizer_state is None:
            eval_fn(parameters)
        else:
            eval_fn(parameters, optimizer_state)
        loss_value = _to_float(loss)

    return MlxSmokeResult(
        availability=availability,
        tensor_shape=_shape_tuple(x),
        tensor_sum=_to_float(tiny_op),
        forward_shape=_shape_tuple(outputs),
        loss=loss_value,
        train_step_ran=train_step,
    )

