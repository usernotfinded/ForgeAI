"""Tiny optional Keras smoke path.

This module must not import keras at module import time. Keras reads
KERAS_BACKEND during import, so run_smoke configures the environment first.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any


DEFAULT_KERAS_BACKEND = "torch"
INSTALL_HINT = 'pip install -e "apps/forge-engine[keras]"'


class KerasUnavailableError(RuntimeError):
    """Raised when the optional Keras integration cannot be used."""


@dataclass(frozen=True)
class KerasAvailability:
    keras_installed: bool
    keras_hub_installed: bool
    backend_env: str | None
    message: str


@dataclass(frozen=True)
class KerasSmokeResult:
    backend: str
    keras_version: str
    keras_hub_available: bool
    output_shape: tuple[int, ...]
    train_loss: float | None


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_availability() -> KerasAvailability:
    """Return optional Keras/KerasHub availability without importing either package."""
    keras_installed = _module_available("keras")
    keras_hub_installed = _module_available("keras_hub")
    backend_env = os.environ.get("KERAS_BACKEND")

    if not keras_installed:
        message = f"Keras is not installed. Install optional dependencies with: {INSTALL_HINT}"
    elif not keras_hub_installed:
        message = "Keras is installed. KerasHub is not installed; smoke does not require it."
    else:
        message = "Keras and KerasHub are installed."

    return KerasAvailability(
        keras_installed=keras_installed,
        keras_hub_installed=keras_hub_installed,
        backend_env=backend_env,
        message=message,
    )


def _import_keras(backend: str = DEFAULT_KERAS_BACKEND) -> ModuleType:
    availability = check_availability()
    if not availability.keras_installed:
        raise KerasUnavailableError(availability.message)

    if "keras" not in sys.modules:
        os.environ["KERAS_BACKEND"] = backend

    try:
        return importlib.import_module("keras")
    except ImportError as exc:
        raise KerasUnavailableError(
            f"Keras could not be imported with backend '{backend}'. "
            f"Install optional dependencies with: {INSTALL_HINT}"
        ) from exc


def _keras_backend_name(keras: ModuleType) -> str:
    backend_module = getattr(keras, "backend", None)
    if backend_module is not None and hasattr(backend_module, "backend"):
        return str(backend_module.backend())

    config_module = getattr(keras, "config", None)
    if config_module is not None and hasattr(config_module, "backend"):
        return str(config_module.backend())

    return os.environ.get("KERAS_BACKEND", DEFAULT_KERAS_BACKEND)


def _as_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return _as_float(value[0])
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def run_smoke(
    *,
    backend: str = DEFAULT_KERAS_BACKEND,
    train_step: bool = True,
) -> KerasSmokeResult:
    """Run a tiny in-memory Keras smoke test with no downloads or KerasHub models."""
    keras = _import_keras(backend=backend)
    np = importlib.import_module("numpy")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(2),
        ],
        name="forgeai_experimental_keras_smoke",
    )

    x = np.ones((2, 4), dtype="float32")
    y = np.array([0, 1], dtype="int32")

    outputs = model(x, training=False)
    output_shape = tuple(int(dim) for dim in outputs.shape)

    loss_value: float | None = None
    if train_step:
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        loss_value = _as_float(model.train_on_batch(x, y))

    return KerasSmokeResult(
        backend=_keras_backend_name(keras),
        keras_version=str(getattr(keras, "__version__", "unknown")),
        keras_hub_available=check_availability().keras_hub_installed,
        output_shape=output_shape,
        train_loss=loss_value,
    )
