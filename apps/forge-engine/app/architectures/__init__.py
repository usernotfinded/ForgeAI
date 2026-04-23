from __future__ import annotations

from typing import Any


def _load_registry_exports() -> tuple[dict[str, Any], Any, Any]:
    from .registry import ARCHITECTURE_REGISTRY, get_architecture, list_architectures

    return ARCHITECTURE_REGISTRY, get_architecture, list_architectures


def _load_transformer_exports() -> tuple[Any, Any]:
    from .transformer import GPT, GPTConfig

    return GPT, GPTConfig


try:
    ARCHITECTURE_REGISTRY, get_architecture, list_architectures = _load_registry_exports()
    GPT, GPTConfig = _load_transformer_exports()
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime deps
    _IMPORT_ERROR = exc
    ARCHITECTURE_REGISTRY = {}

    def _missing(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ModuleNotFoundError(
            "PyTorch dependencies are required to use ForgeAI architectures."
        ) from _IMPORT_ERROR

    get_architecture = _missing
    list_architectures = _missing
    GPT = None
    GPTConfig = None


__all__ = [
    "GPT",
    "GPTConfig",
    "ARCHITECTURE_REGISTRY",
    "get_architecture",
    "list_architectures",
]
