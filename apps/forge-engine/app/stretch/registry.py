from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StretchRecipe:
    architecture: str
    method: str
    supported_backends: tuple[str, ...]
    max_target_context: int
    target_multipliers: tuple[int, ...]
    min_native_context: int
    notes: tuple[str, ...]


STRETCH_REGISTRY_VERSION = "1.0"

# Local/static versioned registry. No remote lookups.
LOCAL_STRETCH_REGISTRY: dict[str, StretchRecipe] = {
    "transformer": StretchRecipe(
        architecture="transformer",
        method="yarn",
        supported_backends=("cuda", "mlx", "mps"),
        max_target_context=262_144,
        target_multipliers=(2, 4, 8),
        min_native_context=1_024,
        notes=(
            "Supports RoPE-based decoder-only transformer checkpoints.",
            "Stretch profiles map to different aggressiveness levels within YaRN only.",
        ),
    ),
}


def get_stretch_recipe(architecture: str) -> StretchRecipe | None:
    return LOCAL_STRETCH_REGISTRY.get(architecture)


def list_registry_entries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recipe in LOCAL_STRETCH_REGISTRY.values():
        rows.append(
            {
                "architecture": recipe.architecture,
                "method": recipe.method,
                "supported_backends": list(recipe.supported_backends),
                "max_target_context": recipe.max_target_context,
                "target_multipliers": list(recipe.target_multipliers),
                "min_native_context": recipe.min_native_context,
                "notes": list(recipe.notes),
                "registry_version": STRETCH_REGISTRY_VERSION,
            }
        )
    return rows
