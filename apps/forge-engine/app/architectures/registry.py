"""
ForgeAI — Architecture Registry
=================================

Starter model presets are intentionally small (≤400M parameters) so the project
stays lightweight and runnable on consumer hardware. All starter models are
bilingual English + Chinese — the two most resource-rich open-source language
corpora (RedPajama, Dolma, WuDaoCorporaOpen, etc.).

When a user wants to fine-tune for another language, they can use a starter model
as the base and continue training on their target language corpus.

Adding a new architecture:
  1. Implement it in its own module (e.g. mamba.py)
  2. Register a factory + presets in ARCHITECTURE_REGISTRY below
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .transformer import GPT, GPTConfig


# ── Registry Entry ────────────────────────────────────────────────────────────


@dataclass
class ArchitectureInfo:
    """Metadata and factory for a model architecture."""
    name: str
    description: str
    strengths: list[str]
    best_for: list[str]
    cpu_friendly: bool
    factory: Callable[..., Any]             # callable(config_dict) -> nn.Module
    config_class: type[Any]                 # e.g. GPTConfig
    presets: dict[str, dict[str, Any]]      # named configs → constructor kwargs


# ── Transformer (GPT-style) Presets ──────────────────────────────────────────
#
# Design constraints for all starter presets:
#   - Max 400M parameters (keeps downloads and RAM usage reasonable)
#   - Bilingual: English + Chinese (largest open-source corpora)
#   - Users can fine-tune on top of any starter for their own language/domain
#
_TRANSFORMER_PRESETS: dict[str, dict[str, Any]] = {
    "forge-nano": {
        "description": (
            "~50M params. Ultra-lightweight. Suitable for smoke tests and "
            "pipeline/tokenizer validation on low-memory hardware."
        ),
        "params_approx": "50M",
        "languages": ["en", "zh"],
        "min_ram_gb": 1,
        "config": dict(
            n_layer=6,
            n_head=8,
            n_kv_head=8,
            n_embd=512,
            context_length=1024,
            vocab_size=32000,
        ),
    },
    "forge-tiny": {
        "description": (
            "~120M params. Practical baseline for local experiments on consumer hardware. "
            "CPU-only runs are possible but typically slow."
        ),
        "params_approx": "120M",
        "languages": ["en", "zh"],
        "min_ram_gb": 4,
        "config": dict(
            n_layer=12,
            n_head=12,
            n_kv_head=12,
            n_embd=768,
            context_length=2048,
            vocab_size=32000,
        ),
    },
    "forge-small": {
        "description": (
            "~310M params. Noticeably better coherence and reasoning than nano/tiny. "
            "Recommended default for fine-tuning experiments. Fits in 8GB VRAM."
        ),
        "params_approx": "310M",
        "languages": ["en", "zh"],
        "min_ram_gb": 8,
        "config": dict(
            n_layer=24,
            n_head=16,
            n_kv_head=8,   # GQA: 2 KV heads per group
            n_embd=1024,
            context_length=4096,
            vocab_size=32000,
        ),
    },
}

# Note: Presets above 400M are intentionally excluded from starter models.
# Users who need larger models should train from scratch or use the custom
# config API: get_architecture("transformer", n_layer=32, n_head=32, n_embd=2048)


def _build_transformer(config_kwargs: dict[str, Any]) -> GPT:
    config = GPTConfig(**config_kwargs)
    return GPT(config)


# ── Registry ──────────────────────────────────────────────────────────────────

ARCHITECTURE_REGISTRY: dict[str, ArchitectureInfo] = {
    "transformer": ArchitectureInfo(
        name="transformer",
        description=(
            "Decoder-only Transformer (GPT architecture). The same fundamental design "
            "used by GPT-2, GPT-3, LLaMA, Mistral, and Gemma. Pre-LayerNorm, RoPE "
            "positional embeddings, Grouped Query Attention, and SwiGLU activation."
        ),
        strengths=[
            "Highest quality output",
            "Well-studied, easy to find training resources",
            "Excellent tooling ecosystem (Flash Attention, FSDP, etc.)",
        ],
        best_for=[
            "General-purpose chat models",
            "Code generation",
            "Instruction following",
            "Most use cases",
        ],
        cpu_friendly=False,
        factory=_build_transformer,
        config_class=GPTConfig,
        presets=_TRANSFORMER_PRESETS,
    ),
    # Future architectures — register here when implemented:
    # "mamba": ArchitectureInfo(...),   # Long-context, lower memory
    # "rwkv": ArchitectureInfo(...),    # CPU-friendly linear RNN
    # "moe": ArchitectureInfo(...),     # Sparse experts, multi-domain
}


# ── Public API ────────────────────────────────────────────────────────────────


def list_architectures() -> list[dict[str, Any]]:
    """Return a JSON-serializable list of all available architectures and their presets."""
    result = []
    for arch_name, info in ARCHITECTURE_REGISTRY.items():
        result.append({
            "name": arch_name,
            "description": info.description,
            "strengths": info.strengths,
            "best_for": info.best_for,
            "cpu_friendly": info.cpu_friendly,
            "presets": [
                {
                    "name": preset_name,
                    "description": preset_data["description"],
                    "params_approx": preset_data["params_approx"],
                    "languages": preset_data["languages"],
                    "min_ram_gb": preset_data["min_ram_gb"],
                }
                for preset_name, preset_data in info.presets.items()
            ],
        })
    return result


def get_architecture(
    arch_name: str,
    preset: str | None = None,
    **config_overrides: Any,
) -> Any:
    """
    Instantiate a model by architecture name and optional preset.

    Args:
        arch_name:         One of the keys in ARCHITECTURE_REGISTRY (e.g. "transformer")
        preset:            Named preset (e.g. "forge-tiny"). If None, config_overrides are used.
        **config_overrides: Override any config values (applied on top of preset defaults).

    Returns:
        An nn.Module ready for training or inference.

    Examples:
        # Use a starter preset
        model = get_architecture("transformer", preset="forge-tiny")

        # Override context length on a preset
        model = get_architecture("transformer", preset="forge-small", context_length=8192)

        # Fully custom (no preset) — build any size you want
        model = get_architecture("transformer", n_layer=32, n_head=32, n_embd=2048, vocab_size=50000)
    """
    if arch_name not in ARCHITECTURE_REGISTRY:
        available = list(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {available}")

    info = ARCHITECTURE_REGISTRY[arch_name]

    if preset is not None:
        if preset not in info.presets:
            available = list(info.presets.keys())
            raise ValueError(
                f"Unknown preset '{preset}' for '{arch_name}'. Available: {available}"
            )
        config_kwargs = dict(info.presets[preset]["config"])
    else:
        config_kwargs = {}

    config_kwargs.update(config_overrides)
    return info.factory(config_kwargs)
