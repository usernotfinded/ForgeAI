"""
ForgeAI — Hardware Backend Auto-Detection
==========================================

Detects the best currently usable backend for ForgeAI runtime flows.

Priority order:
  1. CUDA  — NVIDIA GPU via PyTorch
  2. MPS   — Apple Silicon via PyTorch MPS backend
  3. MLX   — Marker that MLX is available (native MLX training backend not yet implemented)
  4. CPU   — Fallback (usable for tests/debugging, very slow for training)

Usage:
    from app.core.backend import detect_backend, get_device

    backend = detect_backend()
    print(backend)               # BackendInfo(name='cuda', device='cuda:0', ...)
    model = model.to(backend.torch_device)  # PyTorch paths
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


# ── Backend Types ─────────────────────────────────────────────────────────────


class BackendType(str, Enum):
    CUDA = "cuda"       # NVIDIA GPU
    MPS  = "mps"        # Apple Silicon via PyTorch MPS
    MLX  = "mlx"        # MLX runtime detected (native MLX training backend pending)
    CPU  = "cpu"        # CPU-only fallback


@dataclass
class BackendInfo:
    """
    Complete description of the detected compute backend.

    Attributes:
        type:           Backend type (CUDA, MPS, MLX, CPU)
        torch_device:   PyTorch device string (e.g. "cuda:0", "mps", "cpu")
        device_name:    Human-readable device name (e.g. "NVIDIA RTX 4090")
        vram_gb:        VRAM in GB (GPU only), None for CPU
        unified_memory: True for Apple Silicon (RAM is shared with GPU)
        bf16_supported: Whether bfloat16 is natively supported (better than fp16 for training)
        flash_attention: Whether Flash Attention 2 is available
        mlx_available:  Whether the MLX package is installed
        notes:          List of human-readable notes and recommendations
    """
    type: BackendType
    torch_device: str
    device_name: str
    vram_gb: Optional[float]
    unified_memory: bool
    bf16_supported: bool
    flash_attention: bool
    mlx_available: bool
    notes: list[str] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        return self.type in (BackendType.CUDA, BackendType.MPS, BackendType.MLX)

    @property
    def recommended_dtype(self) -> str:
        """Best floating-point dtype for this backend."""
        if self.bf16_supported:
            return "bfloat16"
        if self.type == BackendType.MPS:
            return "float16"
        return "float32"

    @property
    def recommended_preset(self) -> str:
        """Recommended starter model preset for this hardware."""
        if self.type == BackendType.MLX and self.torch_device == "cpu":
            # MLX runtime is present, but PyTorch training falls back to CPU for now.
            return "forge-nano"

        if self.vram_gb is None:
            # CPU only — keep it very small
            return "forge-nano"
        if self.unified_memory:
            # Apple Silicon: unified memory — use effective total RAM
            effective_gb = self.vram_gb
        else:
            effective_gb = self.vram_gb

        if effective_gb < 6:
            return "forge-nano"
        elif effective_gb < 12:
            return "forge-tiny"
        else:
            return "forge-small"

    def __str__(self) -> str:
        lines = [
            f"Backend    : {self.type.value.upper()}",
            f"Device     : {self.device_name}",
            f"Torch dev  : {self.torch_device}",
        ]
        if self.vram_gb:
            mem_label = "Unified RAM" if self.unified_memory else "VRAM"
            lines.append(f"{mem_label:<11}: {self.vram_gb:.1f} GB")
        lines += [
            f"dtype      : {self.recommended_dtype}",
            f"Flash Attn : {'✓' if self.flash_attention else '✗'}",
            f"MLX        : {'✓' if self.mlx_available else '✗'}",
            f"Preset     : {self.recommended_preset}",
        ]
        if self.notes:
            lines.append("Notes      :")
            for note in self.notes:
                lines.append(f"  • {note}")
        return "\n".join(lines)


# ── Detection Logic ───────────────────────────────────────────────────────────


def _check_mlx() -> bool:
    """Return True if the mlx package is importable."""
    if os.environ.get("FORGE_DISABLE_MLX_CHECK", "0") == "1":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _check_flash_attention() -> bool:
    """Return True if flash_attn is installed (CUDA only)."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _get_apple_silicon_memory_gb() -> float:
    """
    On Apple Silicon, RAM is unified (shared CPU/GPU).
    Query total system memory from sysctl.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3
        )
        return int(result.stdout.strip()) / 1e9
    except Exception:
        return 8.0  # conservative fallback


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_backend() -> BackendInfo:
    """
    Auto-detect the best available compute backend.

    Checks in order: CUDA → MPS (Apple) → MLX marker (Apple) → CPU.
    Returns a BackendInfo used by the current PyTorch training pipeline.
    """
    mlx_available = _check_mlx()
    notes: list[str] = []

    # ── 1. CUDA (NVIDIA GPU) ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        device_name = props.name
        flash = _check_flash_attention()

        # BF16: supported on Ampere (sm_80) and later
        bf16 = props.major >= 8

        if not flash:
            notes.append(
                "Install flash-attn for 2-3x faster training: "
                "pip install flash-attn --no-build-isolation"
            )
        if not bf16:
            notes.append(
                f"GPU compute capability {props.major}.{props.minor} — "
                "bfloat16 not supported, using float16. Consider a newer GPU for training."
            )

        return BackendInfo(
            type=BackendType.CUDA,
            torch_device=f"cuda:{torch.cuda.current_device()}",
            device_name=device_name,
            vram_gb=round(vram_gb, 1),
            unified_memory=False,
            bf16_supported=bf16,
            flash_attention=flash,
            mlx_available=mlx_available,
            notes=notes,
        )

    # ── 2. Apple Silicon — PyTorch MPS ───────────────────────────────────────
    if _is_apple_silicon() and torch.backends.mps.is_available():
        ram_gb = _get_apple_silicon_memory_gb()
        notes.append(
            "Training uses PyTorch MPS on Apple Silicon."
        )
        if mlx_available:
            notes.append(
                "MLX package detected: available for mlx-lm inference workflows. "
                "Native MLX training backend is still planned."
            )
        else:
            notes.append(
                "MLX package not detected. Install mlx/mlx-lm to enable MLX inference workflows."
            )

        return BackendInfo(
            type=BackendType.MPS,
            torch_device="mps",
            device_name=f"Apple Silicon (MPS) — {platform.processor() or 'M-series'}",
            vram_gb=round(ram_gb, 1),
            unified_memory=True,
            bf16_supported=False,  # MPS has limited bf16 support
            flash_attention=False,
            mlx_available=mlx_available,
            notes=notes,
        )

    # ── 3. Apple Silicon — MLX marker (no MPS available) ─────────────────────
    if _is_apple_silicon() and mlx_available:
        ram_gb = _get_apple_silicon_memory_gb()
        notes.append(
            "MLX package detected, but PyTorch MPS backend is unavailable."
        )
        notes.append(
            "Native MLX training backend is not implemented yet: ForgeAI training falls back to CPU in this environment."
        )

        return BackendInfo(
            type=BackendType.MLX,
            torch_device="cpu",  # TODO(native-mlx-training): replace with real MLX training device path.
            device_name=f"Apple Silicon (MLX runtime) — {platform.processor() or 'M-series'}",
            vram_gb=round(ram_gb, 1),
            unified_memory=True,
            bf16_supported=False,
            flash_attention=False,
            mlx_available=True,
            notes=notes,
        )

    # ── 4. CPU fallback ───────────────────────────────────────────────────────
    cpu_count = torch.get_num_threads()
    notes.append(
        "No CUDA/MPS backend detected. CPU mode is intended for tests/debugging and very small runs."
    )
    notes.append(
        f"Using {cpu_count} CPU threads. Set OMP_NUM_THREADS to override."
    )

    return BackendInfo(
        type=BackendType.CPU,
        torch_device="cpu",
        device_name=f"CPU ({platform.processor() or platform.machine()}, {cpu_count} threads)",
        vram_gb=None,
        unified_memory=False,
        bf16_supported=False,
        flash_attention=False,
        mlx_available=mlx_available,
        notes=notes,
    )


# ── Convenience ───────────────────────────────────────────────────────────────


_cached_backend: Optional[BackendInfo] = None


def get_backend(force_refresh: bool = False) -> BackendInfo:
    """
    Return the cached backend (detected once on first call).
    Use force_refresh=True to re-detect (e.g. after attaching a GPU).
    """
    global _cached_backend
    if _cached_backend is None or force_refresh:
        _cached_backend = detect_backend()
    return _cached_backend


def get_device() -> str:
    """Shortcut: return the torch device string for the detected backend."""
    return get_backend().torch_device


if __name__ == "__main__":
    print("=" * 50)
    print("ForgeAI — Backend Detection")
    print("=" * 50)
    print()
    backend = detect_backend()
    print(backend)
