"""
ForgeAI — Hardware Backend Auto-Detection
==========================================

Detects the best available compute backend and returns a unified interface
for placing models and tensors on the right device.

Priority order:
  1. CUDA  — NVIDIA GPU (best training performance)
  2. MPS   — Apple Silicon via PyTorch MPS backend
  3. MLX   — Apple Silicon via Apple's MLX framework (optional, if installed)
  4. CPU   — Fallback (inference only; training is very slow)

Usage:
    from app.core.backend import detect_backend, get_device

    backend = detect_backend()
    print(backend)               # BackendInfo(name='cuda', device='cuda:0', ...)
    model = model.to(backend.torch_device)
"""

from __future__ import annotations

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
    MLX  = "mlx"        # Apple Silicon via MLX (preferred on Apple for training)
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

    Checks in order: CUDA → MLX (Apple) → MPS (Apple) → CPU
    Returns a BackendInfo with all relevant information for training configuration.
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

    # ── 2. Apple Silicon — MLX (preferred) ───────────────────────────────────
    if _is_apple_silicon() and mlx_available:
        ram_gb = _get_apple_silicon_memory_gb()
        notes.append(
            "MLX detected — using Apple's native ML framework for best Apple Silicon performance."
        )
        notes.append(
            "MLX uses unified memory: models and activations share the same RAM pool as the OS."
        )

        return BackendInfo(
            type=BackendType.MLX,
            torch_device="cpu",   # MLX has its own array system; PyTorch falls back to CPU
            device_name=f"Apple Silicon (MLX) — {platform.processor() or 'M-series'}",
            vram_gb=round(ram_gb, 1),
            unified_memory=True,
            bf16_supported=True,   # MLX supports bfloat16 natively
            flash_attention=False, # flash_attn is CUDA-only; MLX has its own attention kernels
            mlx_available=True,
            notes=notes,
        )

    # ── 3. Apple Silicon — PyTorch MPS ───────────────────────────────────────
    if _is_apple_silicon() and torch.backends.mps.is_available():
        ram_gb = _get_apple_silicon_memory_gb()
        notes.append(
            "Using PyTorch MPS backend (Apple Silicon). "
            "For better performance, install MLX: pip install mlx"
        )
        notes.append(
            "MPS uses unified memory — models and activations count against total RAM."
        )

        return BackendInfo(
            type=BackendType.MPS,
            torch_device="mps",
            device_name=f"Apple Silicon (MPS) — {platform.processor() or 'M-series'}",
            vram_gb=round(ram_gb, 1),
            unified_memory=True,
            bf16_supported=False,  # MPS has limited bf16 support
            flash_attention=False,
            mlx_available=False,
            notes=notes,
        )

    # ── 4. CPU fallback ───────────────────────────────────────────────────────
    cpu_count = torch.get_num_threads()
    notes.append(
        "No GPU detected. Training will be very slow on CPU. "
        "Inference of forge-nano/tiny is feasible."
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
        mlx_available=False,
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
