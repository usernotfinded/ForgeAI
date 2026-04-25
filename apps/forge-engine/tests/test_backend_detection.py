from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from app.core import backend as backend_module
from app.core.backend import BackendType


def test_detect_backend_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(backend_module, "_check_mlx", lambda: True)
    monkeypatch.setattr(backend_module, "_check_flash_attention", lambda: False)
    monkeypatch.setattr(backend_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(backend_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        backend_module.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=24_000_000_000, name="RTX 4090", major=8, minor=9),
    )

    detected = backend_module.detect_backend()

    assert detected.type == BackendType.CUDA
    assert detected.torch_device == "cuda:0"
    assert detected.mlx_available is True


def test_detect_backend_prefers_mps_over_mlx_marker_on_apple(monkeypatch) -> None:
    monkeypatch.setattr(backend_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(backend_module, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(backend_module, "_check_mlx", lambda: True)
    monkeypatch.setattr(backend_module, "_get_apple_silicon_memory_gb", lambda: 16.0)
    monkeypatch.setattr(backend_module.torch.backends.mps, "is_available", lambda: True, raising=False)

    detected = backend_module.detect_backend()

    assert detected.type == BackendType.MPS
    assert detected.torch_device == "mps"
    assert detected.mlx_available is True
    assert any("training uses pytorch mps" in note.lower() for note in detected.notes)


def test_detect_backend_marks_mlx_when_mps_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(backend_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(backend_module, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(backend_module, "_check_mlx", lambda: True)
    monkeypatch.setattr(backend_module, "_get_apple_silicon_memory_gb", lambda: 16.0)
    monkeypatch.setattr(backend_module.torch.backends.mps, "is_available", lambda: False, raising=False)

    detected = backend_module.detect_backend()

    assert detected.type == BackendType.MLX
    assert detected.torch_device == "cpu"
    assert detected.recommended_preset == "forge-nano"
    assert any("not implemented yet" in note.lower() for note in detected.notes)


def test_detect_backend_cpu_fallback_keeps_mlx_availability(monkeypatch) -> None:
    monkeypatch.setattr(backend_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(backend_module, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(backend_module, "_check_mlx", lambda: True)

    detected = backend_module.detect_backend()

    assert detected.type == BackendType.CPU
    assert detected.torch_device == "cpu"
    assert detected.mlx_available is True
