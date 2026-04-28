from __future__ import annotations

import pytest

from app.experimental import mlx_backend
from app.experimental.mlx_backend import availability as availability_module


def _mlx_importable() -> bool:
    return mlx_backend.check_availability().mlx_importable


def test_mlx_availability_missing_message_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(availability_module, "_module_available", lambda _name: False)

    availability = mlx_backend.check_availability()

    assert not availability.mlx_installed
    assert not availability.mlx_importable
    assert "MLX is not installed" in availability.message
    assert 'pip install -e "apps/forge-engine[mlx]"' in availability.message


def test_mlx_import_failure_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(availability_module, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        availability_module,
        "_probe_mlx_runtime",
        lambda: availability_module._MlxRuntimeProbe(
            importable=False,
            mlx_version=None,
            default_device=None,
            error="ImportError: synthetic import failure",
        ),
    )

    availability = mlx_backend.check_availability()

    assert availability.mlx_installed
    assert not availability.mlx_importable
    assert "runtime probe failed" in availability.message
    assert availability.import_error is not None


@pytest.mark.skipif(not _mlx_importable(), reason="MLX is optional")
def test_mlx_smoke_runs_when_optional_dependency_is_installed() -> None:
    result = mlx_backend.run_smoke(train_step=True)

    assert result.availability.mlx_importable
    assert result.tensor_shape == (2, 4)
    assert result.forward_shape == (2, 2)
    assert result.train_step_ran
    assert result.loss is not None
