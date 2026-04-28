from __future__ import annotations

import pytest
from typer.testing import CliRunner

from app.experimental.mlx_backend import MlxAvailability
import app.experimental.mlx_backend as mlx_backend
from cli.main import app


def _mlx_importable() -> bool:
    return mlx_backend.check_availability().mlx_importable


def test_experimental_mlx_smoke_missing_dependency_has_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mlx_backend,
        "check_availability",
        lambda: MlxAvailability(
            mlx_installed=False,
            mlx_importable=False,
            apple_silicon=False,
            platform="linux",
            machine="x86_64",
            mlx_version=None,
            default_device=None,
            import_error=None,
            message='MLX is not installed. Install optional dependencies with: pip install -e "apps/forge-engine[mlx]"',
        ),
    )

    result = CliRunner().invoke(app, ["experimental", "mlx-smoke"])

    assert result.exit_code == 1
    assert "Experimental MLX backend is not available" in result.stdout
    assert "MLX is not installed" in result.stdout
    assert 'pip install -e "apps/forge-engine[mlx]"' in result.stdout


@pytest.mark.skipif(not _mlx_importable(), reason="MLX is optional")
def test_experimental_mlx_smoke_cli_runs_when_installed() -> None:
    result = CliRunner().invoke(app, ["experimental", "mlx-smoke", "--no-train-step"])

    assert result.exit_code == 0
    assert "Experimental MLX smoke" in result.stdout
    assert "Tensor  : shape" in result.stdout
    assert "Forward : output shape" in result.stdout
