from __future__ import annotations

import importlib.util

import pytest
from typer.testing import CliRunner

from app.experimental.keras_backend import KerasAvailability
import app.experimental.keras_backend as keras_backend
from cli.main import app


def test_experimental_keras_smoke_missing_dependency_has_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        keras_backend,
        "check_availability",
        lambda: KerasAvailability(
            keras_installed=False,
            keras_hub_installed=False,
            backend_env=None,
            message='Keras is not installed. Install optional dependencies with: pip install -e "apps/forge-engine[keras]"',
        ),
    )

    result = CliRunner().invoke(app, ["experimental", "keras-smoke"])

    assert result.exit_code == 1
    assert "Experimental Keras integration is not available" in result.stdout
    assert "Keras is not installed" in result.stdout
    assert 'pip install -e "apps/forge-engine[keras]"' in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("keras") is None, reason="Keras is optional")
def test_experimental_keras_smoke_cli_runs_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KERAS_BACKEND", raising=False)

    result = CliRunner().invoke(app, ["experimental", "keras-smoke", "--no-train-step"])

    assert result.exit_code == 0
    assert "Experimental Keras smoke" in result.stdout
    assert "Backend : torch" in result.stdout
    assert "Forward : output shape" in result.stdout

