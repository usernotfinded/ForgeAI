from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from app.experimental import keras_backend
from app.experimental.keras_backend import smoke


def test_keras_availability_missing_message_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "_module_available", lambda _name: False)

    availability = keras_backend.check_availability()

    assert not availability.keras_installed
    assert "Keras is not installed" in availability.message
    assert 'pip install -e "apps/forge-engine[keras]"' in availability.message


def test_keras_backend_env_is_set_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_keras = ModuleType("keras")

    def fake_import_module(module_name: str) -> ModuleType:
        assert module_name == "keras"
        assert smoke.os.environ["KERAS_BACKEND"] == "torch"
        return fake_keras

    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.delitem(sys.modules, "keras", raising=False)
    monkeypatch.setattr(
        smoke,
        "check_availability",
        lambda: keras_backend.KerasAvailability(
            keras_installed=True,
            keras_hub_installed=False,
            backend_env=None,
            message="Keras is installed.",
        ),
    )
    monkeypatch.setattr(smoke.importlib, "import_module", fake_import_module)

    assert smoke._import_keras() is fake_keras


@pytest.mark.skipif(importlib.util.find_spec("keras") is None, reason="Keras is optional")
def test_keras_smoke_runs_when_optional_dependency_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KERAS_BACKEND", raising=False)

    result = keras_backend.run_smoke(train_step=True)

    assert result.backend == "torch"
    assert result.output_shape == (2, 2)
    assert result.train_loss is not None
