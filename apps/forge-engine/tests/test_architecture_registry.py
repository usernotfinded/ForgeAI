from __future__ import annotations

import pytest

pytest.importorskip("torch")

from app.architectures import get_architecture, list_architectures


def test_transformer_registry_exposes_expected_presets() -> None:
    rows = list_architectures()
    transformer = next(item for item in rows if item["name"] == "transformer")
    preset_names = {preset["name"] for preset in transformer["presets"]}
    assert {"forge-nano", "forge-tiny", "forge-small"}.issubset(preset_names)


def test_get_architecture_raises_on_unknown_preset() -> None:
    with pytest.raises(ValueError):
        get_architecture("transformer", preset="does-not-exist")
