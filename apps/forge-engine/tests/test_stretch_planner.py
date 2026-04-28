from __future__ import annotations

import json
from pathlib import Path

from app.stretch.model_inspector import inspect_model
from app.stretch.planner import (
    AggressivenessProfile,
    analyze_compatibility,
    build_stretch_plan,
    validate_target_context,
)


def _checkpoint(root: Path, metadata: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pt").write_bytes(b"weights")
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _inspect_default(tmp_path: Path):
    checkpoint = _checkpoint(
        tmp_path / "planner-model",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "step": 0,
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )
    return inspect_model(checkpoint)


def test_compatibility_passes_for_supported_backend(tmp_path: Path) -> None:
    inspection = _inspect_default(tmp_path)

    compatibility = analyze_compatibility(
        inspection=inspection,
        backend="cuda",
        vram_gb=24.0,
        unified_memory=False,
    )

    assert compatibility.is_supported is True
    assert all(target > inspection.native_context for target in compatibility.valid_targets)


def test_compatibility_warns_but_does_not_block_low_memory_supported_backend(
    tmp_path: Path,
) -> None:
    inspection = _inspect_default(tmp_path)

    compatibility = analyze_compatibility(
        inspection=inspection,
        backend="cuda",
        vram_gb=1.0,
        unified_memory=False,
    )

    assert compatibility.is_supported is True
    assert compatibility.valid_targets
    assert any("fattibilità hardware" in warning for warning in compatibility.warnings)


def test_compatibility_fails_when_backend_not_supported(tmp_path: Path) -> None:
    inspection = _inspect_default(tmp_path)

    compatibility = analyze_compatibility(
        inspection=inspection,
        backend="cpu",
        vram_gb=0.0,
        unified_memory=False,
    )

    assert compatibility.is_supported is False
    assert any("YaRN" in error for error in compatibility.errors)


def test_validate_target_rejects_equal_and_lower_native() -> None:
    native = 32768
    valid_targets = [65536, 131072]

    equal_result = validate_target_context(native, native, valid_targets)
    lower_result = validate_target_context(native, 16384, valid_targets)

    assert equal_result.is_valid is False
    assert lower_result.is_valid is False
    assert equal_result.suggested_targets


def test_validate_target_rejects_unknown_value() -> None:
    result = validate_target_context(32768, 262144, [65536, 131072])

    assert result.is_valid is False
    assert "target supportati" in (result.reason or "")


def test_build_stretch_plan_uses_single_yarn_method() -> None:
    prudent = build_stretch_plan(
        native_context=32768,
        target_context=65536,
        profile=AggressivenessProfile.PRUDENT,
    )
    ambitious = build_stretch_plan(
        native_context=32768,
        target_context=131072,
        profile=AggressivenessProfile.AMBITIOUS,
    )

    assert prudent.method == "yarn"
    assert ambitious.method == "yarn"
    assert prudent.yarn_config["type"] == "yarn"
    assert ambitious.risk_level >= prudent.risk_level
