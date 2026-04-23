from __future__ import annotations

import json
from pathlib import Path

from app.stretch.model_inspector import ModelInspectionError, ModelType, inspect_model


def _create_checkpoint(
    root: Path,
    *,
    metadata: dict,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pt").write_bytes(b"fake-weights")
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_inspect_model_detects_base_supported(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(
        tmp_path / "base-model",
        metadata={
            "architecture": "transformer",
            "source": "huggingface",
            "step": 0,
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    info = inspect_model(checkpoint)

    assert info.model_type == ModelType.BASE_SUPPORTED
    assert info.native_context == 32768
    assert info.rope_based is True


def test_inspect_model_detects_adapted(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(
        tmp_path / "adapted-model",
        metadata={
            "architecture": "transformer",
            "source": "fine_tuned",
            "step": 1200,
            "model_config": {"context_length": 8192, "rope_theta": 10000.0},
        },
    )

    info = inspect_model(checkpoint)

    assert info.model_type == ModelType.ADAPTED


def test_inspect_model_detects_adapter_separated(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(
        tmp_path / "adapter-model",
        metadata={
            "architecture": "transformer",
            "adapters": {"lora": True},
            "model_config": {"context_length": 4096, "rope_theta": 10000.0},
        },
    )

    info = inspect_model(checkpoint)

    assert info.model_type == ModelType.ADAPTER_SEPARATED
    assert info.limits


def test_inspect_model_detects_already_stretched(tmp_path: Path) -> None:
    checkpoint = _create_checkpoint(
        tmp_path / "stretched-model",
        metadata={
            "architecture": "transformer",
            "model_config": {
                "context_length": 65536,
                "rope_theta": 10000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 32768,
                },
            },
            "stretch": {"method": "yarn"},
        },
    )

    info = inspect_model(checkpoint)

    assert info.model_type == ModelType.ALREADY_STRETCHED


def test_inspect_model_raises_on_missing_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "broken"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "model.pt").write_bytes(b"x")

    try:
        inspect_model(checkpoint)
        assert False, "Expected ModelInspectionError"
    except ModelInspectionError as exc:
        assert "metadata.json" in str(exc)
