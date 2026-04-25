from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from app.architectures.registry import ARCHITECTURE_REGISTRY
from app.wizard.runner import _find_training_command, _validate_adaptation_checkpoint


def _write_checkpoint_dir(base_dir: Path, metadata: dict[str, object]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "model.pt").write_bytes(b"dummy-weights")
    (base_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return base_dir


def _forge_small_config() -> dict[str, int]:
    config = dict(ARCHITECTURE_REGISTRY["transformer"].presets["forge-small"]["config"])
    return {key: int(value) for key, value in config.items()}


def test_find_training_command_prefers_argv() -> None:
    commands = [
        {"name": "prepare_data", "argv": ["forge", "data", "prepare", "./d"]},
        {"name": "train_adapt_existing", "argv": ["forge", "train", "--preset", "forge-small"]},
    ]

    result = _find_training_command(commands)
    assert result == ["forge", "train", "--preset", "forge-small"]


def test_find_training_command_falls_back_to_split_command_string() -> None:
    commands = [
        {
            "name": "train_from_scratch",
            "command": "forge train --preset forge-tiny --max-steps 10",
        }
    ]

    result = _find_training_command(commands)
    assert result == ["forge", "train", "--preset", "forge-tiny", "--max-steps", "10"]


def test_validate_adaptation_checkpoint_accepts_forge_small_checkpoint(tmp_path: Path) -> None:
    metadata = {
        "architecture": "transformer",
        "model_config": _forge_small_config(),
        "source": "forgeai",
    }
    checkpoint = _write_checkpoint_dir(tmp_path / "step_000001", metadata)

    resolved = _validate_adaptation_checkpoint(str(checkpoint))
    assert resolved == checkpoint.resolve()


def test_validate_adaptation_checkpoint_rejects_external_source(tmp_path: Path) -> None:
    metadata = {
        "architecture": "transformer",
        "model_config": _forge_small_config(),
        "source": "huggingface",
    }
    checkpoint = _write_checkpoint_dir(tmp_path / "step_000001", metadata)

    with pytest.raises(ValueError, match="HF/GGUF/MLX"):
        _validate_adaptation_checkpoint(str(checkpoint))


def test_validate_adaptation_checkpoint_rejects_incompatible_model_config(tmp_path: Path) -> None:
    bad_config = _forge_small_config()
    bad_config["n_layer"] = 16
    metadata = {
        "architecture": "transformer",
        "model_config": bad_config,
        "source": "forgeai",
    }
    checkpoint = _write_checkpoint_dir(tmp_path / "step_000001", metadata)

    with pytest.raises(ValueError, match="forge-small"):
        _validate_adaptation_checkpoint(str(checkpoint))
