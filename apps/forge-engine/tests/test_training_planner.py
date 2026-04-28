from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from rich.console import Console

from app.training.planner import (
    HARDWARE_ADVISORY_POLICY,
    _parse_params,
    estimate_training,
    inspect_data_path,
)


def test_parse_params_supports_common_suffixes() -> None:
    assert _parse_params("400M") == 400_000_000
    assert _parse_params("1.3B") == 1_300_000_000
    assert _parse_params(" 50m ") == 50_000_000
    assert _parse_params("1200000") == 1_200_000


def test_parse_params_raises_for_invalid_input() -> None:
    with pytest.raises(ValueError):
        _parse_params("not-a-number")


def test_estimate_training_returns_positive_values_for_cpu_backend() -> None:
    backend = SimpleNamespace(
        type=SimpleNamespace(value="cpu"),
        device_name="cpu-test",
        vram_gb=None,
        recommended_dtype="float32",
    )

    plan = estimate_training(
        arch="transformer",
        params="50M",
        data_path=".",
        backend=backend,
    )

    assert plan.params == 50_000_000
    assert plan.estimated_tokens > 0
    assert plan.estimated_hours > 0
    assert plan.recommended_batch_size >= 1
    assert plan.feasibility_status in {"good", "risky", "not recommended"}
    assert isinstance(plan.data_inspection, dict)


def test_estimate_training_warns_but_does_not_fail_for_unrealistic_hardware() -> None:
    backend = SimpleNamespace(
        type=SimpleNamespace(value="cpu"),
        device_name="tiny-cpu",
        vram_gb=0.1,
        recommended_dtype="float32",
    )

    plan = estimate_training(
        arch="transformer",
        params="7B",
        data_path=".",
        backend=backend,
    )

    assert plan.feasibility_status == "not recommended"
    assert plan.memory_risk == "not recommended"
    assert any("not recommended" in warning.lower() for warning in plan.warnings)


def test_training_plan_output_mentions_hardware_advisory_policy() -> None:
    backend = SimpleNamespace(
        type=SimpleNamespace(value="cpu"),
        device_name="cpu-test",
        vram_gb=None,
        recommended_dtype="float32",
    )
    plan = estimate_training(
        arch="transformer",
        params="50M",
        data_path=".",
        backend=backend,
    )
    console = Console(force_terminal=False, color_system=None, width=140)

    with console.capture() as capture:
        plan.print_summary(console)

    assert HARDWARE_ADVISORY_POLICY in capture.get()


def test_estimate_training_raises_for_missing_data_path() -> None:
    backend = SimpleNamespace(
        type=SimpleNamespace(value="cpu"),
        device_name="cpu-test",
        vram_gb=None,
        recommended_dtype="float32",
    )
    with pytest.raises(FileNotFoundError, match="Dataset path not found"):
        estimate_training(
            arch="transformer",
            params="50M",
            data_path="./path-does-not-exist-forgeai",
            backend=backend,
        )


def test_inspect_data_path_reads_prepared_metadata(tmp_path: Path) -> None:
    metadata = {
        "total_tokens": 123456,
        "num_shards": 3,
        "context_length": 2048,
        "token_dtype": "uint32",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    inspection = inspect_data_path(tmp_path)

    assert inspection["prepared_metadata_found"] is True
    assert inspection["prepared_total_tokens"] == 123456
    assert inspection["prepared_num_shards"] == 3
    assert inspection["prepared_context_length"] == 2048
    assert inspection["prepared_token_dtype"] == "uint32"


def test_inspect_data_path_reads_metadata_from_shard_parent(tmp_path: Path) -> None:
    metadata = {
        "total_tokens": 42,
        "num_shards": 1,
        "context_length": 128,
        "token_dtype": "uint32",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "shard_0000.bin").write_bytes(b"\x00\x00\x00\x00")

    inspection = inspect_data_path(tmp_path / "shard_0000.bin")
    assert inspection["prepared_metadata_found"] is True
    assert inspection["prepared_total_tokens"] == 42
