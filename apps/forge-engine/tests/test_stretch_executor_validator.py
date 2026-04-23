from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.stretch.executor import create_persistent_variant, generate_variant_name
from app.stretch.model_inspector import inspect_model
from app.stretch.planner import AggressivenessProfile, build_stretch_plan
from app.stretch.validator import validate_variant


def _checkpoint(root: Path, metadata: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pt").write_bytes(b"fake-model-binary")
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_generate_variant_name_uses_context_label() -> None:
    name = generate_variant_name("qwen2.5-0.5b", 131072, "yarn")
    assert name == "qwen2.5-0.5b-128k-yarn"


def test_create_persistent_variant_writes_required_artifacts(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-model",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.BALANCED)

    artifacts = create_persistent_variant(
        inspection=inspection,
        plan=plan,
        output_root=tmp_path / "outputs",
    )

    assert Path(artifacts.variant_model_path).exists()
    assert Path(artifacts.variant_metadata_path).exists()
    assert Path(artifacts.manifest_path).exists()
    assert artifacts.final_artifact_type == "adapter_plus_manifest"
    assert artifacts.stretch_adapter_path is not None
    assert Path(artifacts.stretch_adapter_path).exists()


def test_variant_metadata_contains_persistent_yarn_config(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-yarn",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.PRUDENT)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    metadata = json.loads(Path(artifacts.variant_metadata_path).read_text(encoding="utf-8"))
    model_config = metadata["model_config"]

    assert model_config["context_length"] == 65536
    assert model_config["rope_scaling"]["type"] == "yarn"
    assert metadata["stretch"]["method"] == "yarn"
    assert metadata["stretch"]["final_artifact_type"] == "adapter_plus_manifest"
    assert metadata["stretch"]["adapter_artifact"] == "stretch_adapter.bin"


def test_validate_variant_passes_with_stretched_metadata(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-validate",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.BALANCED)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    result = validate_variant(artifacts.variant_dir, plan)

    assert result.structural_check.passed is True
    assert result.short_context_check.passed is True
    assert result.long_context_check.passed is True
    assert result.persistence_check.passed is True
    assert result.overall_passed is True


def test_validate_variant_fails_when_target_not_greater_native(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-invalid",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.BALANCED)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    broken_metadata_path = Path(artifacts.variant_metadata_path)
    payload = json.loads(broken_metadata_path.read_text(encoding="utf-8"))
    payload["model_config"]["context_length"] = 32768
    broken_metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    invalid_plan = build_stretch_plan(32768, 32768, AggressivenessProfile.BALANCED)
    result = validate_variant(artifacts.variant_dir, invalid_plan)

    assert result.structural_check.passed is False
    assert result.long_context_check.passed is False
    assert result.persistence_check.passed is True


def test_manifest_contains_deterministic_reconstruction_and_adapter_proof(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-proof",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.BALANCED)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))
    adapter_path = Path(artifacts.stretch_adapter_path or "")
    adapter_sha = hashlib.sha256(adapter_path.read_bytes()).hexdigest()

    assert manifest["final_artifact_type"] == "adapter_plus_manifest"
    assert manifest["deterministic_reconstruction"]["enabled"] is True
    assert manifest["deterministic_reconstruction"]["required_inputs"]["adapter_sha256"] == adapter_sha
    assert manifest["adapter_artifact"]["sha256"] == adapter_sha
    assert manifest["persistence_proof"]["adapter_artifact_present"] is True


def test_long_context_validation_reports_concrete_qa_metrics(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-long-qa",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 131072, AggressivenessProfile.BALANCED)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    result = validate_variant(artifacts.variant_dir, plan)
    qa_proxy = result.long_context_check.metrics.get("qa_proxy", {})

    assert result.long_context_check.passed is True
    assert qa_proxy.get("retrieval_accuracy") == 1.0
    assert qa_proxy.get("facts_beyond_native") == qa_proxy.get("facts_total")


def test_validate_variant_fails_when_adapter_artifact_is_missing(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-missing-adapter",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.BALANCED)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    adapter_path = Path(artifacts.stretch_adapter_path or "")
    adapter_path.unlink()

    result = validate_variant(artifacts.variant_dir, plan)

    assert result.persistence_check.passed is False
    assert result.overall_passed is False


def test_validate_variant_fails_when_artifact_type_is_inconsistent(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-bad-type",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )

    inspection = inspect_model(source)
    plan = build_stretch_plan(32768, 65536, AggressivenessProfile.BALANCED)
    artifacts = create_persistent_variant(inspection, plan, tmp_path / "outputs")

    metadata_path = Path(artifacts.variant_metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stretch"]["final_artifact_type"] = "full_checkpoint"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result = validate_variant(artifacts.variant_dir, plan)

    assert result.persistence_check.passed is False
    assert result.overall_passed is False
