from __future__ import annotations

import json
from pathlib import Path

from app.stretch.executor import create_persistent_variant
from app.stretch.model_inspector import inspect_model
from app.stretch.planner import AggressivenessProfile, build_stretch_plan
from app.stretch.reconstructor import (
    reconstruct_variant_from_manifest,
    run_minimal_reconstruction_demo,
)


def _checkpoint(root: Path, metadata: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pt").write_bytes(b"fake-model-binary")
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_reconstruct_adapter_plus_manifest_and_use_it_for_long_context_demo(tmp_path: Path) -> None:
    source = _checkpoint(
        tmp_path / "source-model",
        {
            "architecture": "transformer",
            "source": "huggingface",
            "model_config": {"context_length": 32768, "rope_theta": 10000.0},
        },
    )
    inspection = inspect_model(source)
    plan = build_stretch_plan(
        native_context=32768,
        target_context=131072,
        profile=AggressivenessProfile.BALANCED,
    )

    artifacts = create_persistent_variant(
        inspection=inspection,
        plan=plan,
        output_root=tmp_path / "outputs",
    )
    reconstructed = reconstruct_variant_from_manifest(artifacts.manifest_path)

    assert reconstructed.final_artifact_type == "adapter_plus_manifest"
    assert reconstructed.native_context == 32768
    assert reconstructed.target_context == 131072

    demo = run_minimal_reconstruction_demo(reconstructed)

    assert demo["stretch_retrieved_all"] is True
    assert demo["baseline_retrieved_all"] is False
    assert len(demo["retrieved_with_stretch"]) > len(demo["retrieved_without_stretch"])
