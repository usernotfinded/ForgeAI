from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_inspector import ModelInspection
from .planner import StretchPlan

FINAL_ARTIFACT_TYPE_FULL_CHECKPOINT = "full_checkpoint"
FINAL_ARTIFACT_TYPE_ADAPTER_PLUS_MANIFEST = "adapter_plus_manifest"
ADAPTER_FORMAT = "forgeai.stretch.yarn.adapter.v1"


@dataclass(frozen=True, slots=True)
class StretchAdapterArtifact:
    path: str
    sha256: str
    size_bytes: int
    entry_count: int
    format: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count,
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class StretchExecutionArtifacts:
    variant_name: str
    variant_dir: str
    variant_model_path: str
    variant_metadata_path: str
    manifest_path: str
    final_artifact_type: str
    stretch_adapter_path: str | None
    source_model_sha256: str
    variant_model_sha256: str
    variant_model_is_copy: bool
    output_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "variant_dir": self.variant_dir,
            "variant_model_path": self.variant_model_path,
            "variant_metadata_path": self.variant_metadata_path,
            "manifest_path": self.manifest_path,
            "final_artifact_type": self.final_artifact_type,
            "stretch_adapter_path": self.stretch_adapter_path,
            "source_model_sha256": self.source_model_sha256,
            "variant_model_sha256": self.variant_model_sha256,
            "variant_model_is_copy": self.variant_model_is_copy,
            "output_paths": list(self.output_paths),
        }


class StretchExecutionError(RuntimeError):
    pass


def create_persistent_variant(
    inspection: ModelInspection,
    plan: StretchPlan,
    output_root: str | Path,
) -> StretchExecutionArtifacts:
    source_dir = Path(inspection.resolved_path)
    source_model = source_dir / "model.pt"
    source_metadata = source_dir / "metadata.json"

    if not source_model.exists() or not source_metadata.exists():
        raise StretchExecutionError(
            "Source checkpoint is incomplete: model.pt and metadata.json are required."
        )

    variant_name = generate_variant_name(
        model_name=inspection.model_name,
        target_context=plan.target_context,
        method=plan.method,
    )

    output_root_path = Path(output_root)
    variant_dir = output_root_path / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    variant_model = variant_dir / "model.pt"
    variant_metadata = variant_dir / "metadata.json"
    stretch_manifest = variant_dir / "stretch_manifest.json"
    stretch_metadata = variant_dir / "stretch_metadata.json"
    stretch_adapter = variant_dir / "stretch_adapter.bin"

    _copy_or_link(source_model, variant_model)
    source_model_sha256 = _sha256_file(source_model)
    variant_model_sha256 = _sha256_file(variant_model)
    variant_model_is_copy = variant_model_sha256 == source_model_sha256

    # v1 persistent output mode: explicit adapter artifact + deterministic manifest.
    final_artifact_type = FINAL_ARTIFACT_TYPE_ADAPTER_PLUS_MANIFEST
    adapter_artifact = _create_yarn_adapter_artifact(
        output_path=stretch_adapter,
        native_context=plan.native_context,
        target_context=plan.target_context,
        factor=float(plan.yarn_config.get("factor", 1.0)),
    )

    metadata_payload = json.loads(source_metadata.read_text(encoding="utf-8"))
    updated_metadata = _build_stretched_metadata(
        metadata_payload=metadata_payload,
        inspection=inspection,
        plan=plan,
        final_artifact_type=final_artifact_type,
        stretch_adapter_relpath=stretch_adapter.name,
    )
    variant_metadata.write_text(json.dumps(updated_metadata, indent=2), encoding="utf-8")

    stretch_meta_payload = {
        "method": plan.method,
        "profile": plan.profile.value,
        "native_context": plan.native_context,
        "target_context": plan.target_context,
        "context_ratio": plan.context_ratio,
        "yarn": plan.yarn_config,
        "final_artifact_type": final_artifact_type,
        "adapter_artifact": adapter_artifact.to_dict(),
        "variant_model_is_copy": variant_model_is_copy,
        "generated_at": _utc_now_iso(),
    }
    stretch_metadata.write_text(json.dumps(stretch_meta_payload, indent=2), encoding="utf-8")

    manifest_payload = {
        "variant_name": variant_name,
        "source_model_path": str(source_model.resolve()),
        "source_metadata_path": str(source_metadata.resolve()),
        "source_model_sha256": source_model_sha256,
        "source_metadata_sha256": _sha256_file(source_metadata),
        "variant_model_sha256": variant_model_sha256,
        "variant_model_is_copy": variant_model_is_copy,
        "variant_metadata_path": str(variant_metadata.resolve()),
        "stretch_metadata_path": str(stretch_metadata.resolve()),
        "final_artifact_type": final_artifact_type,
        "adapter_artifact": adapter_artifact.to_dict(),
        "deterministic_reconstruction": {
            "enabled": True,
            "method": plan.method,
            "variant_rebuild_protocol": "forgeai-stretch-adapter-v1",
            "required_inputs": {
                "source_model_sha256": source_model_sha256,
                "source_metadata_sha256": _sha256_file(source_metadata),
                "adapter_sha256": adapter_artifact.sha256,
            },
            "steps": [
                "Load source model checkpoint and source metadata.",
                "Load stretch_adapter.bin and verify sha256.",
                "Apply YaRN rope scaling from stretch_metadata.json.",
                "Compose long-context variant using adapter artifact + metadata.",
            ],
        },
        "persistence_proof": {
            "declared_type": final_artifact_type,
            "variant_model_differs_from_source": not variant_model_is_copy,
            "adapter_artifact_present": True,
            "adapter_sha256": adapter_artifact.sha256,
        },
        "method": plan.method,
        "profile": plan.profile.value,
        "native_context": plan.native_context,
        "target_context": plan.target_context,
        "created_at": _utc_now_iso(),
    }
    stretch_manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    output_paths = [
        str(variant_model.resolve()),
        str(variant_metadata.resolve()),
        str(stretch_metadata.resolve()),
        str(stretch_manifest.resolve()),
        adapter_artifact.path,
    ]

    return StretchExecutionArtifacts(
        variant_name=variant_name,
        variant_dir=str(variant_dir.resolve()),
        variant_model_path=str(variant_model.resolve()),
        variant_metadata_path=str(variant_metadata.resolve()),
        manifest_path=str(stretch_manifest.resolve()),
        final_artifact_type=final_artifact_type,
        stretch_adapter_path=adapter_artifact.path,
        source_model_sha256=source_model_sha256,
        variant_model_sha256=variant_model_sha256,
        variant_model_is_copy=variant_model_is_copy,
        output_paths=output_paths,
    )


def generate_variant_name(model_name: str, target_context: int, method: str) -> str:
    context_label = _format_context_label(target_context)
    return f"{model_name}-{context_label}-{method}"


def _format_context_label(target_context: int) -> str:
    if target_context % 1024 == 0:
        k_value = target_context // 1024
        return f"{k_value}k"
    return str(target_context)


def _copy_or_link(source: Path, destination: Path) -> None:
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_stretched_metadata(
    metadata_payload: dict[str, Any],
    inspection: ModelInspection,
    plan: StretchPlan,
    final_artifact_type: str,
    stretch_adapter_relpath: str | None,
) -> dict[str, Any]:
    result = dict(metadata_payload)
    model_config = dict(result.get("model_config", {}))

    model_config["context_length"] = plan.target_context
    model_config["rope_scaling"] = dict(plan.yarn_config)

    result["model_config"] = model_config
    result["stretch"] = {
        "method": plan.method,
        "profile": plan.profile.value,
        "native_context": inspection.native_context,
        "target_context": plan.target_context,
        "context_ratio": plan.context_ratio,
        "risk_level": plan.risk_level,
        "risk_notes": list(plan.risk_notes),
        "final_artifact_type": final_artifact_type,
        "adapter_artifact": stretch_adapter_relpath,
        "generated_at": _utc_now_iso(),
    }

    result["source_model"] = {
        "resolved_path": inspection.resolved_path,
        "model_type": inspection.model_type.value,
        "native_context": inspection.native_context,
    }

    return result


def _create_yarn_adapter_artifact(
    output_path: Path,
    native_context: int,
    target_context: int,
    factor: float,
) -> StretchAdapterArtifact:
    positions, mapped_positions = _build_adapter_table(
        native_context=native_context,
        target_context=target_context,
        factor=factor,
    )

    with output_path.open("wb") as file_obj:
        file_obj.write(b"FSTRYARN")
        file_obj.write(struct.pack("<I", len(positions)))
        for position, mapped in zip(positions, mapped_positions):
            file_obj.write(struct.pack("<f", float(position)))
            file_obj.write(struct.pack("<f", float(mapped)))

    sha256 = _sha256_file(output_path)
    return StretchAdapterArtifact(
        path=str(output_path.resolve()),
        sha256=sha256,
        size_bytes=output_path.stat().st_size,
        entry_count=len(positions),
        format=ADAPTER_FORMAT,
    )


def _build_adapter_table(
    native_context: int,
    target_context: int,
    factor: float,
) -> tuple[list[int], list[float]]:
    if target_context <= native_context:
        raise StretchExecutionError(
            "Target context must be strictly greater than native context when creating adapter artifact."
        )
    if factor <= 1.0:
        raise StretchExecutionError("Invalid YaRN factor. Expected factor > 1.0.")

    step = max(1, (target_context - native_context) // 512)
    positions = list(range(native_context, target_context, step))
    if not positions:
        positions = [target_context - 1]
    elif positions[-1] != target_context - 1:
        positions.append(target_context - 1)

    mapped_positions = [
        _yarn_position_map(position=position, original_context=native_context, factor=factor)
        for position in positions
    ]
    return positions, mapped_positions


def _yarn_position_map(position: int, original_context: int, factor: float) -> float:
    if position <= original_context:
        return float(position)

    tail = position - original_context
    return float(original_context) + (tail / max(factor, 1e-6))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
