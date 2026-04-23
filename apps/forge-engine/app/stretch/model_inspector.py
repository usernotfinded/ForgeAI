from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ModelType(str, Enum):
    BASE_SUPPORTED = "base_supported"
    ADAPTED = "adapted"
    ADAPTER_SEPARATED = "adapter_separated"
    ALREADY_STRETCHED = "already_stretched"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelInspection:
    model_name: str
    input_path: str
    resolved_path: str
    architecture: str
    native_context: int
    rope_based: bool
    model_type: ModelType
    metadata: dict[str, Any]
    warnings: list[str]
    limits: list[str]

    @property
    def metadata_path(self) -> str:
        return str(Path(self.resolved_path) / "metadata.json")


class ModelInspectionError(RuntimeError):
    pass


def inspect_model(model_path: str | Path) -> ModelInspection:
    input_path = Path(model_path)
    if not input_path.exists():
        raise ModelInspectionError(f"Il percorso modello non esiste: {model_path}")

    resolved_dir = _resolve_checkpoint_dir(input_path)
    metadata_path = resolved_dir / "metadata.json"
    model_weights_path = resolved_dir / "model.pt"

    if not metadata_path.exists():
        raise ModelInspectionError(
            f"metadata.json mancante in {resolved_dir}. "
            "Stretch v1 supporta cartelle checkpoint compatibili ForgeAI."
        )
    if not model_weights_path.exists():
        raise ModelInspectionError(
            f"model.pt mancante in {resolved_dir}. "
            "Stretch v1 richiede un artefatto checkpoint persistente del modello."
        )

    with metadata_path.open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    architecture = str(metadata.get("architecture", "unknown"))
    model_config = metadata.get("model_config", {}) if isinstance(metadata, dict) else {}

    native_context = int(model_config.get("context_length", 0) or 0)
    rope_based = "rope_theta" in model_config or "rope_scaling" in model_config

    model_type = _detect_model_type(metadata, resolved_dir)

    warnings: list[str] = []
    limits: list[str] = []

    if native_context <= 0:
        limits.append("I metadati modello non definiscono un contesto nativo valido.")
    if not rope_based:
        limits.append("La config modello non espone metadati RoPE richiesti da YaRN stretch.")

    if model_type == ModelType.ADAPTER_SEPARATED:
        limits.append(
            "I checkpoint con adapter separati non sono supportati in stretch v1. "
            "Usa prima una variante con checkpoint unificato."
        )

    if model_type == ModelType.ALREADY_STRETCHED:
        warnings.append(
            "Il modello risulta già stretched. Un secondo stretch può aumentare il rischio di drift qualitativo."
        )

    return ModelInspection(
        model_name=resolved_dir.name,
        input_path=str(input_path.resolve()),
        resolved_path=str(resolved_dir.resolve()),
        architecture=architecture,
        native_context=native_context,
        rope_based=rope_based,
        model_type=model_type,
        metadata=metadata,
        warnings=warnings,
        limits=limits,
    )


def _resolve_checkpoint_dir(path: Path) -> Path:
    if path.is_file():
        return path.parent

    latest = path / "latest"
    if latest.exists():
        try:
            return latest.resolve()
        except OSError:
            return path
    return path


def _detect_model_type(metadata: dict[str, Any], resolved_dir: Path) -> ModelType:
    stretch_meta = metadata.get("stretch")
    model_config = metadata.get("model_config", {}) if isinstance(metadata, dict) else {}

    if stretch_meta or (isinstance(model_config, dict) and model_config.get("rope_scaling")):
        rope_scaling = model_config.get("rope_scaling") if isinstance(model_config, dict) else None
        if isinstance(rope_scaling, dict) and rope_scaling.get("type") == "yarn":
            return ModelType.ALREADY_STRETCHED

    adapter_keys = ("adapter", "adapters", "peft", "lora")
    meta_keys = {str(key).lower() for key in metadata.keys()}
    has_adapter_key = any(key in meta_keys for key in adapter_keys)
    has_adapter_folder = (resolved_dir / "adapters").exists()
    if has_adapter_key or has_adapter_folder:
        return ModelType.ADAPTER_SEPARATED

    source = str(metadata.get("source", "")).lower()
    step = int(metadata.get("step", 0) or 0)

    if step > 0 or source in {"fine_tuned", "continued_pretraining", "adapted"}:
        return ModelType.ADAPTED

    if source in {"huggingface", "forge", ""}:
        return ModelType.BASE_SUPPORTED

    return ModelType.UNKNOWN
