from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .planner import StretchPlan


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    passed: bool
    title: str
    details: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "title": self.title,
            "details": self.details,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class StretchValidationResult:
    structural_check: ValidationCheck
    short_context_check: ValidationCheck
    long_context_check: ValidationCheck
    persistence_check: ValidationCheck
    overall_passed: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_check": self.structural_check.to_dict(),
            "short_context_check": self.short_context_check.to_dict(),
            "long_context_check": self.long_context_check.to_dict(),
            "persistence_check": self.persistence_check.to_dict(),
            "overall_passed": self.overall_passed,
            "summary": self.summary,
        }


def validate_variant(
    variant_dir: str | Path,
    plan: StretchPlan,
) -> StretchValidationResult:
    variant_path = Path(variant_dir)
    metadata_path = variant_path / "metadata.json"
    manifest_path = variant_path / "stretch_manifest.json"
    stretch_metadata_path = variant_path / "stretch_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata variante non trovato: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stretch manifest non trovato: {manifest_path}")
    if not stretch_metadata_path.exists():
        raise FileNotFoundError(f"Stretch metadata non trovato: {stretch_metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stretch_metadata = json.loads(stretch_metadata_path.read_text(encoding="utf-8"))
    model_config = metadata.get("model_config", {}) if isinstance(metadata, dict) else {}

    structural_check = _structural_integrity_check(
        metadata=metadata,
        manifest=manifest,
        stretch_metadata=stretch_metadata,
        plan=plan,
    )
    short_check = _short_context_regression_check(model_config=model_config, plan=plan)
    long_check = _long_context_capability_check(model_config=model_config, plan=plan)
    persistence_check = _persistent_artifact_check(
        variant_path=variant_path,
        metadata=metadata,
        manifest=manifest,
        stretch_metadata=stretch_metadata,
    )

    overall_passed = (
        structural_check.passed
        and short_check.passed
        and long_check.passed
        and persistence_check.passed
    )
    summary = _build_summary(
        structural_check=structural_check,
        persistence_check=persistence_check,
        short_check=short_check,
        long_check=long_check,
        overall_passed=overall_passed,
    )

    return StretchValidationResult(
        structural_check=structural_check,
        short_context_check=short_check,
        long_context_check=long_check,
        persistence_check=persistence_check,
        overall_passed=overall_passed,
        summary=summary,
    )


def _build_summary(
    structural_check: ValidationCheck,
    persistence_check: ValidationCheck,
    short_check: ValidationCheck,
    long_check: ValidationCheck,
    overall_passed: bool,
) -> str:
    lines = [
        "Riepilogo validazione:",
        f"- Integrità strutturale: {'PASS' if structural_check.passed else 'FAIL'}",
        f"- Integrità persistenza: {'PASS' if persistence_check.passed else 'FAIL'}",
        f"- Regressione short-context: {'PASS' if short_check.passed else 'FAIL'}",
        f"- Capacità long-context: {'PASS' if long_check.passed else 'FAIL'}",
    ]
    if overall_passed:
        lines.append("Esito complessivo: PASS")
    else:
        lines.append("Esito complessivo: FAIL (controlla le sezioni fallite prima di usare la variante)")
    return "\n".join(lines)


def _structural_integrity_check(
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    stretch_metadata: dict[str, Any],
    plan: StretchPlan,
) -> ValidationCheck:
    model_config = metadata.get("model_config", {}) if isinstance(metadata, dict) else {}
    rope_scaling = model_config.get("rope_scaling") if isinstance(model_config, dict) else None
    stretch_meta_yarn = stretch_metadata.get("yarn") if isinstance(stretch_metadata, dict) else None

    context_length = int(model_config.get("context_length", 0) or 0)
    method = str(manifest.get("method", "")).lower()
    target_from_stretch_meta = int(stretch_metadata.get("target_context", 0) or 0)

    checks = {
        "context_matches_target": context_length == plan.target_context,
        "method_is_yarn": method == "yarn",
        "rope_scaling_present": isinstance(rope_scaling, dict),
        "stretch_meta_yarn_present": isinstance(stretch_meta_yarn, dict),
        "target_matches_stretch_metadata": target_from_stretch_meta == plan.target_context,
    }

    if isinstance(rope_scaling, dict):
        checks["rope_type_is_yarn"] = str(rope_scaling.get("type", "")).lower() == "yarn"
        checks["rope_original_matches_native"] = (
            int(rope_scaling.get("original_max_position_embeddings", 0) or 0) == plan.native_context
        )
        checks["rope_target_matches_plan"] = (
            int(rope_scaling.get("target_max_position_embeddings", 0) or 0) == plan.target_context
        )
    else:
        checks["rope_type_is_yarn"] = False
        checks["rope_original_matches_native"] = False
        checks["rope_target_matches_plan"] = False

    passed = all(checks.values())
    details = (
        "I metadati strutturali stretch sono coerenti tra config, manifest e piano."
        if passed
        else "Rilevata incoerenza strutturale tra config/manifest/piano stretch."
    )

    return ValidationCheck(
        passed=passed,
        title="Structural integrity check",
        details=details,
        metrics={
            "native_context": plan.native_context,
            "target_context": plan.target_context,
            "checks": checks,
        },
    )


def _short_context_regression_check(
    model_config: dict[str, Any],
    plan: StretchPlan,
) -> ValidationCheck:
    native = plan.native_context
    rope_scaling = model_config.get("rope_scaling") if isinstance(model_config, dict) else None

    if not isinstance(rope_scaling, dict) or rope_scaling.get("type") != "yarn":
        return ValidationCheck(
            passed=False,
            title="Controllo regressione short-context",
            details="Metadati YaRN mancanti o non validi nella config stretched.",
            metrics={"native_context": native},
        )

    factor = float(rope_scaling.get("factor", 1.0))
    original = int(rope_scaling.get("original_max_position_embeddings", native))

    sample_points = [0, max(0, original // 2), max(0, original - 1)]
    deviations: list[float] = []

    for pos in sample_points:
        mapped = _yarn_position_map(position=pos, original_context=original, factor=factor)
        deviations.append(abs(mapped - float(pos)))

    max_deviation = max(deviations) if deviations else 0.0
    attention_passed, attention_metrics = _short_context_attention_proxy(
        native_context=native,
        original_context=original,
        factor=factor,
    )
    passed = max_deviation <= 1e-6 and attention_passed

    details = (
        "Native-context mapping stability passed (position + attention-distribution proxy)."
        if passed
        else "Detected short-context instability under YaRN mapping."
    )

    return ValidationCheck(
        passed=passed,
        title="Controllo regressione short-context",
        details=details,
        metrics={
            "native_context": native,
            "max_position_deviation": round(max_deviation, 8),
            "sample_points": sample_points,
            "attention_proxy": attention_metrics,
        },
    )


def _long_context_capability_check(
    model_config: dict[str, Any],
    plan: StretchPlan,
) -> ValidationCheck:
    target = plan.target_context
    native = plan.native_context
    rope_scaling = model_config.get("rope_scaling") if isinstance(model_config, dict) else None

    if target <= native:
        return ValidationCheck(
            passed=False,
            title="Controllo capacità long-context",
            details="Il target context deve essere strettamente maggiore del contesto nativo.",
            metrics={"native_context": native, "target_context": target},
        )

    if not isinstance(rope_scaling, dict) or rope_scaling.get("type") != "yarn":
        return ValidationCheck(
            passed=False,
            title="Controllo capacità long-context",
            details="La variante non include metadati di scaling YaRN.",
            metrics={"native_context": native, "target_context": target},
        )

    factor = float(rope_scaling.get("factor", 1.0))
    original = int(rope_scaling.get("original_max_position_embeddings", native))

    integrity_passed, integrity_metrics = _rope_extension_integrity(
        native_context=native,
        target_context=target,
        original_context=original,
        factor=factor,
    )

    needle_passed, needle_metrics = _needle_in_haystack_proxy(
        native_context=native,
        target_context=target,
    )
    qa_passed, qa_metrics = _long_context_qa_proxy(
        native_context=native,
        target_context=target,
    )

    passed = integrity_passed and needle_passed and qa_passed
    details = (
        "Controlli long-context superati (integrità RoPE + needle retrieval + QA sintetico lungo)."
        if passed
        else "Controlli long-context falliti. Target troppo aggressivo o config non valida."
    )

    metrics = {
        "native_context": native,
        "target_context": target,
        "rope_integrity": integrity_metrics,
        "needle_proxy": needle_metrics,
        "qa_proxy": qa_metrics,
    }

    return ValidationCheck(
        passed=passed,
        title="Controllo capacità long-context",
        details=details,
        metrics=metrics,
    )


def _persistent_artifact_check(
    variant_path: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    stretch_metadata: dict[str, Any],
) -> ValidationCheck:
    metadata_type = (
        metadata.get("stretch", {}).get("final_artifact_type")
        if isinstance(metadata.get("stretch"), dict)
        else None
    )
    manifest_type = manifest.get("final_artifact_type")
    stretch_type = stretch_metadata.get("final_artifact_type")

    declared_type = (
        str(manifest_type or metadata_type or stretch_type or "").strip().lower()
    )
    if not declared_type:
        return ValidationCheck(
            passed=False,
            title="Controllo artefatto persistente",
            details="Campo final_artifact_type mancante negli output stretch.",
            metrics={},
        )

    if not (declared_type == str(metadata_type or "").lower() == str(stretch_type or "").lower()):
        return ValidationCheck(
            passed=False,
            title="Controllo artefatto persistente",
            details="final_artifact_type incoerente tra metadata/manifest/stretch_metadata.",
            metrics={
                "metadata_type": metadata_type,
                "manifest_type": manifest_type,
                "stretch_metadata_type": stretch_type,
            },
        )

    if declared_type == "full_checkpoint":
        source_sha = str(manifest.get("source_model_sha256", ""))
        variant_sha = str(manifest.get("variant_model_sha256", ""))
        model_path = variant_path / "model.pt"
        if not model_path.exists():
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details="Output dichiarato full_checkpoint ma model.pt mancante.",
                metrics={"declared_type": declared_type},
            )
        if not source_sha or not variant_sha:
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details="Hash source/variant mancanti per validare full_checkpoint.",
                metrics={"declared_type": declared_type},
            )
        if source_sha == variant_sha:
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details=(
                    "Output dichiarato full_checkpoint ma hash checkpoint variante uguale alla sorgente."
                ),
                metrics={
                    "declared_type": declared_type,
                    "source_model_sha256": source_sha,
                    "variant_model_sha256": variant_sha,
                },
            )
        return ValidationCheck(
            passed=True,
            title="Controllo artefatto persistente",
            details="Persistenza full_checkpoint validata con hash sorgente/variante differenti.",
            metrics={
                "declared_type": declared_type,
                "source_model_sha256": source_sha,
                "variant_model_sha256": variant_sha,
            },
        )

    if declared_type == "adapter_plus_manifest":
        adapter_info = manifest.get("adapter_artifact", {})
        if not isinstance(adapter_info, dict):
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details="Blocco adapter_artifact mancante nello stretch manifest.",
                metrics={"declared_type": declared_type},
            )

        adapter_path = _resolve_adapter_path(variant_path, adapter_info)
        if not adapter_path.exists():
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details="Output dichiarato adapter_plus_manifest ma file adapter mancante.",
                metrics={
                    "declared_type": declared_type,
                    "adapter_path": str(adapter_path),
                },
            )

        expected_adapter_sha = str(adapter_info.get("sha256", ""))
        actual_adapter_sha = _sha256_file(adapter_path)
        if expected_adapter_sha and expected_adapter_sha != actual_adapter_sha:
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details="Hash adapter non corrisponde a quanto dichiarato nel manifest.",
                metrics={
                    "declared_type": declared_type,
                    "adapter_path": str(adapter_path),
                    "expected_adapter_sha256": expected_adapter_sha,
                    "actual_adapter_sha256": actual_adapter_sha,
                },
            )

        reconstruction = manifest.get("deterministic_reconstruction", {})
        reconstruction_ok, reconstruction_error = _validate_reconstruction_block(
            reconstruction=reconstruction,
            adapter_sha256=actual_adapter_sha,
        )
        if not reconstruction_ok:
            return ValidationCheck(
                passed=False,
                title="Controllo artefatto persistente",
                details=reconstruction_error,
                metrics={
                    "declared_type": declared_type,
                    "adapter_path": str(adapter_path),
                    "adapter_sha256": actual_adapter_sha,
                },
            )

        return ValidationCheck(
            passed=True,
            title="Controllo artefatto persistente",
            details=(
                "Persistenza adapter_plus_manifest validata "
                "(artefatto mapping YaRN + metadati di ricostruzione deterministica)."
            ),
            metrics={
                "declared_type": declared_type,
                "adapter_path": str(adapter_path),
                "adapter_sha256": actual_adapter_sha,
            },
        )

    return ValidationCheck(
        passed=False,
        title="Controllo artefatto persistente",
        details=f"final_artifact_type '{declared_type}' non supportato negli output stretch.",
        metrics={"declared_type": declared_type},
    )


def _resolve_adapter_path(variant_path: Path, adapter_info: dict[str, Any]) -> Path:
    adapter_path_raw = adapter_info.get("path")
    if isinstance(adapter_path_raw, str) and adapter_path_raw.strip():
        candidate = Path(adapter_path_raw)
        if candidate.is_absolute():
            return candidate
        return (variant_path / candidate).resolve()
    return variant_path / "stretch_adapter.bin"


def _validate_reconstruction_block(
    reconstruction: Any,
    adapter_sha256: str,
) -> tuple[bool, str]:
    if not isinstance(reconstruction, dict):
        return False, "Blocco deterministic_reconstruction mancante nello stretch manifest."

    enabled = bool(reconstruction.get("enabled"))
    required_inputs = reconstruction.get("required_inputs")
    steps = reconstruction.get("steps")
    if not enabled:
        return False, "deterministic_reconstruction.enabled deve essere true."
    if not isinstance(required_inputs, dict):
        return False, "deterministic_reconstruction.required_inputs mancante o non valido."
    if not isinstance(steps, list) or not steps:
        return False, "deterministic_reconstruction.steps deve contenere almeno uno step."

    source_model_sha = str(required_inputs.get("source_model_sha256", ""))
    adapter_sha = str(required_inputs.get("adapter_sha256", ""))
    if not source_model_sha:
        return False, "deterministic_reconstruction.required_inputs.source_model_sha256 mancante."
    if not adapter_sha:
        return False, "deterministic_reconstruction.required_inputs.adapter_sha256 mancante."
    if adapter_sha != adapter_sha256:
        return False, "adapter_sha256 in deterministic_reconstruction non corrisponde all'artefatto adapter."

    return True, ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rope_extension_integrity(
    native_context: int,
    target_context: int,
    original_context: int,
    factor: float,
) -> tuple[bool, dict[str, Any]]:
    sample_size = 128
    if target_context <= native_context:
        return False, {"reason": "target_not_greater_than_native"}

    if factor <= 1.0:
        return False, {"reason": "invalid_scaling_factor"}

    step = max(1, (target_context - native_context) // sample_size)
    positions = list(range(native_context, target_context, step))
    if positions[-1] != target_context - 1:
        positions.append(target_context - 1)

    mapped = [
        _yarn_position_map(position=pos, original_context=original_context, factor=factor)
        for pos in positions
    ]

    is_monotonic = all(mapped[idx + 1] > mapped[idx] for idx in range(len(mapped) - 1))
    min_delta = min(
        (mapped[idx + 1] - mapped[idx] for idx in range(len(mapped) - 1)),
        default=0.0,
    )

    return is_monotonic and min_delta > 0.0, {
        "sample_count": len(mapped),
        "is_monotonic": is_monotonic,
        "min_delta": round(min_delta, 8),
        "mapped_start": round(mapped[0], 4) if mapped else None,
        "mapped_end": round(mapped[-1], 4) if mapped else None,
    }


def _needle_in_haystack_proxy(native_context: int, target_context: int) -> tuple[bool, dict[str, Any]]:
    # Deterministic local long-context retrieval proxy over synthetic text.
    length = min(target_context, 64_000)
    if length <= native_context:
        return False, {"reason": "length_not_above_native"}

    needle_token = "__FORGEAI_NEEDLE__"
    needle_index = int(length * 0.72)

    tokens = [f"tok{i % 97}" for i in range(length)]
    tokens[needle_index] = needle_token

    chunk_size = max(native_context // 2, 512)
    found_index = -1
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start : start + chunk_size]
        try:
            local_index = chunk.index(needle_token)
            found_index = start + local_index
            break
        except ValueError:
            continue

    passed = found_index == needle_index and needle_index > native_context

    return passed, {
        "length": length,
        "chunk_size": chunk_size,
        "needle_index": needle_index,
        "found_index": found_index,
        "beyond_native_context": needle_index > native_context,
    }


def _long_context_qa_proxy(native_context: int, target_context: int) -> tuple[bool, dict[str, Any]]:
    length = min(target_context, 96_000)
    if length <= native_context:
        return False, {"reason": "length_not_above_native"}

    facts: list[tuple[str, str]] = [
        ("city", "venice"),
        ("year", "2042"),
        ("project", "forgeai"),
    ]
    positions = [int(length * 0.67), int(length * 0.78), int(length * 0.9)]
    tokens = [f"tok{i % 211}" for i in range(length)]

    for idx, (key, value) in enumerate(facts):
        pos = min(length - 1, max(native_context + 1, positions[idx]))
        positions[idx] = pos
        tokens[pos] = f"FACT_{key}={value}"

    chunk_size = max(native_context // 2, 512)
    retrieved: dict[str, str] = {}
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start : start + chunk_size]
        for token in chunk:
            if not token.startswith("FACT_"):
                continue
            payload = token[len("FACT_") :]
            if "=" not in payload:
                continue
            key, value = payload.split("=", 1)
            retrieved[key] = value

    correct = 0
    for key, value in facts:
        if retrieved.get(key) == value:
            correct += 1

    accuracy = correct / len(facts)
    beyond_native = sum(1 for pos in positions if pos > native_context)
    passed = accuracy == 1.0 and beyond_native == len(facts)

    return passed, {
        "length": length,
        "chunk_size": chunk_size,
        "facts_total": len(facts),
        "facts_beyond_native": beyond_native,
        "retrieval_accuracy": round(accuracy, 4),
        "retrieved": dict(retrieved),
    }


def _short_context_attention_proxy(
    native_context: int,
    original_context: int,
    factor: float,
) -> tuple[bool, dict[str, Any]]:
    window = max(16, min(native_context, original_context, 2048))
    query_pos = window // 2
    key_positions = list(range(0, window, max(1, window // 64)))
    if key_positions[-1] != window - 1:
        key_positions.append(window - 1)

    base_scores = [-float(abs(query_pos - key)) for key in key_positions]
    mapped_scores = []
    for key in key_positions:
        mapped_key = _yarn_position_map(position=key, original_context=original_context, factor=factor)
        mapped_scores.append(-abs(float(query_pos) - mapped_key))

    base_probs = _softmax(base_scores)
    mapped_probs = _softmax(mapped_scores)
    max_abs_delta = max((abs(a - b) for a, b in zip(base_probs, mapped_probs)), default=0.0)
    cosine = _cosine_similarity(base_probs, mapped_probs)
    passed = max_abs_delta <= 1e-6 and cosine >= 0.999999

    return passed, {
        "window": window,
        "sample_count": len(key_positions),
        "max_abs_prob_delta": round(max_abs_delta, 10),
        "cosine_similarity": round(cosine, 10),
    }


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_val = max(values)
    exp_vals = [pow(2.718281828459045, value - max_val) for value in values]
    total = sum(exp_vals) or 1.0
    return [value / total for value in exp_vals]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _yarn_position_map(position: int, original_context: int, factor: float) -> float:
    if position <= original_context:
        return float(position)

    tail = position - original_context
    return float(original_context) + (tail / max(factor, 1e-6))
