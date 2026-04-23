from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ADAPTER_MAGIC = b"FSTRYARN"


@dataclass(frozen=True, slots=True)
class ReconstructedStretchVariant:
    variant_dir: str
    source_model_path: str
    source_model_sha256: str
    adapter_path: str
    adapter_sha256: str
    manifest_path: str
    native_context: int
    target_context: int
    final_artifact_type: str
    method: str
    adapter_positions: list[float]
    adapter_mapped_positions: list[float]

    def map_position(self, position: int) -> float:
        if position <= self.native_context:
            return float(position)

        points = self.adapter_positions
        mapped = self.adapter_mapped_positions
        if not points or not mapped:
            raise ValueError("Mappa adapter vuota.")

        pos_value = float(position)
        if pos_value <= points[0]:
            return mapped[0]
        if pos_value >= points[-1]:
            return mapped[-1]

        # Piecewise-linear interpolation over deterministic adapter table.
        for idx in range(len(points) - 1):
            x0 = points[idx]
            x1 = points[idx + 1]
            if x0 <= pos_value <= x1:
                y0 = mapped[idx]
                y1 = mapped[idx + 1]
                if x1 == x0:
                    return y0
                ratio = (pos_value - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)

        return mapped[-1]


def reconstruct_variant_from_manifest(
    manifest_path: str | Path,
) -> ReconstructedStretchVariant:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    final_artifact_type = str(manifest.get("final_artifact_type", ""))
    if final_artifact_type != "adapter_plus_manifest":
        raise ValueError(
            f"Tipo artefatto non supportato per la ricostruzione: '{final_artifact_type}'. "
            "Atteso 'adapter_plus_manifest'."
        )

    source_model_path = Path(str(manifest.get("source_model_path", "")))
    source_model_sha256 = str(manifest.get("source_model_sha256", ""))
    if not source_model_path.exists():
        raise FileNotFoundError(f"Checkpoint modello sorgente non trovato: {source_model_path}")
    if source_model_sha256 and _sha256_file(source_model_path) != source_model_sha256:
        raise ValueError("Hash sha256 del modello sorgente non corrisponde; ricostruzione annullata.")

    adapter_info = manifest.get("adapter_artifact")
    if not isinstance(adapter_info, dict):
        raise ValueError("Blocco adapter_artifact mancante nel manifest.")

    adapter_path = _resolve_adapter_path(path.parent, adapter_info)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Artefatto adapter non trovato: {adapter_path}")

    expected_adapter_sha = str(adapter_info.get("sha256", ""))
    adapter_sha = _sha256_file(adapter_path)
    if expected_adapter_sha and expected_adapter_sha != adapter_sha:
        raise ValueError("Hash sha256 dell'adapter non corrisponde; ricostruzione annullata.")

    deterministic_block = manifest.get("deterministic_reconstruction", {})
    _validate_deterministic_block(deterministic_block, source_model_sha256, adapter_sha)

    native_context = int(manifest.get("native_context", 0) or 0)
    target_context = int(manifest.get("target_context", 0) or 0)
    method = str(manifest.get("method", ""))
    if native_context <= 0 or target_context <= native_context:
        raise ValueError("Contesti nel manifest non validi per la ricostruzione.")
    if method != "yarn":
        raise ValueError(f"Metodo '{method}' non supportato per la ricostruzione v1.")

    adapter_positions, adapter_mapped_positions = _read_adapter_table(adapter_path)
    return ReconstructedStretchVariant(
        variant_dir=str(path.parent.resolve()),
        source_model_path=str(source_model_path.resolve()),
        source_model_sha256=source_model_sha256,
        adapter_path=str(adapter_path.resolve()),
        adapter_sha256=adapter_sha,
        manifest_path=str(path.resolve()),
        native_context=native_context,
        target_context=target_context,
        final_artifact_type=final_artifact_type,
        method=method,
        adapter_positions=adapter_positions,
        adapter_mapped_positions=adapter_mapped_positions,
    )


def run_minimal_reconstruction_demo(
    reconstructed: ReconstructedStretchVariant,
) -> dict[str, Any]:
    """
    Minimal 'used for real' demo:
    - builds a long synthetic sequence with distant facts
    - reconstructs mapped positions via adapter table
    - simulates fixed attention budget in mapped space
    - compares against non-stretched baseline budget
    """
    length = min(reconstructed.target_context, 131_072)
    query_position = length - 1
    native_budget = float(reconstructed.native_context)

    facts = [
        ("city", "venice"),
        ("year", "2042"),
        ("project", "forgeai"),
    ]
    raw_positions = [int(length * 0.67), int(length * 0.78), int(length * 0.9)]
    fact_positions: list[int] = []

    tokens = [f"tok{i % 113}" for i in range(length)]
    for idx, (key, value) in enumerate(facts):
        pos = min(length - 1, max(reconstructed.native_context + 1, raw_positions[idx]))
        fact_positions.append(pos)
        tokens[pos] = f"FACT_{key}={value}"

    query_mapped = reconstructed.map_position(query_position)
    retrieved_stretch: dict[str, str] = {}
    retrieved_baseline: dict[str, str] = {}

    for pos, token in enumerate(tokens):
        if not token.startswith("FACT_"):
            continue
        payload = token[len("FACT_") :]
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)

        mapped_distance = query_mapped - reconstructed.map_position(pos)
        if mapped_distance <= native_budget:
            retrieved_stretch[key] = value

        baseline_distance = float(query_position - pos)
        if baseline_distance <= native_budget:
            retrieved_baseline[key] = value

    stretch_ok = all(retrieved_stretch.get(key) == value for key, value in facts)
    baseline_ok = all(retrieved_baseline.get(key) == value for key, value in facts)

    return {
        "sequence_length": length,
        "native_context": reconstructed.native_context,
        "target_context": reconstructed.target_context,
        "fact_positions": fact_positions,
        "retrieved_with_stretch": dict(retrieved_stretch),
        "retrieved_without_stretch": dict(retrieved_baseline),
        "stretch_retrieved_all": stretch_ok,
        "baseline_retrieved_all": baseline_ok,
    }


def _validate_deterministic_block(
    deterministic_block: Any,
    source_model_sha256: str,
    adapter_sha256: str,
) -> None:
    if not isinstance(deterministic_block, dict):
        raise ValueError("Blocco deterministic_reconstruction mancante.")
    if not bool(deterministic_block.get("enabled")):
        raise ValueError("deterministic_reconstruction.enabled deve essere true.")

    required_inputs = deterministic_block.get("required_inputs")
    if not isinstance(required_inputs, dict):
        raise ValueError("deterministic_reconstruction.required_inputs mancante.")

    if str(required_inputs.get("source_model_sha256", "")) != source_model_sha256:
        raise ValueError("Hash modello sorgente non coerente in deterministic_reconstruction.")
    if str(required_inputs.get("adapter_sha256", "")) != adapter_sha256:
        raise ValueError("Hash adapter non coerente in deterministic_reconstruction.")

    steps = deterministic_block.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("deterministic_reconstruction.steps deve essere non vuoto.")


def _resolve_adapter_path(variant_dir: Path, adapter_info: dict[str, Any]) -> Path:
    raw = adapter_info.get("path")
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        return (variant_dir / candidate).resolve()
    return (variant_dir / "stretch_adapter.bin").resolve()


def _read_adapter_table(adapter_path: Path) -> tuple[list[float], list[float]]:
    with adapter_path.open("rb") as file_obj:
        header = file_obj.read(len(ADAPTER_MAGIC))
        if header != ADAPTER_MAGIC:
            raise ValueError("Header artefatto adapter non valido.")

        count_bytes = file_obj.read(4)
        if len(count_bytes) != 4:
            raise ValueError("Artefatto adapter non valido: conteggio entry mancante.")
        count = struct.unpack("<I", count_bytes)[0]
        if count == 0:
            raise ValueError("Artefatto adapter senza entry.")

        positions: list[float] = []
        mapped_positions: list[float] = []
        for _ in range(count):
            pair = file_obj.read(8)
            if len(pair) != 8:
                raise ValueError("Artefatto adapter troncato.")
            position, mapped = struct.unpack("<ff", pair)
            positions.append(float(position))
            mapped_positions.append(float(mapped))

    return positions, mapped_positions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
