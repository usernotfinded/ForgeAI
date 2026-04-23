from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .analysis import DatasetAnalysis


class WizardPath(str, Enum):
    ADAPT_EXISTING = "adapt_existing"
    TRAIN_FROM_SCRATCH = "train_from_scratch"


class ObjectiveCategory(str, Enum):
    PERSONAL_ASSISTANT = "personal_assistant"
    DOC_ASSISTANT = "doc_assistant"
    CODE_ASSISTANT = "code_assistant"
    CLASSIFICATION = "classification"
    NEW_LANGUAGE = "new_language"
    RESEARCH = "research"
    OTHER = "other"


class StrategyPreset(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    MAX_QUALITY = "max_quality"


OBJECTIVE_LABELS: dict[ObjectiveCategory, str] = {
    ObjectiveCategory.PERSONAL_ASSISTANT: "Assistente personale privato",
    ObjectiveCategory.DOC_ASSISTANT: "Assistente su documenti interni",
    ObjectiveCategory.CODE_ASSISTANT: "Supporto alla programmazione",
    ObjectiveCategory.CLASSIFICATION: "Classificazione o estrazione testi",
    ObjectiveCategory.NEW_LANGUAGE: "Nuova lingua o dialetto",
    ObjectiveCategory.RESEARCH: "Studio e sperimentazione",
    ObjectiveCategory.OTHER: "Altro",
}


@dataclass(frozen=True, slots=True)
class RouteRecommendation:
    recommended_path: WizardPath
    confidence: float
    reasons: list[str]
    warnings: list[str]
    risk_level: int
    supported_adaptation_mode: str
    unsupported_adaptation_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_path": self.recommended_path.value,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "risk_level": self.risk_level,
            "supported_adaptation_mode": self.supported_adaptation_mode,
            "unsupported_adaptation_modes": list(self.unsupported_adaptation_modes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteRecommendation":
        path_raw = str(payload.get("recommended_path", WizardPath.ADAPT_EXISTING.value))
        path = WizardPath(path_raw) if path_raw in {item.value for item in WizardPath} else WizardPath.ADAPT_EXISTING
        return cls(
            recommended_path=path,
            confidence=float(payload.get("confidence", 0.5)),
            reasons=[str(item) for item in payload.get("reasons", [])],
            warnings=[str(item) for item in payload.get("warnings", [])],
            risk_level=int(payload.get("risk_level", 1)),
            supported_adaptation_mode=str(
                payload.get("supported_adaptation_mode", "continued_pretraining")
            ),
            unsupported_adaptation_modes=[
                str(item) for item in payload.get("unsupported_adaptation_modes", [])
            ],
        )


def recommend_route(
    objective: ObjectiveCategory,
    analysis: DatasetAnalysis,
    available_memory_gb: float | None,
    unified_memory: bool,
) -> RouteRecommendation:
    reasons: list[str] = []
    warnings: list[str] = []

    instruction_like = analysis.dataset_type in {"instructions", "conversations", "qa_pairs"}
    raw_like = analysis.dataset_type in {"raw_text", "code", "mixed"}

    tokens = analysis.estimated_tokens
    memory_gb = float(available_memory_gb or 0.0)
    strong_hardware = memory_gb >= 12.0 or (unified_memory and memory_gb >= 16.0)
    enough_tokens_for_scratch = tokens >= 25_000_000
    strong_tokens_for_scratch = tokens >= 120_000_000
    high_quality = analysis.quality_score >= 65.0

    recommended_path = WizardPath.ADAPT_EXISTING
    confidence = 0.85

    if instruction_like:
        recommended_path = WizardPath.ADAPT_EXISTING
        reasons.append(
            "I dati sembrano istruzioni/conversazioni: adattare un modello base è il modo più rapido."
        )
    elif (
        objective in {ObjectiveCategory.NEW_LANGUAGE, ObjectiveCategory.RESEARCH}
        and raw_like
        and strong_hardware
        and strong_tokens_for_scratch
        and high_quality
    ):
        recommended_path = WizardPath.TRAIN_FROM_SCRATCH
        confidence = 0.72
        reasons.append(
            "Hai dati ampi e hardware adeguato: addestrare da zero è realistico in questo scenario."
        )
    else:
        recommended_path = WizardPath.ADAPT_EXISTING
        reasons.append(
            "Per ridurre tempi e rischio, il percorso più realistico è adattare un modello esistente."
        )

    if analysis.dataset_type == "unknown":
        warnings.append("Il formato del dataset non è chiaro: conviene sistemarlo prima di investire ore di training.")

    if tokens < 5_000_000:
        warnings.append("Pochi dati utili: da zero il rischio di scarsa qualità è alto.")

    if not strong_hardware:
        warnings.append("Hardware limitato: meglio partire da adattamento leggero e iterare.")

    if recommended_path == WizardPath.TRAIN_FROM_SCRATCH and not enough_tokens_for_scratch:
        warnings.append(
            "Il volume dati è sotto la soglia consigliata per il training da zero: qualità incerta."
        )

    if objective == ObjectiveCategory.CODE_ASSISTANT and analysis.dataset_type != "code":
        warnings.append("Obiettivo codice ma dataset non chiaramente code-first: valuta un dataset più mirato.")

    risk_level = 0
    if not strong_hardware:
        risk_level += 1
    if analysis.quality_score < 55.0:
        risk_level += 1
    if tokens < 25_000_000:
        risk_level += 1

    return RouteRecommendation(
        recommended_path=recommended_path,
        confidence=round(confidence, 2),
        reasons=reasons,
        warnings=warnings,
        risk_level=min(risk_level, 3),
        supported_adaptation_mode="continued_pretraining",
        unsupported_adaptation_modes=["LoRA", "QLoRA"],
    )


def build_strategy_config(
    path: WizardPath,
    preset: StrategyPreset,
    hardware_batch_hint: int,
) -> dict[str, Any]:
    base_batch = max(1, min(hardware_batch_hint, 8))

    if path == WizardPath.ADAPT_EXISTING:
        if preset == StrategyPreset.FAST:
            return {
                "label": "Veloce",
                "max_steps": 1200,
                "learning_rate": 2e-4,
                "batch_size": base_batch,
                "grad_accum": 1,
                "val_split": 0.05,
                "save_every": 400,
                "val_every": 200,
                "gradient_checkpointing": False,
                "tradeoff": "Risultato rapido, qualità limitata su casi difficili.",
            }
        if preset == StrategyPreset.MAX_QUALITY:
            return {
                "label": "Massima Qualità",
                "max_steps": 7000,
                "learning_rate": 1.5e-4,
                "batch_size": max(1, base_batch - 1),
                "grad_accum": 4,
                "val_split": 0.08,
                "save_every": 600,
                "val_every": 150,
                "gradient_checkpointing": True,
                "tradeoff": "Qualità migliore, richiede molte più ore e più checkpoint.",
            }
        return {
            "label": "Bilanciato",
            "max_steps": 3500,
            "learning_rate": 1.8e-4,
            "batch_size": base_batch,
            "grad_accum": 2,
            "val_split": 0.06,
            "save_every": 500,
            "val_every": 200,
            "gradient_checkpointing": False,
            "tradeoff": "Buon compromesso tra qualità e tempi.",
        }

    # Train from scratch
    if preset == StrategyPreset.FAST:
        return {
            "label": "Veloce",
            "model_preset": "forge-nano",
            "max_steps": 4000,
            "learning_rate": 3e-4,
            "batch_size": base_batch,
            "grad_accum": 1,
            "val_split": 0.05,
            "save_every": 500,
            "val_every": 250,
            "gradient_checkpointing": False,
            "tradeoff": "Setup didattico: tempi contenuti ma capacità limitata.",
        }
    if preset == StrategyPreset.MAX_QUALITY:
        return {
            "label": "Massima Qualità",
            "model_preset": "forge-small",
            "max_steps": 26000,
            "learning_rate": 2.4e-4,
            "batch_size": max(1, base_batch - 1),
            "grad_accum": 4,
            "val_split": 0.08,
            "save_every": 1000,
            "val_every": 250,
            "gradient_checkpointing": True,
            "tradeoff": "Più qualità, ma costo e durata molto più alti.",
        }
    return {
        "label": "Bilanciato",
        "model_preset": "forge-tiny",
        "max_steps": 12000,
        "learning_rate": 2.8e-4,
        "batch_size": base_batch,
        "grad_accum": 2,
        "val_split": 0.06,
        "save_every": 800,
        "val_every": 250,
        "gradient_checkpointing": False,
        "tradeoff": "Compromesso realistico per qualità/costi da zero.",
    }


def stratified_consent(
    chosen_path: WizardPath,
    recommendation: RouteRecommendation,
    analysis: DatasetAnalysis,
    available_memory_gb: float | None,
) -> tuple[int, list[str]]:
    """
    Returns `(max_level, messages)`.

    Level 1: consiglio base.
    Level 2: warning forte.
    Level 3: override consapevole necessario.
    """
    messages: list[str] = []
    level = 1

    if chosen_path != recommendation.recommended_path:
        level = max(level, 2)
        messages.append("Hai scelto un percorso diverso da quello consigliato automaticamente.")

    memory_gb = float(available_memory_gb or 0.0)
    if chosen_path == WizardPath.TRAIN_FROM_SCRATCH and memory_gb < 10.0:
        level = max(level, 3)
        messages.append("Training da zero con memoria ridotta: rischio alto di tempi lunghi o blocchi.")

    if chosen_path == WizardPath.TRAIN_FROM_SCRATCH and analysis.estimated_tokens < 20_000_000:
        level = max(level, 3)
        messages.append("Token stimati bassi per training da zero: rischio qualità insufficiente.")

    if recommendation.risk_level >= 2:
        level = max(level, 2)
        messages.append("Analisi rischio: il setup è fragile, meglio procedere con attenzione.")

    if chosen_path == WizardPath.ADAPT_EXISTING and recommendation.unsupported_adaptation_modes:
        level = max(level, 2)
        modes = ", ".join(recommendation.unsupported_adaptation_modes)
        messages.append(
            f"Modalità non disponibili in v1 ({modes}): verrà usata l'alternativa supportata oggi."
        )

    return level, messages
