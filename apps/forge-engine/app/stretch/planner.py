from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .model_inspector import ModelInspection, ModelType
from .registry import StretchRecipe, get_stretch_recipe


class AggressivenessProfile(str, Enum):
    PRUDENT = "prudent"
    BALANCED = "balanced"
    AMBITIOUS = "ambitious"


@dataclass(frozen=True, slots=True)
class StretchCompatibility:
    is_supported: bool
    backend: str
    method: str | None
    errors: list[str]
    warnings: list[str]
    valid_targets: list[int]
    recommended_target: int | None
    prudent_target: int | None
    ambitious_target: int | None
    max_realistic_target: int | None
    recipe: StretchRecipe | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_supported": self.is_supported,
            "backend": self.backend,
            "method": self.method,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "valid_targets": list(self.valid_targets),
            "recommended_target": self.recommended_target,
            "prudent_target": self.prudent_target,
            "ambitious_target": self.ambitious_target,
            "max_realistic_target": self.max_realistic_target,
            "recipe": self.recipe.architecture if self.recipe else None,
        }


@dataclass(frozen=True, slots=True)
class TargetValidation:
    is_valid: bool
    reason: str | None
    suggested_targets: list[int]


@dataclass(frozen=True, slots=True)
class StretchPlan:
    method: str
    profile: AggressivenessProfile
    native_context: int
    target_context: int
    context_ratio: float
    yarn_config: dict[str, Any]
    estimated_cost_multiplier: float
    estimated_time_multiplier: float
    risk_level: int
    risk_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "profile": self.profile.value,
            "native_context": self.native_context,
            "target_context": self.target_context,
            "context_ratio": self.context_ratio,
            "yarn_config": dict(self.yarn_config),
            "estimated_cost_multiplier": self.estimated_cost_multiplier,
            "estimated_time_multiplier": self.estimated_time_multiplier,
            "risk_level": self.risk_level,
            "risk_notes": list(self.risk_notes),
        }


def analyze_compatibility(
    inspection: ModelInspection,
    backend: str,
    vram_gb: float | None,
    unified_memory: bool,
) -> StretchCompatibility:
    errors: list[str] = []
    warnings = list(inspection.warnings)

    recipe = get_stretch_recipe(inspection.architecture)
    if recipe is None:
        errors.append(
            f"Nessuna ricetta stretch locale trovata per l'architettura '{inspection.architecture}'."
        )

    if not inspection.rope_based:
        errors.append("Il modello non risulta RoPE-based secondo i metadati locali.")

    if inspection.model_type == ModelType.ADAPTER_SEPARATED:
        errors.append("Lo stato con adapter separati non è compatibile con stretch v1.")

    if recipe is not None and backend not in recipe.supported_backends:
        errors.append(
            f"YaRN non è supportato sul backend '{backend}' in stretch v1. "
            f"Backend supportati: {', '.join(recipe.supported_backends)}."
        )

    native = inspection.native_context
    if recipe is not None and native < recipe.min_native_context:
        errors.append(
            f"Il contesto nativo ({native}) è sotto il minimo supportato ({recipe.min_native_context})."
        )

    max_realistic_target = None
    valid_targets: list[int] = []

    if recipe is not None and not errors:
        max_realistic_target = _estimate_realistic_target_limit(
            native_context=native,
            vram_gb=vram_gb,
            unified_memory=unified_memory,
            recipe_max=recipe.max_target_context,
        )
        valid_targets = generate_valid_targets(
            native_context=native,
            recipe=recipe,
            max_realistic_target=max_realistic_target,
        )
        if not valid_targets:
            errors.append(
                "Nessun target context realistico disponibile su questo hardware per stretch v1."
            )

    prudent_target = valid_targets[0] if valid_targets else None
    recommended_target = _select_balanced_target(valid_targets)
    ambitious_target = valid_targets[-1] if valid_targets else None

    if inspection.model_type == ModelType.ALREADY_STRETCHED:
        warnings.append("Questo modello contiene già metadati stretch. Valuta il target in modo conservativo.")
    if inspection.model_type == ModelType.ADAPTED:
        warnings.append(
            "Il modello risulta già adattato/fine-tuned. Lo stretch è possibile, ma va validato con attenzione."
        )

    return StretchCompatibility(
        is_supported=not errors,
        backend=backend,
        method=recipe.method if recipe else None,
        errors=errors,
        warnings=warnings,
        valid_targets=valid_targets,
        recommended_target=recommended_target,
        prudent_target=prudent_target,
        ambitious_target=ambitious_target,
        max_realistic_target=max_realistic_target,
        recipe=recipe,
    )


def generate_valid_targets(
    native_context: int,
    recipe: StretchRecipe,
    max_realistic_target: int,
) -> list[int]:
    candidates: list[int] = []
    for multiplier in recipe.target_multipliers:
        target = native_context * multiplier
        if target <= native_context:
            continue
        if target > recipe.max_target_context:
            continue
        if target > max_realistic_target:
            continue
        candidates.append(target)

    if not candidates:
        return []

    # Ensure deterministic ordering and uniqueness.
    return sorted(set(candidates))


def validate_target_context(
    native_context: int,
    target_context: int,
    valid_targets: list[int],
) -> TargetValidation:
    if target_context <= native_context:
        return TargetValidation(
            is_valid=False,
            reason=(
                f"Il target context deve essere strettamente maggiore del contesto nativo ({native_context})."
            ),
            suggested_targets=valid_targets[:3],
        )

    if target_context not in valid_targets:
        return TargetValidation(
            is_valid=False,
            reason=(
                f"Il target context {target_context} non è nella lista dei target realistici supportati per questa configurazione."
            ),
            suggested_targets=valid_targets[:3],
        )

    return TargetValidation(is_valid=True, reason=None, suggested_targets=[])


def build_stretch_plan(
    native_context: int,
    target_context: int,
    profile: AggressivenessProfile,
) -> StretchPlan:
    ratio = target_context / max(native_context, 1)

    safety_margin: float
    attention_factor: float
    risk_bias: int
    profile_note: str

    if profile == AggressivenessProfile.PRUDENT:
        safety_margin = 0.35
        attention_factor = 1.0
        risk_bias = 0
        profile_note = "Rischio minore, estensione moderata."
    elif profile == AggressivenessProfile.BALANCED:
        safety_margin = 0.22
        attention_factor = 1.08
        risk_bias = 1
        profile_note = "Compromesso bilanciato tra affidabilità e contesto esteso."
    else:
        safety_margin = 0.12
        attention_factor = 1.16
        risk_bias = 2
        profile_note = "Target massimo con rischio maggiore di drift qualitativo."

    estimated_cost_multiplier = round(max(1.0, ratio ** 1.08), 2)
    estimated_time_multiplier = round(max(1.0, ratio ** 1.2), 2)

    base_risk = 0
    if ratio >= 4.0:
        base_risk += 1
    if ratio >= 8.0:
        base_risk += 1

    risk_level = min(3, base_risk + risk_bias)

    risk_notes = [
        "Un contesto più lungo non garantisce automaticamente qualità migliore.",
        "Target più alti possono aumentare molto latenza e uso memoria.",
        "La validazione è necessaria per individuare regressioni sul contesto corto.",
        profile_note,
    ]

    yarn_config = {
        "type": "yarn",
        "factor": round(ratio, 4),
        "original_max_position_embeddings": native_context,
        "target_max_position_embeddings": target_context,
        "safety_margin": safety_margin,
        "attention_factor": attention_factor,
    }

    return StretchPlan(
        method="yarn",
        profile=profile,
        native_context=native_context,
        target_context=target_context,
        context_ratio=round(ratio, 4),
        yarn_config=yarn_config,
        estimated_cost_multiplier=estimated_cost_multiplier,
        estimated_time_multiplier=estimated_time_multiplier,
        risk_level=risk_level,
        risk_notes=risk_notes,
    )


def stratified_consent_level(plan: StretchPlan, compatibility: StretchCompatibility) -> tuple[int, list[str]]:
    level = 1
    messages: list[str] = []

    if compatibility.warnings:
        level = max(level, 2)
        messages.extend(compatibility.warnings)

    if plan.risk_level >= 2:
        level = max(level, 2)
        messages.append("Il profilo stretch selezionato ha rischio alto di drift qualitativo.")

    if plan.risk_level >= 3 or plan.context_ratio >= 8.0:
        level = max(level, 3)
        messages.append(
            "Il target è molto aggressivo per la v1; serve conferma esplicita di override."
        )

    return level, messages


def _estimate_realistic_target_limit(
    native_context: int,
    vram_gb: float | None,
    unified_memory: bool,
    recipe_max: int,
) -> int:
    memory = float(vram_gb or 0.0)

    if memory <= 0:
        # Unknown memory: keep conservative.
        multiplier = 2
    elif memory < 8.0:
        multiplier = 2
    elif memory < 16.0:
        multiplier = 4
    elif memory < 32.0:
        multiplier = 8 if unified_memory else 4
    else:
        multiplier = 8

    return min(recipe_max, native_context * multiplier)


def _select_balanced_target(valid_targets: list[int]) -> int | None:
    if not valid_targets:
        return None
    if len(valid_targets) == 1:
        return valid_targets[0]

    # Prefer middle option when available; else first realistic target.
    mid = len(valid_targets) // 2
    return valid_targets[mid]
