from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaseModelSpec:
    name: str
    hf_id: str
    params: str
    min_memory_gb: float
    preferred_memory_gb: float
    strength: str
    dominant_language: str
    license: str


@dataclass(frozen=True, slots=True)
class BaseModelCandidate:
    name: str
    hf_id: str
    params: str
    min_memory_gb: float
    preferred_memory_gb: float
    strength: str
    dominant_language: str
    license: str
    recommendation_level: str


LOCAL_BASE_MODEL_CATALOG: tuple[BaseModelSpec, ...] = (
    BaseModelSpec(
        name="smollm-135m",
        hf_id="HuggingFaceTB/SmolLM-135M",
        params="135M",
        min_memory_gb=4.0,
        preferred_memory_gb=6.0,
        strength="Molto leggero: rapido da adattare e ottimo per primi test.",
        dominant_language="Prevalenza inglese, supporto multilingua base.",
        license="Apache 2.0",
    ),
    BaseModelSpec(
        name="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        params="494M",
        min_memory_gb=8.0,
        preferred_memory_gb=12.0,
        strength="Buon equilibrio qualità/costi e ottima robustezza generale.",
        dominant_language="Multilingua forte (EN/ZH + altre lingue).",
        license="Apache 2.0",
    ),
    BaseModelSpec(
        name="tinyllama-1b",
        hf_id="TinyLlama/TinyLlama_v1.1",
        params="1.1B",
        min_memory_gb=12.0,
        preferred_memory_gb=16.0,
        strength="Più capace su compiti complessi rispetto ai modelli più piccoli.",
        dominant_language="Prevalenza inglese.",
        license="Apache 2.0",
    ),
)


def list_catalog_models() -> list[BaseModelSpec]:
    return list(LOCAL_BASE_MODEL_CATALOG)


def get_model_spec(name: str) -> BaseModelSpec | None:
    for model in LOCAL_BASE_MODEL_CATALOG:
        if model.name == name:
            return model
    return None


def select_compatible_models(
    available_memory_gb: float | None,
    limit: int = 5,
) -> list[BaseModelCandidate]:
    """
    Return base model candidates using only the local catalog.

    Memory compatibility is advisory. Low-memory candidates are still returned
    so the wizard can warn without blocking solely on estimated hardware.
    """
    memory_gb = float(available_memory_gb or 0.0)
    ranked_candidates: list[tuple[int, float, BaseModelCandidate]] = []

    for model in LOCAL_BASE_MODEL_CATALOG:
        if memory_gb >= model.preferred_memory_gb:
            recommendation_level = "alta"
            recommendation_score = 2
        elif memory_gb >= model.min_memory_gb + 2.0:
            recommendation_level = "media"
            recommendation_score = 1
        elif memory_gb >= model.min_memory_gb:
            recommendation_level = "bassa"
            recommendation_score = 0
        else:
            recommendation_level = "non consigliata"
            recommendation_score = -1

        candidate = BaseModelCandidate(
            name=model.name,
            hf_id=model.hf_id,
            params=model.params,
            min_memory_gb=model.min_memory_gb,
            preferred_memory_gb=model.preferred_memory_gb,
            strength=model.strength,
            dominant_language=model.dominant_language,
            license=model.license,
            recommendation_level=recommendation_level,
        )
        # Keep score outside dataclass to avoid leaking extra fields in artifacts.
        ranked_candidates.append((recommendation_score, model.min_memory_gb, candidate))

    ranked_candidates.sort(key=lambda item: (-item[0], item[1]))
    ordered = [candidate for _, _, candidate in ranked_candidates]
    return ordered[:limit]
