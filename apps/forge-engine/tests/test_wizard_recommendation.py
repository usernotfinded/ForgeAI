from __future__ import annotations

from app.wizard.analysis import DatasetAnalysis
from app.wizard.catalog import select_compatible_models
from app.wizard.recommendation import (
    ObjectiveCategory,
    RouteRecommendation,
    StrategyPreset,
    WizardPath,
    build_strategy_config,
    recommend_route,
    stratified_consent,
)


def _analysis(
    dataset_type: str,
    tokens: int,
    quality: float,
) -> DatasetAnalysis:
    return DatasetAnalysis(
        data_path="/tmp/data",
        files_scanned=1,
        documents_scanned=100,
        estimated_tokens=tokens,
        duplicate_ratio=0.01,
        short_doc_ratio=0.05,
        dominant_language="english",
        dataset_type=dataset_type,
        quality_score=quality,
        quality_notes=["ok"],
    )


def test_recommend_adapt_for_instruction_like_data() -> None:
    rec = recommend_route(
        objective=ObjectiveCategory.DOC_ASSISTANT,
        analysis=_analysis("instructions", 12_000_000, 72.0),
        available_memory_gb=16.0,
        unified_memory=False,
    )

    assert rec.recommended_path == WizardPath.ADAPT_EXISTING


def test_recommend_scratch_for_large_clean_raw_dataset() -> None:
    rec = recommend_route(
        objective=ObjectiveCategory.RESEARCH,
        analysis=_analysis("raw_text", 180_000_000, 78.0),
        available_memory_gb=24.0,
        unified_memory=False,
    )

    assert rec.recommended_path == WizardPath.TRAIN_FROM_SCRATCH


def test_adds_warning_for_unknown_dataset() -> None:
    rec = recommend_route(
        objective=ObjectiveCategory.OTHER,
        analysis=_analysis("unknown", 8_000_000, 52.0),
        available_memory_gb=8.0,
        unified_memory=False,
    )

    assert any("formato" in warning.lower() for warning in rec.warnings)


def test_build_strategy_config_fast_adapt() -> None:
    cfg = build_strategy_config(
        path=WizardPath.ADAPT_EXISTING,
        preset=StrategyPreset.FAST,
        hardware_batch_hint=4,
    )

    assert cfg["max_steps"] == 1200
    assert cfg["batch_size"] >= 1


def test_stratified_consent_requires_level_three_for_risky_scratch() -> None:
    rec = RouteRecommendation(
        recommended_path=WizardPath.ADAPT_EXISTING,
        confidence=0.9,
        reasons=["fallback"],
        warnings=["rischio"],
        risk_level=2,
        supported_adaptation_mode="continued_pretraining",
        unsupported_adaptation_modes=["LoRA"],
    )

    level, messages = stratified_consent(
        chosen_path=WizardPath.TRAIN_FROM_SCRATCH,
        recommendation=rec,
        analysis=_analysis("raw_text", 5_000_000, 50.0),
        available_memory_gb=8.0,
    )

    assert level == 3
    assert len(messages) >= 2


def test_select_compatible_models_orders_by_recommendation() -> None:
    models = select_compatible_models(available_memory_gb=16.0)

    assert len(models) >= 2
    assert models[0].recommendation_level in {"alta", "media"}


def test_select_compatible_models_warns_instead_of_filtering_when_memory_too_low() -> None:
    models = select_compatible_models(available_memory_gb=1.0)
    assert models
    assert all(model.recommendation_level == "non consigliata" for model in models)
