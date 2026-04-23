from __future__ import annotations

from typing import Any

from .analysis import DatasetAnalysis, analyze_dataset, detect_dominant_language
from .catalog import BaseModelCandidate, BaseModelSpec, list_catalog_models, select_compatible_models
from .recommendation import (
    ObjectiveCategory,
    RouteRecommendation,
    StrategyPreset,
    WizardPath,
    build_strategy_config,
    recommend_route,
    stratified_consent,
)
from .session import SessionStore, WizardSessionState


def run_wizard(*args: Any, **kwargs: Any) -> Any:
    from .runner import run_wizard as _run_wizard

    return _run_wizard(*args, **kwargs)

__all__ = [
    "BaseModelCandidate",
    "BaseModelSpec",
    "DatasetAnalysis",
    "ObjectiveCategory",
    "RouteRecommendation",
    "SessionStore",
    "StrategyPreset",
    "WizardPath",
    "WizardSessionState",
    "analyze_dataset",
    "build_strategy_config",
    "detect_dominant_language",
    "list_catalog_models",
    "recommend_route",
    "run_wizard",
    "select_compatible_models",
    "stratified_consent",
]
