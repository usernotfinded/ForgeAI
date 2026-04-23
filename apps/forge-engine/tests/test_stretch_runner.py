from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from app.stretch.planner import AggressivenessProfile, build_stretch_plan
from app.stretch.runner import _build_final_report, _step_5_review_consent
from app.stretch.session import StretchSessionState, StretchSessionStore


def _build_state_for_level3() -> StretchSessionState:
    native = 32768
    target = 262144  # ratio 8x -> level 3 override path
    plan = build_stretch_plan(
        native_context=native,
        target_context=target,
        profile=AggressivenessProfile.AMBITIOUS,
    )

    state = StretchSessionState(session_id="stretch-level3")
    state.model_path = "/tmp/model"
    state.model_type = "base_supported"
    state.architecture = "transformer"
    state.native_context = native
    state.target_context = target
    state.aggressiveness = AggressivenessProfile.AMBITIOUS.value
    state.plan = plan.to_dict()
    state.compatibility = {
        "is_supported": True,
        "backend": "cuda",
        "method": "yarn",
        "errors": [],
        "warnings": [],
        "valid_targets": [65536, 131072, 262144],
        "recommended_target": 131072,
        "prudent_target": 65536,
        "ambitious_target": 262144,
        "max_realistic_target": 262144,
    }
    return state


def test_step5_level3_rejects_when_override_phrase_is_not_confirmed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = _build_state_for_level3()
    store = StretchSessionStore(tmp_path / "session")
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)

    monkeypatch.setattr("app.stretch.runner.typer.prompt", lambda *args, **kwargs: "NO")

    accepted = _step_5_review_consent(
        console=console,
        state=state,
        store=store,
        output_dir=str(tmp_path / "outputs"),
        non_interactive=False,
    )

    assert accepted is False
    assert state.process_status == "stopped"


def test_step5_level3_accepts_when_override_phrase_is_confirmed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = _build_state_for_level3()
    store = StretchSessionStore(tmp_path / "session")
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)

    monkeypatch.setattr(
        "app.stretch.runner.typer.prompt",
        lambda *args, **kwargs: "PROCEDI COMUNQUE",
    )
    monkeypatch.setattr("app.stretch.runner.typer.confirm", lambda *args, **kwargs: True)

    accepted = _step_5_review_consent(
        console=console,
        state=state,
        store=store,
        output_dir=str(tmp_path / "outputs"),
        non_interactive=False,
    )

    assert accepted is True
    assert state.generated_config is not None
    assert state.current_step >= 6


def test_final_report_explicitly_states_limits_and_breakdown(tmp_path: Path) -> None:
    state = StretchSessionState(session_id="stretch-report")
    state.model_path = "/tmp/source-model"
    state.model_type = "base_supported"
    state.architecture = "transformer"
    state.native_context = 32768
    state.target_context = 65536
    state.aggressiveness = "balanced"
    state.process_status = "completed"
    state.generated_config = {
        "variant": {
            "final_artifact_type": "adapter_plus_manifest",
            "variant_dir": "/tmp/source-model-64k-yarn",
        }
    }
    state.validation = {
        "structural_check": {"passed": True},
        "persistence_check": {"passed": True},
        "reconstruction_check": {"passed": True},
        "short_context_check": {"passed": True},
        "long_context_check": {"passed": False},
    }
    state.produced_outputs = ["/tmp/source-model-64k-yarn/stretch_manifest.json"]

    store = StretchSessionStore(tmp_path / "session")
    report = _build_final_report(state, store)

    assert "Tipo persistenza finale: `adapter_plus_manifest`" in report
    assert "## Esito Validazioni" in report
    assert "Controllo ricostruzione variante" in report
    assert "validazione è locale/proxy" in report.lower()
