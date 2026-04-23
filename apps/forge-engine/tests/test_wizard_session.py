from __future__ import annotations

import json
from pathlib import Path

from app.wizard.session import SessionStore, WizardSessionState


def test_session_store_save_and_load(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    state = WizardSessionState(session_id="wizard-test")
    state.answers["objective"] = "research"
    store.save(state)

    loaded = store.load()

    assert loaded is not None
    assert loaded.session_id == "wizard-test"
    assert loaded.answers["objective"] == "research"


def test_session_reset_removes_state_file(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.save(WizardSessionState(session_id="wizard-reset"))
    assert store.state_path.exists()

    store.reset()

    assert not store.state_path.exists()


def test_mark_step_completed_advances_current_step() -> None:
    state = WizardSessionState(session_id="wizard-steps")

    state.mark_step_completed(1)
    state.mark_step_completed(3)

    assert state.completed_steps == [1, 3]
    assert state.current_step == 4


def test_state_file_contains_expected_keys(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    state = WizardSessionState(session_id="wizard-keys")
    state.data_analysis = {"dataset_type": "raw_text"}
    state.hardware = {"backend": "cpu"}
    state.recommendation = {"recommended_path": "adapt_existing"}
    store.save(state)

    payload = json.loads(store.state_path.read_text(encoding="utf-8"))

    required = {
        "current_step",
        "answers",
        "data_analysis",
        "hardware",
        "recommendation",
        "selected_path",
        "preset",
    }
    assert required.issubset(payload.keys())
