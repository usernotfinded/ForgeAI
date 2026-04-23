from __future__ import annotations

from pathlib import Path

from app.stretch.session import StretchSessionState, StretchSessionStore


def test_stretch_session_store_roundtrip(tmp_path: Path) -> None:
    store = StretchSessionStore(tmp_path)
    state = StretchSessionState(session_id="stretch-test")
    state.model_path = "/tmp/model"
    state.native_context = 32768
    state.target_context = 65536
    state.add_log("created")
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert loaded.session_id == "stretch-test"
    assert loaded.target_context == 65536
    assert loaded.logs


def test_stretch_session_mark_step_advances_current_step() -> None:
    state = StretchSessionState(session_id="stretch-steps")
    state.mark_step_completed(1)
    state.mark_step_completed(4)

    assert state.current_step == 5
    assert state.completed_steps == [1, 4]
