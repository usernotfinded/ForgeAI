from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class WizardSessionState:
    """Persistent state for `forge wizard` sessions."""

    version: int = 1
    session_id: str = "wizard-session"
    current_step: int = 1
    completed_steps: list[int] = field(default_factory=list)
    answers: dict[str, Any] = field(default_factory=dict)
    data_analysis: dict[str, Any] | None = None
    hardware: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    selected_path: str | None = None
    selected_base_model: str | None = None
    preset: str | None = None
    generated_config: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=lambda: {"status": "idle"})
    updated_at: str = field(default_factory=_utc_now_iso)

    def mark_step_completed(self, step: int) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
            self.completed_steps.sort()
        if self.current_step <= step:
            self.current_step = step + 1
        self.updated_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        self.updated_at = _utc_now_iso()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WizardSessionState":
        state = cls()
        for key, value in payload.items():
            if hasattr(state, key):
                setattr(state, key, value)

        if not isinstance(state.completed_steps, list):
            state.completed_steps = []
        if not isinstance(state.answers, dict):
            state.answers = {}
        if not isinstance(state.artifacts, list):
            state.artifacts = []
        if not isinstance(state.execution, dict):
            state.execution = {"status": "idle"}

        return state


class SessionStore:
    """Load/save/reset wizard state and artifact paths."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.state_path = self.session_dir / "wizard_state.json"
        self.artifact_dir = self.session_dir / "artifacts"
        self.generated_config_path = self.artifact_dir / "generated_config.json"
        self.summary_path = self.artifact_dir / "final_summary.md"
        self.execution_plan_path = self.artifact_dir / "execution_plan.sh"

    def ensure_dirs(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> WizardSessionState | None:
        if not self.state_path.exists():
            return None

        with self.state_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return WizardSessionState.from_dict(payload)

    def save(self, state: WizardSessionState) -> None:
        self.ensure_dirs()
        with self.state_path.open("w", encoding="utf-8") as file_obj:
            json.dump(state.to_dict(), file_obj, indent=2, ensure_ascii=True)

    def reset(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    def read_summary(self) -> str | None:
        if not self.summary_path.exists():
            return None
        return self.summary_path.read_text(encoding="utf-8")
