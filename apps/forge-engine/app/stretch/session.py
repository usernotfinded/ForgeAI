from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class StretchSessionState:
    version: int = 1
    session_id: str = "stretch-session"
    current_step: int = 1
    completed_steps: list[int] = field(default_factory=list)
    model_path: str | None = None
    model_type: str | None = None
    architecture: str | None = None
    native_context: int | None = None
    target_context: int | None = None
    aggressiveness: str | None = None
    process_status: str = "idle"
    logs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    produced_outputs: list[str] = field(default_factory=list)
    generated_config: dict[str, Any] | None = None
    compatibility: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    updated_at: str = field(default_factory=_utc_now_iso)

    def add_log(self, message: str) -> None:
        timestamp = _utc_now_iso()
        self.logs.append(f"[{timestamp}] {message}")
        self.updated_at = timestamp

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
    def from_dict(cls, payload: dict[str, Any]) -> "StretchSessionState":
        state = cls()
        for key, value in payload.items():
            if hasattr(state, key):
                setattr(state, key, value)

        if not isinstance(state.completed_steps, list):
            state.completed_steps = []
        if not isinstance(state.logs, list):
            state.logs = []
        if not isinstance(state.expected_outputs, list):
            state.expected_outputs = []
        if not isinstance(state.produced_outputs, list):
            state.produced_outputs = []
        return state


class StretchSessionStore:
    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.state_path = self.session_dir / "stretch_state.json"
        self.artifacts_dir = self.session_dir / "artifacts"
        self.generated_config_path = self.artifacts_dir / "stretch_config.json"
        self.final_report_path = self.artifacts_dir / "stretch_report.md"
        self.summary_path = self.artifacts_dir / "latest_summary.md"

    def ensure_dirs(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> StretchSessionState | None:
        if not self.state_path.exists():
            return None
        with self.state_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return StretchSessionState.from_dict(payload)

    def save(self, state: StretchSessionState) -> None:
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
