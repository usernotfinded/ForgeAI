from __future__ import annotations

import importlib
import re

from typer.testing import CliRunner

from cli.main import app

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _normalize_cli_output(value: str) -> str:
    without_ansi = ANSI_ESCAPE_RE.sub("", value)
    return " ".join(without_ansi.split())


def test_cli_module_imports() -> None:
    module = importlib.import_module("cli.main")
    assert hasattr(module, "app")


def test_cli_help_runs_without_crashing() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ForgeAI" in result.stdout


def test_train_help_mentions_hardware_advisory_policy() -> None:
    result = CliRunner().invoke(app, ["train", "--help"], color=False)
    normalized = _normalize_cli_output(result.stdout)

    assert result.exit_code == 0
    assert "Hardware feasibility checks are advisory by default" in normalized
    assert "--strict-hardware-checks" in normalized
    assert "hard failures" in normalized
