from __future__ import annotations

import importlib

from typer.testing import CliRunner

from cli.main import app


def test_cli_module_imports() -> None:
    module = importlib.import_module("cli.main")
    assert hasattr(module, "app")


def test_cli_help_runs_without_crashing() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ForgeAI" in result.stdout
