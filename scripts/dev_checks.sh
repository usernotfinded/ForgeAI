#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Duplicate-file guard"
./scripts/check_no_duplicate_suffix.sh

echo "[2/5] Ruff"
ruff check .

echo "[3/5] Pytest (forge-engine)"
pytest apps/forge-engine/tests -q

echo "[4/5] CLI help smoke"
if command -v forge >/dev/null 2>&1; then
  forge --help >/dev/null
  forge train --help >/dev/null
  forge stretch --help >/dev/null
else
  export PYTHONPATH="$ROOT_DIR/apps/forge-engine:${PYTHONPATH:-}"
  python -m cli.main --help >/dev/null
  python -m cli.main train --help >/dev/null
  python -m cli.main stretch --help >/dev/null
fi

echo "[5/5] Import smoke"
python - <<'PY'
import importlib

for module_name in (
    "cli.main",
    "app.training.planner",
    "app.wizard.analysis",
    "app.stretch.runner",
):
    importlib.import_module(module_name)

print("Import checks passed.")
PY

echo "All development checks passed."
