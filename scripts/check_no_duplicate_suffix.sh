#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

duplicates="$(find . -type f \
  -not -path './.git/*' \
  -not -path './.venv/*' \
  -not -path './node_modules/*' \
  -not -path './apps/web/node_modules/*' \
  -not -path '*/__pycache__/*' \
  -name '* 2*' | sort)"

if [[ -n "$duplicates" ]]; then
  echo "Found accidental duplicate files matching '* 2*':"
  echo "$duplicates"
  echo
  echo "Remove or rename these files before committing."
  exit 1
fi

echo "No accidental '* 2*' duplicate files found."
