#!/usr/bin/env bash
# ForgeAI — Local development startup script
# Starts forge-engine (FastAPI) and the web UI (Next.js) concurrently.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "ForgeAI — Starting local development environment"
echo ""

# ── forge-engine ──────────────────────────────────────────────────────────────

echo "Starting forge-engine on http://localhost:8000 ..."
cd "$ROOT/apps/forge-engine"

if ! python -c "import uvicorn" 2>/dev/null; then
  echo "  Installing forge-engine dependencies..."
  pip install -e ".[mlx]" -q
fi

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
ENGINE_PID=$!

# ── web UI ────────────────────────────────────────────────────────────────────

echo "Starting web UI on http://localhost:3000 ..."
cd "$ROOT/apps/web"

if [ ! -d "node_modules" ]; then
  echo "  Installing web dependencies..."
  npm install -q
fi

npm run dev &
WEB_PID=$!

# ── Cleanup ───────────────────────────────────────────────────────────────────

trap "kill $ENGINE_PID $WEB_PID 2>/dev/null; echo ''; echo 'Stopped.'" EXIT

echo ""
echo "  forge-engine : http://localhost:8000"
echo "  Web UI       : http://localhost:3000"
echo "  API docs     : http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop."

wait
