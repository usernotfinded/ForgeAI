#!/usr/bin/env bash
# ForgeAI reproducible smoke demo (CPU-friendly)
# Runs a tiny end-to-end path to verify pipeline correctness, not model quality.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Keep smoke demo deterministic and CPU-friendly even on machines with partial MLX setup.
export FORGE_DISABLE_MLX_CHECK=1

STAMP="${1:-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_DIR="$ROOT_DIR/artifacts/smoke/$STAMP"
LOG_DIR="$ARTIFACT_DIR/logs"
DATA_DIR="$ARTIFACT_DIR/data"
TOKENIZER_DIR="$ARTIFACT_DIR/tokenizer"
PROCESSED_DIR="$ARTIFACT_DIR/data_processed"
CHECKPOINT_DIR="$ARTIFACT_DIR/checkpoints"
SAMPLE_PATH="$ARTIFACT_DIR/sample.txt"
SUMMARY_PATH="$ARTIFACT_DIR/summary.md"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$PROCESSED_DIR" "$CHECKPOINT_DIR"

if command -v forge >/dev/null 2>&1; then
  FORGE_CMD=(forge)
else
  export PYTHONPATH="$ROOT_DIR/apps/forge-engine:${PYTHONPATH:-}"
  FORGE_CMD=(python -m cli.main)
fi

run_step() {
  local step_name="$1"
  shift
  echo
  echo ">>> [$step_name] $*"
  "$@" 2>&1 | tee "$LOG_DIR/${step_name}.log"
}

echo "========================================"
echo " ForgeAI Reproducible Smoke Demo"
echo " Artifact dir: $ARTIFACT_DIR"
echo "========================================"

cat <<'EOF' > "$DATA_DIR/toy_base.txt"
ForgeAI is a local-first training workbench.
This toy corpus is for pipeline verification only.
Short runs validate wiring, not model quality.
Tokenizer, sharding, planning, training, and inference should all execute.
EOF

: > "$DATA_DIR/toy.txt"
for _i in $(seq 1 120); do
  cat "$DATA_DIR/toy_base.txt" >> "$DATA_DIR/toy.txt"
done

run_step "tokenizer-train" \
  "${FORGE_CMD[@]}" tokenizer train \
  --data "$DATA_DIR" \
  --vocab-size 128 \
  --output "$TOKENIZER_DIR"

run_step "data-prepare" \
  "${FORGE_CMD[@]}" data prepare "$DATA_DIR" \
  --output "$PROCESSED_DIR" \
  --tokenizer "$TOKENIZER_DIR" \
  --context-length 128

run_step "plan" \
  "${FORGE_CMD[@]}" plan \
  --arch transformer \
  --params 50M \
  --data "$PROCESSED_DIR"

run_step "train" \
  "${FORGE_CMD[@]}" train \
  --arch transformer \
  --params 50M \
  --data "$PROCESSED_DIR" \
  --tokenizer "$TOKENIZER_DIR" \
  --output "$CHECKPOINT_DIR" \
  --batch-size 1 \
  --max-steps 3 \
  --save-every 3 \
  --val-every 1000 \
  --context-length 128

run_step "eval" \
  "${FORGE_CMD[@]}" eval "$CHECKPOINT_DIR/latest" \
  --benchmark perplexity \
  --data "$PROCESSED_DIR" \
  --max-batches 1

echo
echo ">>> [checkpoint-and-sample] load checkpoint and generate a tiny sample"
PYTHONPATH="$ROOT_DIR/apps/forge-engine:${PYTHONPATH:-}" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
TOKENIZER_DIR="$TOKENIZER_DIR" \
SAMPLE_PATH="$SAMPLE_PATH" \
python - <<'PY' 2>&1 | tee "$LOG_DIR/checkpoint-and-sample.log"
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from app.architectures import get_architecture
from app.checkpoints.manager import load_checkpoint
from app.tokenizer import load_tokenizer

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
tokenizer_dir = Path(os.environ["TOKENIZER_DIR"])
sample_path = Path(os.environ["SAMPLE_PATH"])

resolved_ckpt = (checkpoint_dir / "latest").resolve()
meta_path = resolved_ckpt / "metadata.json"
with meta_path.open("r", encoding="utf-8") as fh:
    meta = json.load(fh)

model = get_architecture(meta["architecture"], **meta["model_config"])
load_checkpoint(resolved_ckpt, model, device="cpu")
model.eval()

tokenizer = load_tokenizer(tokenizer_dir)
prompt = "ForgeAI smoke demo:"
encoded = tokenizer.encode(prompt).ids
if not encoded:
    encoded = [0]

torch.manual_seed(0)
input_ids = torch.tensor([encoded], dtype=torch.long)
generated = model.generate(
    input_ids,
    max_new_tokens=16,
    temperature=0.8,
    top_k=20,
    top_p=0.95,
)
text = tokenizer.decode(generated[0].tolist())
sample_path.write_text(text + "\n", encoding="utf-8")
print(f"Sample written to {sample_path}")
print(f"Preview: {text[:160]}")
PY

[[ -f "$CHECKPOINT_DIR/latest/model.pt" ]] || { echo "Missing checkpoint model.pt"; exit 1; }
[[ -f "$CHECKPOINT_DIR/train_log.jsonl" ]] || { echo "Missing train_log.jsonl"; exit 1; }
[[ -f "$CHECKPOINT_DIR/latest/eval_results.json" ]] || { echo "Missing eval_results.json"; exit 1; }
[[ -f "$SAMPLE_PATH" ]] || { echo "Missing sample.txt"; exit 1; }

cp "$CHECKPOINT_DIR/latest/eval_results.json" "$ARTIFACT_DIR/eval_results.json"

cat > "$SUMMARY_PATH" <<EOF
# ForgeAI Smoke Demo Summary

- timestamp: \`$STAMP\`
- artifact_dir: \`$ARTIFACT_DIR\`
- checkpoint: \`$CHECKPOINT_DIR/latest\`
- train_log: \`$CHECKPOINT_DIR/train_log.jsonl\`
- eval_results: \`$ARTIFACT_DIR/eval_results.json\`
- sample: \`$SAMPLE_PATH\`

This run validates pipeline wiring only (data -> tokenizer -> shards -> planner -> short train -> checkpoint load -> tiny generation).
It does **not** validate model quality.
EOF

echo
echo "========================================"
echo "✅ Smoke demo completed successfully"
echo "Artifacts:"
echo "  - $SUMMARY_PATH"
echo "  - $CHECKPOINT_DIR/train_log.jsonl"
echo "  - $ARTIFACT_DIR/eval_results.json"
echo "  - $SAMPLE_PATH"
echo "========================================"
