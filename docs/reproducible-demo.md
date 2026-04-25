# ForgeAI Reproducible Demo (CPU-Friendly)

This document describes a tiny, reproducible end-to-end demo that verifies ForgeAI pipeline wiring on local hardware (including CPU-only machines).

## One-command run

From repository root:

```bash
./scripts/smoke_test.sh
```

Optional deterministic artifact folder name:

```bash
./scripts/smoke_test.sh 20260425-1200
```

The script exports `FORGE_DISABLE_MLX_CHECK=1` to avoid optional MLX probing issues on machines where MLX runtime detection is unstable. This keeps the demo path CPU-friendly and deterministic.

## What this demo runs

The script executes the minimum useful path:

1. create tiny toy text corpus
2. train tokenizer
3. prepare token shards
4. run planner
5. run a very short training loop (`max-steps=3`)
6. run a tiny eval (`perplexity`, `max-batches=1`)
7. load checkpoint and generate a short sample to `sample.txt`

No external model download is used.

## Hardware requirements

- Python 3.11+
- `forge-engine` installed (`pip install -e ./apps/forge-engine`)
- CPU is enough (GPU/MLX are not required)
- Typical run time: short (intended for smoke checks)

## Artifact convention

Each run writes outputs under:

```text
artifacts/smoke/YYYYMMDD-HHMMSS/
```

Main files:

- `logs/*.log`: step logs
- `checkpoints/train_log.jsonl`: training metrics log
- `checkpoints/latest/`: latest checkpoint
- `eval_results.json`: copied from checkpoint eval output
- `sample.txt`: tiny generated sample from loaded checkpoint
- `summary.md`: run summary and key paths

## Expected success signals

- Script exits with status `0`
- Final banner shows `Smoke demo completed successfully`
- The following files exist:
  - `checkpoints/latest/model.pt`
  - `checkpoints/train_log.jsonl`
  - `eval_results.json`
  - `sample.txt`

## What this proves

- Core CLI pipeline executes end-to-end.
- Checkpoint save/load works.
- Inference generation path can run from produced checkpoint.
- Basic reproducibility and artifacts are available for debugging.

## What this does **not** prove

- Real model quality
- Convergence quality
- Benchmark competitiveness
- Production readiness on large datasets/models

This is a wiring/integration smoke demo, not a model-performance benchmark.
