# ForgeAI Status

This file describes the **current, verified** repository state.
It is intentionally conservative.

## What works today

- CLI-first workflows in `forge-engine`:
  - `forge plan`
  - `forge tokenizer train`
  - `forge data prepare`
  - `forge train`
  - `forge eval`
  - `forge eval-compare`
  - `forge chat`
  - `forge model pull`
  - `forge wizard`
  - `forge stretch` (v1)
- Local training pipeline for small models using **PyTorch**:
  - CUDA path
  - MPS path on Apple Silicon
  - CPU fallback path (slow, but useful for smoke tests)
- Checkpoint save/load/resume.
- Basic evaluation flows (perplexity and lightweight benchmark helpers).
- Stretch v1 with:
  - YaRN-only strategy
  - strict target-context validation (`target > native`)
  - persistent output mode: `adapter_plus_manifest` (source checkpoint copy + YaRN mapping artifact + deterministic manifest)
  - deterministic reconstruction path from manifest
- CPU-friendly reproducible smoke demo:
  - `./scripts/smoke_test.sh`
- Lightweight CI and local dev checks:
  - `.github/workflows/ci.yml`
  - `./scripts/dev_checks.sh`

## Experimental

- Web app (`apps/web`) for local monitoring and workflow support.
- Some benchmark/eval flows are intentionally lightweight and best used for regression tracking, not broad capability claims.
- `forge wizard` and `forge stretch` UX/resume/reporting are functional but still evolving.
- `forge wizard` adaptation path is intentionally narrow in v1 (compatible ForgeAI-native checkpoints only).

## Planned

- Native MLX training backend (project is Apple Silicon and MLX oriented, but current training loop is still PyTorch-based).
- Broader architecture support beyond the current Transformer baseline.
- Deeper evaluation coverage for long-context and task-level quality.
- Stronger web UI coverage across all CLI workflows.

## Not yet implemented

- Full multi-node distributed training.
- Production-grade fault-tolerant orchestration.
- Full-checkpoint persistence mode for stretch v1 (`full_checkpoint` is not available yet).
- Universal model compatibility for every architecture/checkpoint format.
- Arbitrary external-model adaptation in wizard v1.

## Known limitations

- Native MLX training is not complete; when `mlx` is detected without usable MPS training path, training falls back to CPU.
- Hardware and model-size guidance is heuristic. Hardware feasibility checks are advisory by default. Use --strict-hardware-checks to turn warnings into hard failures.
- Smoke/demo flow validates pipeline wiring, not model quality.
- Evaluation in v1 is useful for checks/regressions but not a substitute for product-grade benchmark suites.
- Web app is still MVP/experimental; routes are intentionally limited and frontend tests require local npm install (no committed lockfile yet).

## Tested environments

- **Verified in CI**:
  - Ubuntu (`ubuntu-latest`)
  - Python 3.11
  - CPU-only checks (no CUDA/MLX required)
- **Actively targeted but not CI-verified in this repository**:
  - Apple Silicon local runs via PyTorch MPS
  - CUDA local runs on contributor hardware

## Verification commands

```bash
./scripts/dev_checks.sh
./scripts/smoke_test.sh
```
