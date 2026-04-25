# ForgeAI Architecture (Current)

This document describes the **current repository architecture**.
It does not describe future design as if already implemented.

## High-level layout

- `apps/forge-engine/`: Python engine and CLI.
- `apps/web/`: local web UI (MVP).
- `scripts/`: reproducible smoke/demo and dev-check helpers.
- `docs/`: feature docs and ADRs.

## CLI (`apps/forge-engine/cli/main.py`)

Primary interface is the `forge` CLI (Typer-based).
Current command groups include:

- planning: `forge plan`
- data/tokenizer: `forge tokenizer train`, `forge data prepare`
- training/evaluation/chat: `forge train`, `forge eval`, `forge eval-compare`, `forge chat`
- model import: `forge model pull`
- guided flows: `forge wizard`, `forge stretch`

The CLI orchestrates modules in `app/` and writes local artifacts.

## forge-engine core (`apps/forge-engine/app/`)

### `core/`

- `backend.py`: backend detection and runtime notes.
- Current training behavior is PyTorch-based (CUDA/MPS/CPU).
- MLX detection exists; native MLX training backend is planned/in progress.

### `architectures/`

- Registry of model architectures and presets.
- Current implemented architecture: decoder-only Transformer.
- Registry allows adding more architectures incrementally.

### `tokenizer/`

- Tokenizer training/loading utilities.
- Used by both training and data preparation flows.

### `data/`

- Converts text corpus into sharded tokenized datasets used by trainer.
- Powers `forge data prepare` and training dataloaders.

### `training/`

- `planner.py`: heuristic estimation for runtime/cost/resource planning.
- `trainer.py`: PyTorch training loop with checkpointing, resume, validation logging.
- `scheduler.py`: learning-rate scheduling helpers.

### `checkpoints/`

- Save/load metadata and model artifacts.
- Conversion/import helpers for supported external model sources.

### `evaluation/`

- Perplexity and lightweight benchmark utilities.
- Primarily for run-to-run comparison and regression checks in current state.

### `wizard/`

- Guided training workflow with persisted session state and resumability.
- Includes data analysis, recommendation logic, and output summary generation.

### `stretch/`

- `forge stretch` v1 implementation.
- Current method: YaRN-only for compatible RoPE-based models.
- Current persistence mode: `adapter_plus_manifest` (source checkpoint copy + YaRN mapping artifact + deterministic manifest).
- Includes planner/executor/validator/session modules and deterministic reconstruction flow.

## API service (`apps/forge-engine/app/main.py`)

A lightweight FastAPI service exposes:

- architecture metadata
- hardware/backend info
- planning endpoint
- training logs/checkpoints listing
- basic evaluation endpoints

It is intended for local integration and dashboard support, not a cloud control plane.

## Web app (`apps/web/`)

- Next.js local UI.
- Current role: MVP support for local monitoring/workflow interaction.
- It is useful but not feature-complete for all CLI capabilities.

## Execution model and artifacts

- Local-first execution by default.
- Artifacts are written to local filesystem (`checkpoints/`, `artifacts/`, session folders).
- `scripts/smoke_test.sh` provides a reproducible CPU-friendly pipeline check.

## Non-goals of current architecture

- No claim of full multi-node orchestration.
- No claim of complete native MLX training coverage today.
- No claim that smoke/eval utilities prove production model quality.
