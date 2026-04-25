# ForgeAI Roadmap

This roadmap is staged and implementation-driven. It is not a promise of fixed dates.

## Phase 0 — Reliability and Smoke Tests

Goal: make local verification predictable for contributors.

- Keep CI CPU-only, fast, and deterministic.
- Maintain a one-command smoke pipeline (`./scripts/smoke_test.sh`).
- Keep import/CLI smoke checks stable on clean environments.
- Tighten docs so contributors can validate changes quickly.

Definition of done:

- New contributors can run setup + checks in under 10 minutes.
- CI failures are actionable and reproducible locally.

## Phase 1 — Local Training Workflow

Goal: reliable local-first training workflow for small and mid-sized models.

- Maintain end-to-end path: data -> tokenizer -> planner -> train -> eval -> chat.
- Improve resume/recovery behavior and artifact traceability.
- Keep training ergonomics practical on consumer hardware.
- Continue hardening `forge wizard` and `forge stretch` v1 flows.

Definition of done:

- The core pipeline is stable across repeated local runs.
- Artifacts and reports are clear enough for debugging and review.

## Phase 2 — Apple Silicon / MLX-Native Improvements

Goal: reduce mismatch between Apple Silicon focus and current PyTorch-first training path.

- Implement and validate native MLX training path incrementally.
- Keep behavior explicit when runtime falls back to PyTorch MPS/CPU.
- Improve Apple Silicon performance guidance with measured examples.

Definition of done:

- Native MLX training exists for a limited supported subset and is documented as such.

## Phase 3 — Evaluation and Model Quality

Goal: make model-quality claims measurable and harder to overstate.

- Expand evaluation coverage beyond minimal regression checks.
- Improve long-context validation quality and reporting.
- Add clearer baselines and comparison workflows for local runs.

Definition of done:

- Evaluation outputs can support technical review decisions, not only smoke validation.

## Phase 4 — UI and Advanced Orchestration

Goal: improve usability without changing local-first identity.

- Evolve `apps/web` from MVP dashboards to stronger workflow coverage.
- Improve session management, reporting, and guided flows.
- Add optional non-interactive/batch-friendly orchestration entry points.

Definition of done:

- CLI and web provide consistent, transparent workflow outcomes.

## Out of scope for near-term phases

- Frontier-scale multi-node claims without verified implementation.
- Marketing-style capability promises not backed by repository evidence.
