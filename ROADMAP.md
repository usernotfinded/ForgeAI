# ForgeAI Roadmap

This roadmap is staged and implementation-driven. It is not a promise of fixed dates.

ForgeAI prioritizes correctness, local-first workflows, and honest capability boundaries over broad feature claims.

## Phase 0 — Reliability and Smoke Tests

Goal: make local verification predictable for contributors.

- Keep CI CPU-only, fast, and deterministic.
- Maintain a one-command smoke pipeline (`./scripts/smoke_test.sh`).
- Keep import/CLI smoke checks stable on clean environments.
- Keep torch-enabled core tests running in CI so model/checkpoint regressions are not silently skipped.
- Keep `mypy --strict apps/forge-engine/app` enforced in CI.
- Tighten docs so contributors can validate changes quickly.

Definition of done:

- New contributors can run setup + checks in under 10 minutes.
- CI failures are actionable and reproducible locally.
- Core torch-dependent tests execute in the release-relevant CI lane.

## Phase 1 — Local Training Workflow

Goal: reliable local-first training workflow for small and mid-sized ForgeAI-native models.

- Maintain end-to-end path: data -> tokenizer -> planner -> train -> eval -> chat.
- Improve resume/recovery behavior and artifact traceability.
- Keep training ergonomics practical on consumer hardware.
- Continue hardening `forge wizard` and `forge stretch` v1 flows.
- Keep arbitrary external model adaptation out of the stable path until it is genuinely implemented.
- Keep smoke/demo output clearly framed as pipeline validation, not model-quality proof.

Definition of done:

- The core pipeline is stable across repeated local runs.
- Artifacts and reports are clear enough for debugging and review.
- v0.1 users can understand exactly what checkpoint formats and workflows are supported.

## Phase 2 — Apple Silicon / MLX-Native Improvements

Goal: reduce mismatch between Apple Silicon focus and the current PyTorch-first training path.

- Add experimental MLX backend foundation:
  - MLX availability checks.
  - `forge experimental mlx-smoke`.
  - tiny MLX forward/loss/train-step smoke if practical.
- Implement and validate native MLX training incrementally.
- Keep behavior explicit when runtime falls back to PyTorch MPS/CPU.
- Improve Apple Silicon performance guidance with measured examples.
- Keep MLX optional until the native path is tested and documented.

Definition of done:

- Native MLX training exists for a limited supported subset and is documented as such.
- Apple Silicon users can run a verified MLX smoke path.
- Docs clearly distinguish PyTorch MPS, MLX inference/smoke, and true MLX-native training.

## Phase 3 — Evaluation, Distillation, and Model Quality

Goal: make model-quality claims measurable and harder to overstate.

- Expand evaluation coverage beyond minimal regression checks.
- Improve long-context validation quality and reporting.
- Add clearer baselines and comparison workflows for local runs.
- Keep evaluation labels honest:
  - perplexity = real local metric;
  - TinyStories-style checks = proxy/lightweight;
  - HellaSwag-style checks = local-file dependent unless bundled/tested.
- Add an experimental distillation pipeline:
  - prompt dataset input;
  - teacher output collection;
  - optional multi-teacher aggregation;
  - filtering/deduplication;
  - distilled JSONL dataset export;
  - integration with existing `forge data prepare` and `forge train`.
- Track teacher metadata, prompt source, generation settings, and license/usage notes where possible.
- Avoid claiming “better reasoning” or “dense student” without evaluation evidence.

Definition of done:

- Evaluation outputs can support technical review decisions, not only smoke validation.
- Distillation can produce auditable local training datasets without hiding teacher/model provenance.
- Student improvements are measured against documented baselines.

## Phase 4 — UI and Advanced Orchestration

Goal: improve usability without changing local-first identity.

- Evolve `apps/web` from MVP dashboards to stronger workflow coverage.
- Improve session management, reporting, and guided flows.
- Add optional non-interactive/batch-friendly orchestration entry points.
- Keep web UI aligned with CLI-supported workflows.
- Add web CI only after lockfile/package strategy is settled.
- Keep experimental features clearly separated from stable workflows.

Definition of done:

- CLI and web provide consistent, transparent workflow outcomes.
- Web workflows do not overclaim support beyond the underlying engine.

## Experimental Integrations

These integrations may exist before they are part of the stable roadmap. They must remain optional and clearly labeled.

- Keras 3 / KerasHub:
  - optional experimental integration only;
  - useful for prototyping and future model exploration;
  - does not replace the ForgeAI-native PyTorch core.
- MLX:
  - optional experimental backend foundation first;
  - native training comes later and only for a limited supported subset.
- Distillation:
  - experimental dataset-generation pipeline first;
  - no automatic quality claims without evaluation.

## Out of scope for near-term phases

- Frontier-scale multi-node claims without verified implementation.
- Marketing-style capability promises not backed by repository evidence.
- Production-grade web orchestration before the CLI workflows are stable.
- Arbitrary external model adaptation unless checkpoint/model compatibility is explicitly implemented and tested.
- Full long-context checkpoint conversion unless the generated artifact is actually loadable and validated.