# Contributing to ForgeAI

Thanks for contributing.
This guide focuses on quick, reproducible engineering contributions.

## 1. Development setup

```bash
git clone https://github.com/usernotfinded/ForgeAI.git
cd ForgeAI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./apps/forge-engine --no-deps
python -m pip install pytest ruff typer rich numpy tqdm pydantic fastapi uvicorn tokenizers safetensors
```

Notes:

- The default contributor checks are CPU-only.
- CUDA/MLX are optional for local hardware-specific work.

## 2. Run checks before opening a PR

Use the same lightweight checks used in CI:

```bash
./scripts/dev_checks.sh
```

This runs:

- `ruff check .`
- `pytest apps/forge-engine/tests -q`
- CLI help smoke checks
- key module import smoke checks

## 3. Run the reproducible smoke demo

```bash
./scripts/smoke_test.sh
```

The smoke demo verifies pipeline wiring on toy data and writes artifacts under:

- `artifacts/smoke/<timestamp>/`

It proves basic end-to-end correctness, not real model quality.

## 4. Coding style and expectations

- Keep changes scoped and reviewable.
- Prefer simple, explicit logic over hidden behavior.
- Keep docs aligned with actual implementation.
- Do not overstate feature maturity in comments, help text, or README/docs.
- Add/adjust tests when changing behavior.

Python conventions in this repo:

- Use type hints for public interfaces.
- Keep `ruff check .` clean.
- Keep tests deterministic (no network, no large downloads, no flaky timing).

## 5. Submitting issues

When opening an issue, include:

- environment (OS, Python version, CPU/GPU)
- exact command run
- error output/logs
- expected vs actual behavior
- minimal reproduction steps

For feature proposals, separate:

- verified current behavior
- desired behavior
- why existing behavior is insufficient

## 6. Submitting pull requests

- Link related issue(s).
- Keep PRs focused on one problem area.
- Include a short test plan with exact commands run.
- Update docs when behavior/help text changes.
- Avoid bundling unrelated refactors.

## 7. Security and data handling

ForgeAI is local-first, but contributors should still avoid committing:

- private datasets
- secrets/tokens
- large generated artifacts

If you discover a security issue, report it privately to maintainers before public disclosure.
