from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")
import torch

from app.architectures.transformer import GPT, GPTConfig
from app.checkpoints.manager import CheckpointMetadata, save_checkpoint
from cli.main import app


class _DummyTokenizer:
    def get_vocab_size(self) -> int:
        return 64


class _DummyModel:
    def __init__(self, *, context_length: int, vocab_size: int = 64) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            context_length=context_length,
            n_layer=1,
            n_head=2,
            n_kv_head=1,
            n_embd=16,
        )

    def num_parameters(self) -> int:
        return 1024


class _LargeDummyModel(_DummyModel):
    def num_parameters(self) -> int:
        return 7_000_000_000


def _backend_stub() -> SimpleNamespace:
    return SimpleNamespace(
        type=SimpleNamespace(value="cpu"),
        device_name="CPU test",
        recommended_dtype="float32",
        torch_device="cpu",
        vram_gb=None,
        unified_memory=False,
        mlx_available=False,
    )


def _prepare_tiny_dataset(path: Path, total_tokens: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    shard_path = path / "shard_0000.bin"
    np.arange(total_tokens, dtype=np.uint32).tofile(shard_path)
    metadata = {
        "total_tokens": total_tokens,
        "num_shards": 1,
        "shard_paths": [str(shard_path.resolve())],
        "context_length": 128,
        "vocab_size": 64,
        "eos_token_id": 0,
        "token_dtype": "uint32",
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _tiny_checkpoint(run_dir: Path) -> Path:
    config = GPTConfig(
        vocab_size=64,
        context_length=8,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        dropout=0.0,
        attn_dropout=0.0,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    metadata = CheckpointMetadata(
        step=1,
        epoch=0,
        loss=1.0,
        val_loss=None,
        learning_rate=1e-3,
        total_tokens_seen=128,
        model_config={
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
        },
        architecture="transformer",
        backend="cpu",
        dtype="float32",
    )
    save_checkpoint(run_dir, model, optimizer, scheduler, metadata)
    return run_dir


def test_plan_missing_data_path_exits_non_zero_with_clear_message() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "plan",
            "--arch",
            "transformer",
            "--params",
            "50M",
            "--data",
            "./path-does-not-exist-forgeai-cli-test",
        ],
    )

    assert result.exit_code == 1
    assert "Dataset path not found" in result.stdout


def test_train_missing_prepared_data_path_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()

    monkeypatch.setattr("app.core.backend.get_backend", lambda: _backend_stub())
    monkeypatch.setattr("app.tokenizer.load_tokenizer", lambda _path: _DummyTokenizer())
    monkeypatch.setattr(
        "app.architectures.get_architecture",
        lambda *_args, **kwargs: _DummyModel(
            context_length=int(kwargs.get("context_length", 128)),
            vocab_size=int(kwargs.get("vocab_size", 64)),
        ),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--arch",
            "transformer",
            "--preset",
            "forge-nano",
            "--data",
            str(tmp_path / "missing_prepared"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--output",
            str(tmp_path / "checkpoints"),
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "No shard_*.bin files found" in result.stdout


def test_train_dataset_too_small_for_context_exits_non_zero_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    prepared = _prepare_tiny_dataset(tmp_path / "prepared", total_tokens=32)

    monkeypatch.setattr("app.core.backend.get_backend", lambda: _backend_stub())
    monkeypatch.setattr("app.tokenizer.load_tokenizer", lambda _path: _DummyTokenizer())
    monkeypatch.setattr(
        "app.architectures.get_architecture",
        lambda *_args, **kwargs: _DummyModel(
            context_length=int(kwargs.get("context_length", 128)),
            vocab_size=int(kwargs.get("vocab_size", 64)),
        ),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--arch",
            "transformer",
            "--preset",
            "forge-nano",
            "--data",
            str(prepared),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--output",
            str(tmp_path / "checkpoints"),
            "--context-length",
            "128",
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "Dataset troppo piccolo per il context_length" in result.stdout


def test_train_hardware_feasibility_warning_does_not_block_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    prepared = _prepare_tiny_dataset(tmp_path / "prepared", total_tokens=100)

    backend = _backend_stub()
    backend.vram_gb = 0.1
    monkeypatch.setattr("app.core.backend.get_backend", lambda: backend)
    monkeypatch.setattr("app.tokenizer.load_tokenizer", lambda _path: _DummyTokenizer())
    monkeypatch.setattr(
        "app.architectures.get_architecture",
        lambda *_args, **kwargs: _LargeDummyModel(
            context_length=int(kwargs.get("context_length", 8)),
            vocab_size=int(kwargs.get("vocab_size", 64)),
        ),
    )
    monkeypatch.setattr(
        "app.training.trainer.train",
        lambda **_kwargs: {"final_step": 1, "final_loss": 1.0, "best_val_loss": None},
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--arch",
            "transformer",
            "--preset",
            "forge-nano",
            "--data",
            str(prepared),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--output",
            str(tmp_path / "checkpoints"),
            "--context-length",
            "8",
            "--batch-size",
            "1",
            "--val-split",
            "0.25",
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Hardware warning" in result.stdout
    assert "hardware feasibility is advisory" in result.stdout
    assert "Training complete" in result.stdout


def test_train_strict_hardware_checks_fail_on_advisory_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    prepared = _prepare_tiny_dataset(tmp_path / "prepared", total_tokens=100)

    backend = _backend_stub()
    backend.vram_gb = 0.1
    monkeypatch.setattr("app.core.backend.get_backend", lambda: backend)
    monkeypatch.setattr("app.tokenizer.load_tokenizer", lambda _path: _DummyTokenizer())
    monkeypatch.setattr(
        "app.architectures.get_architecture",
        lambda *_args, **kwargs: _LargeDummyModel(
            context_length=int(kwargs.get("context_length", 8)),
            vocab_size=int(kwargs.get("vocab_size", 64)),
        ),
    )
    monkeypatch.setattr(
        "app.training.trainer.train",
        lambda **_kwargs: pytest.fail("training should not run in strict hardware mode"),
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--arch",
            "transformer",
            "--preset",
            "forge-nano",
            "--data",
            str(prepared),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--output",
            str(tmp_path / "checkpoints"),
            "--context-length",
            "8",
            "--batch-size",
            "1",
            "--val-split",
            "0.25",
            "--max-steps",
            "1",
            "--strict-hardware-checks",
        ],
    )

    assert result.exit_code == 1
    assert "Strict hardware checks enabled" in result.stdout


def test_eval_perplexity_only_without_data_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    run_dir = _tiny_checkpoint(tmp_path / "run")
    monkeypatch.setattr("app.core.backend.get_backend", lambda: _backend_stub())

    result = runner.invoke(
        app,
        [
            "eval",
            str(run_dir),
            "--benchmark",
            "perplexity",
        ],
    )

    assert result.exit_code == 1
    assert "Perplexity requested but --data is missing" in result.stdout


def test_eval_perplexity_missing_data_warns_and_skips_when_other_benchmarks_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    run_dir = _tiny_checkpoint(tmp_path / "run")

    monkeypatch.setattr("app.core.backend.get_backend", lambda: _backend_stub())
    monkeypatch.setattr("app.tokenizer.load_tokenizer", lambda _path: object())
    monkeypatch.setattr(
        "app.evaluation.eval_tinystories",
        lambda *_args, **_kwargs: {
            "benchmark": "tinystories",
            "avg_coherence": 0.5,
            "num_samples": 1,
            "samples": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "eval",
            str(run_dir),
            "--benchmark",
            "perplexity",
            "--benchmark",
            "tinystories",
            "--tokenizer",
            str(tmp_path / "tok"),
        ],
    )

    assert result.exit_code == 0
    assert "Perplexity requested but --data is missing" in result.stdout
    assert "Skipping perplexity and continuing with other checks." in result.stdout
