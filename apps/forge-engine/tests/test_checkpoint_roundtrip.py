from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from app.architectures.transformer import GPT, GPTConfig
from app.checkpoints.manager import CheckpointMetadata, load_checkpoint, save_checkpoint


def _tiny_model() -> GPT:
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
    return GPT(config)


def _metadata(step: int = 3) -> CheckpointMetadata:
    return CheckpointMetadata(
        step=step,
        epoch=1,
        loss=1.23,
        val_loss=1.11,
        learning_rate=1e-3,
        total_tokens_seen=4096,
        model_config={"vocab_size": 64, "context_length": 8},
        architecture="transformer",
        backend="cpu",
        dtype="float32",
    )


def test_save_checkpoint_writes_required_files_and_metadata(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    run_dir = tmp_path / "run"
    ckpt_dir = save_checkpoint(run_dir, model, optimizer, scheduler, _metadata(step=7))

    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "optimizer.pt").exists()
    assert (ckpt_dir / "scheduler.pt").exists()
    assert (ckpt_dir / "metadata.json").exists()
    assert (run_dir / "latest").exists()

    payload = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    assert payload["step"] == 7
    assert payload["architecture"] == "transformer"
    assert payload["backend"] == "cpu"
    assert payload["dtype"] == "float32"


def test_scheduler_artifact_is_optional_when_scheduler_not_passed(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = save_checkpoint(tmp_path / "run", model, optimizer, None, _metadata(step=5))

    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "optimizer.pt").exists()
    assert (ckpt_dir / "metadata.json").exists()
    assert not (ckpt_dir / "scheduler.pt").exists()


def test_load_checkpoint_restores_matching_eval_logits(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model_a = _tiny_model()
    model_a.eval()

    input_ids = torch.randint(0, model_a.config.vocab_size, (2, 5))
    with torch.no_grad():
        expected_logits, _ = model_a(input_ids)

    optimizer = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    ckpt_dir = save_checkpoint(tmp_path / "run", model_a, optimizer, scheduler, _metadata(step=9))

    torch.manual_seed(999)
    model_b = _tiny_model()
    model_b.eval()
    load_checkpoint(ckpt_dir, model_b, device="cpu")

    with torch.no_grad():
        loaded_logits, _ = model_b(input_ids)

    assert torch.allclose(expected_logits, loaded_logits, atol=0.0, rtol=0.0)


def test_load_checkpoint_from_run_dir_resolves_latest_via_public_api(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    run_dir = tmp_path / "run"

    save_checkpoint(run_dir, model, optimizer, scheduler, _metadata(step=3))
    save_checkpoint(run_dir, model, optimizer, scheduler, _metadata(step=8))

    fresh_model = _tiny_model()
    loaded_meta = load_checkpoint(run_dir, fresh_model, device="cpu")

    assert loaded_meta.step == 8


def test_checkpoint_metadata_roundtrip_fields(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    meta = _metadata(step=11)

    run_dir = tmp_path / "run"
    save_checkpoint(run_dir, model, optimizer, scheduler, meta)

    loaded_model = _tiny_model()
    loaded_meta = load_checkpoint(run_dir, loaded_model, device="cpu")

    assert loaded_meta.step == 11
    assert loaded_meta.epoch == 1
    assert loaded_meta.total_tokens_seen == 4096
    assert loaded_meta.architecture == "transformer"
    assert loaded_meta.backend == "cpu"
    assert loaded_meta.dtype == "float32"
