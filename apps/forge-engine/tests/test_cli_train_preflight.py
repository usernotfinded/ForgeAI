from __future__ import annotations

import pytest

from cli.main import _compute_train_val_samples, _validate_training_data_preflight


def test_compute_train_val_samples_valid_case() -> None:
    train_samples, val_samples = _compute_train_val_samples(
        100,
        batch_size=4,
        val_split=0.1,
    )
    assert train_samples == 90
    assert val_samples == 10


def test_compute_train_val_samples_rejects_zero_samples() -> None:
    with pytest.raises(ValueError, match="Dataset privo di sequenze utili"):
        _compute_train_val_samples(
            0,
            batch_size=2,
            val_split=0.1,
        )


def test_compute_train_val_samples_rejects_invalid_val_split() -> None:
    with pytest.raises(ValueError, match="--val-split"):
        _compute_train_val_samples(
            100,
            batch_size=4,
            val_split=1.0,
        )


def test_compute_train_val_samples_rejects_train_loader_empty_with_drop_last() -> None:
    with pytest.raises(ValueError, match="Train loader vuoto"):
        _compute_train_val_samples(
            5,
            batch_size=4,
            val_split=0.5,
        )


def test_compute_train_val_samples_rejects_val_loader_empty_with_drop_last() -> None:
    with pytest.raises(ValueError, match="Validation loader vuoto"):
        _compute_train_val_samples(
            20,
            batch_size=8,
            val_split=0.1,
        )


def test_validate_training_data_preflight_rejects_context_length_too_large() -> None:
    with pytest.raises(ValueError, match="Dataset troppo piccolo per il context_length"):
        _validate_training_data_preflight(
            total_tokens=100,
            total_samples=2,
            context_length=100,
            batch_size=1,
            val_split=0.1,
        )


def test_validate_training_data_preflight_rejects_invalid_context_length() -> None:
    with pytest.raises(ValueError, match="--context-length deve essere > 1"):
        _validate_training_data_preflight(
            total_tokens=1000,
            total_samples=100,
            context_length=1,
            batch_size=4,
            val_split=0.1,
        )

