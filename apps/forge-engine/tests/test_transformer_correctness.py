from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from app.architectures import get_architecture
from app.architectures.transformer import GPT, GPTConfig


def _tiny_config(**overrides: int) -> GPTConfig:
    kwargs = {
        "vocab_size": 64,
        "context_length": 8,
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 1,
        "dropout": 0.0,
        "attn_dropout": 0.0,
    }
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def test_forward_logits_shape_with_targets() -> None:
    torch.manual_seed(0)
    model = GPT(_tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))

    logits, loss = model(idx, targets=targets)

    assert logits.shape == (2, 5, model.config.vocab_size)
    assert loss is not None


def test_forward_returns_scalar_loss_with_targets() -> None:
    torch.manual_seed(1)
    model = GPT(_tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (1, 4))
    targets = torch.randint(0, model.config.vocab_size, (1, 4))

    _, loss = model(idx, targets=targets)

    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss).item() is True


def test_generate_appends_exact_new_tokens() -> None:
    torch.manual_seed(7)
    model = GPT(_tiny_config())
    prompt = torch.randint(0, model.config.vocab_size, (1, 3))

    generated = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    assert generated.shape == (1, 7)


def test_forward_raises_value_error_on_context_overflow() -> None:
    model = GPT(_tiny_config(context_length=6))
    idx = torch.randint(0, model.config.vocab_size, (1, 7))

    with pytest.raises(ValueError, match="exceeds max context length"):
        model(idx)


def test_config_rejects_invalid_gqa_ratio() -> None:
    with pytest.raises(ValueError, match="n_head .* must be divisible by n_kv_head"):
        _tiny_config(n_head=3, n_kv_head=2)


def test_config_rejects_invalid_n_embd_head_ratio() -> None:
    with pytest.raises(ValueError, match="n_embd .* must be divisible by n_head"):
        _tiny_config(n_embd=18, n_head=4)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("vocab_size", 0),
        ("context_length", 0),
        ("n_layer", 0),
        ("n_head", -1),
        ("n_kv_head", 0),
        ("n_embd", -8),
    ],
)
def test_config_rejects_non_positive_critical_fields(field_name: str, value: int) -> None:
    kwargs = {
        "vocab_size": 64,
        "context_length": 8,
        "n_embd": 16,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 1,
        "dropout": 0.0,
        "attn_dropout": 0.0,
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name} must be > 0"):
        GPTConfig(**kwargs)


def test_registry_custom_tiny_transformer_instantiation() -> None:
    model = get_architecture(
        "transformer",
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=16,
        vocab_size=64,
        context_length=8,
    )

    assert model.config.n_layer == 1
    assert model.config.n_head == 2
    assert model.config.n_kv_head == 1
    assert model.config.n_embd == 16
    assert model.config.context_length == 8

    idx = torch.randint(0, model.config.vocab_size, (1, 4))
    logits, loss = model(idx)
    assert logits.shape == (1, 1, model.config.vocab_size)
    assert loss is None
