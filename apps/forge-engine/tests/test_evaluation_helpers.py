from __future__ import annotations

import pytest

pytest.importorskip("torch")

from app.evaluation.benchmarks import eval_hellaswag_mini


def test_eval_hellaswag_mini_returns_placeholder_when_data_missing() -> None:
    result = eval_hellaswag_mini(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        data_path=None,
    )

    assert result["benchmark"] == "hellaswag-mini"
    assert result["accuracy"] is None
    assert result["num_samples"] == 0
    assert "local data not found" in str(result["note"]).lower()
