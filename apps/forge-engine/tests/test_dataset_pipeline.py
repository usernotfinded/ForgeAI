from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from app.data.dataset import ShardedTokenDataset, prepare_dataset


@dataclass
class _FakeEncoding:
    ids: list[int]


class _FakeTokenizer:
    def __init__(self, ids_by_doc: dict[str, list[int]], *, vocab_size: int = 80000) -> None:
        self._ids_by_doc = ids_by_doc
        self._vocab_size = vocab_size

    def token_to_id(self, token: str) -> int | None:
        if token in {"<|eos|>", "</s>"}:
            return 0
        return None

    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding(ids=list(self._ids_by_doc.get(text, [])))

    def get_vocab_size(self) -> int:
        return self._vocab_size


def test_prepare_dataset_uses_uint32_and_preserves_large_token_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text("doc-alpha", encoding="utf-8")
    tokenizer = _FakeTokenizer({"doc-alpha": [1, 70000, 3]}, vocab_size=80001)
    output_dir = tmp_path / "prepared"

    metadata = prepare_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        output_dir=output_dir,
        context_length=2,
    )

    assert metadata["token_dtype"] == "uint32"
    dataset = ShardedTokenDataset(output_dir, context_length=2)
    x, y = dataset[0]
    assert x.tolist() == [1, 70000]
    assert y.tolist() == [70000, 3]


def test_prepare_dataset_rejects_uint16_overflow(tmp_path: Path) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text("doc-overflow", encoding="utf-8")
    tokenizer = _FakeTokenizer({"doc-overflow": [70000]}, vocab_size=80001)

    with pytest.raises(ValueError, match="exceeds uint16 capacity"):
        prepare_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            output_dir=tmp_path / "prepared",
            context_length=32,
            token_dtype="uint16",
        )


def test_prepare_dataset_rejects_empty_corpus(tmp_path: Path) -> None:
    data_path = tmp_path / "empty.txt"
    data_path.write_text("\n", encoding="utf-8")
    tokenizer = _FakeTokenizer({})

    with pytest.raises(ValueError, match="Dataset vuoto"):
        prepare_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            output_dir=tmp_path / "prepared",
            context_length=32,
        )


def test_prepare_dataset_rejects_context_length_below_two(tmp_path: Path) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text("x", encoding="utf-8")
    tokenizer = _FakeTokenizer({"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="context_length must be > 1"):
        prepare_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            output_dir=tmp_path / "prepared",
            context_length=1,
        )


def test_sharded_dataset_reads_legacy_uint16_when_metadata_dtype_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "legacy"
    output_dir.mkdir(parents=True)

    np.array([10, 20, 30], dtype=np.uint16).tofile(output_dir / "shard_0000.bin")
    metadata = {
        "total_tokens": 3,
        "num_shards": 1,
        "shard_paths": [str((output_dir / "shard_0000.bin").resolve())],
        "context_length": 2,
        "vocab_size": 32000,
        "eos_token_id": 0,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    dataset = ShardedTokenDataset(output_dir, context_length=2)
    x, y = dataset[0]
    assert x.tolist() == [10, 20]
    assert y.tolist() == [20, 30]
