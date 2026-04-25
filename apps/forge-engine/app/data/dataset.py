"""
ForgeAI — Data Pipeline
========================

Loads text corpora, tokenizes them, and produces memory-mapped binary shards
for efficient training. Supports:
  - Raw text files (.txt, .md)
  - JSONL (each line: {"text": "..."})
  - Directories of mixed formats

The output is a set of .bin files (uint32 numpy arrays of token IDs) that can
be memory-mapped during training for zero-copy data loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer

_TOKEN_DTYPE_MAP: dict[str, Any] = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


# ── Text Extraction ──────────────────────────────────────────────────────────


def iter_documents(data_path: str | Path) -> Iterator[str]:
    """
    Yield documents (strings) from a file or directory.

    Supported formats:
      - .txt, .md: entire file content as one document
      - .jsonl: each line is {"text": "..."}, yields the text field
    """
    path = Path(data_path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.suffix in (".txt", ".jsonl", ".md")
            and p.is_file()
        )
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    for f in files:
        if f.suffix == ".jsonl":
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue
        else:
            text = f.read_text(encoding="utf-8").strip()
            if text:
                yield text


# ── Tokenization + Sharding ─────────────────────────────────────────────────


def prepare_dataset(
    data_path: str | Path,
    tokenizer: Tokenizer,
    output_dir: str | Path,
    context_length: int = 2048,
    shard_size: int = 100_000_000,  # ~100M tokens per shard
    eos_token_id: int | None = None,
    token_dtype: str = "uint32",
) -> dict[str, Any]:
    """
    Tokenize a corpus and save as binary shards.

    Each shard is a numpy .bin file of token IDs. Documents are
    concatenated with EOS tokens between them, then chunked into
    context_length-sized sequences.

    Args:
        data_path:      Path to text corpus
        tokenizer:      Trained tokenizer
        output_dir:     Where to save shards
        context_length: Sequence length for training
        shard_size:     Max tokens per shard file
        eos_token_id:   EOS token ID (auto-detected if None)
        token_dtype:    Token dtype for shard storage (default: uint32)

    Returns:
        dict with metadata (total_tokens, num_shards, shard_paths)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_prepare_params(context_length=context_length, shard_size=shard_size)
    dtype_name, dtype = _resolve_token_dtype(token_dtype)
    dtype_max = int(np.iinfo(dtype).max)

    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<|eos|>")
        if eos_token_id is None:
            eos_token_id = tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            eos_token_id = 0  # fallback
    if int(eos_token_id) > dtype_max:
        raise ValueError(
            f"EOS token id {eos_token_id} exceeds {dtype_name} capacity ({dtype_max})."
        )

    all_tokens: list[int] = []
    shard_idx = 0
    shard_paths: list[str] = []
    total_tokens = 0
    docs_processed = 0

    def _flush_shard(tokens: list[int]) -> None:
        nonlocal shard_idx, shard_paths, total_tokens
        if not tokens:
            return
        arr = np.array(tokens, dtype=dtype)
        shard_path = output_dir / f"shard_{shard_idx:04d}.bin"
        arr.tofile(str(shard_path))
        shard_paths.append(str(shard_path))
        total_tokens += len(tokens)
        shard_idx += 1

    for doc in iter_documents(data_path):
        docs_processed += 1
        encoded = tokenizer.encode(doc)
        token_ids = encoded.ids
        if token_ids:
            max_token_id = max(token_ids)
            if int(max_token_id) > dtype_max:
                raise ValueError(
                    f"Token id {max_token_id} exceeds {dtype_name} capacity ({dtype_max}). "
                    "Use a larger shard dtype."
                )
        all_tokens.extend(token_ids)
        all_tokens.append(eos_token_id)

        # Flush when shard is full
        while len(all_tokens) >= shard_size:
            _flush_shard(all_tokens[:shard_size])
            all_tokens = all_tokens[shard_size:]

    # Flush remaining tokens
    if all_tokens:
        _flush_shard(all_tokens)

    if total_tokens == 0 or docs_processed == 0:
        raise ValueError(
            "Dataset vuoto: nessun token valido prodotto. "
            "Controlla path, formato file e contenuto testuale."
        )

    # Save metadata
    metadata = {
        "total_tokens": total_tokens,
        "num_shards": len(shard_paths),
        "shard_paths": shard_paths,
        "context_length": context_length,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token_id": eos_token_id,
        "token_dtype": dtype_name,
        "documents_processed": docs_processed,
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ── Dataset for Training ────────────────────────────────────────────────────


class ShardedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Memory-mapped dataset over binary token shards.

    Each sample is a (context_length,) tensor of token IDs.
    Targets are the same sequence shifted by 1 (next-token prediction).
    """

    def __init__(self, data_dir: str | Path, context_length: int = 2048):
        self.data_dir = Path(data_dir)
        if context_length <= 1:
            raise ValueError("context_length must be > 1 for next-token training.")
        self.context_length = context_length

        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        token_dtype_name = "uint16"
        if isinstance(self.metadata, dict):
            raw_dtype = self.metadata.get("token_dtype")
            if isinstance(raw_dtype, str) and raw_dtype.strip():
                token_dtype_name = raw_dtype.strip().lower()
        _, token_dtype = _resolve_token_dtype(token_dtype_name)

        # Find all shard files
        self.shard_paths = sorted(self.data_dir.glob("shard_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.bin files found in {data_dir}")

        # Memory-map all shards and compute total length
        self.shards: list[np.ndarray] = []
        self.shard_offsets: list[int] = []  # cumulative token offsets
        offset = 0
        for sp in self.shard_paths:
            mmap = np.memmap(str(sp), dtype=token_dtype, mode="r")
            self.shards.append(mmap)
            self.shard_offsets.append(offset)
            offset += len(mmap)
        self.total_tokens = offset

        # Number of complete sequences we can form
        # We need context_length + 1 tokens per sample (input + target)
        self.num_samples = max(0, (self.total_tokens - 1) // self.context_length)

    def __len__(self) -> int:
        return self.num_samples

    def _get_token(self, global_idx: int) -> int:
        """Get a single token by global index across all shards."""
        # Binary search for the right shard
        shard_i = 0
        for i, off in enumerate(self.shard_offsets):
            if off <= global_idx:
                shard_i = i
            else:
                break
        local_idx = global_idx - self.shard_offsets[shard_i]
        return int(self.shards[shard_i][local_idx])

    def _get_tokens(self, start: int, length: int) -> np.ndarray:
        """Get a contiguous slice of tokens across shard boundaries."""
        result = np.empty(length, dtype=np.int64)
        pos = 0
        remaining = length

        # Find starting shard
        shard_i = 0
        for i, off in enumerate(self.shard_offsets):
            if off <= start:
                shard_i = i
            else:
                break

        global_pos = start
        while remaining > 0:
            local_start = global_pos - self.shard_offsets[shard_i]
            shard_len = len(self.shards[shard_i])
            available = shard_len - local_start
            to_read = min(available, remaining)

            result[pos:pos + to_read] = self.shards[shard_i][local_start:local_start + to_read]
            pos += to_read
            remaining -= to_read
            global_pos += to_read
            shard_i += 1

        return result

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_length
        # Get context_length + 1 tokens: input is [0:ctx], target is [1:ctx+1]
        tokens = self._get_tokens(start, self.context_length + 1)
        x = torch.from_numpy(tokens[:-1].copy()).long()
        y = torch.from_numpy(tokens[1:].copy()).long()
        return x, y


def create_dataloader(
    data_dir: str | Path,
    context_length: int = 2048,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a DataLoader from processed shard data."""
    dataset = ShardedTokenDataset(data_dir, context_length=context_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def _resolve_token_dtype(token_dtype: str) -> tuple[str, Any]:
    normalized = token_dtype.strip().lower()
    if normalized not in _TOKEN_DTYPE_MAP:
        allowed = ", ".join(sorted(_TOKEN_DTYPE_MAP.keys()))
        raise ValueError(f"Unsupported token dtype '{token_dtype}'. Allowed: {allowed}.")
    return normalized, _TOKEN_DTYPE_MAP[normalized]


def _validate_prepare_params(context_length: int, shard_size: int) -> None:
    if context_length <= 1:
        raise ValueError("context_length must be > 1.")
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0.")
