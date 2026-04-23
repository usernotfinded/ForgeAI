"""
ForgeAI — BPE Tokenizer Trainer
================================

Trains a Byte-Pair Encoding tokenizer on a text corpus using the HuggingFace
`tokenizers` library. Produces a fast tokenizer that can be saved/loaded and
used directly during training and inference.

Supports:
  - Training from text files or directories of text files
  - Configurable vocab size, special tokens, min frequency
  - SentencePiece-style byte fallback for unknown characters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


# ── Special Tokens ────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|eos|>",
    "<|bos|>",
    "<|unk|>",
]


# ── Corpus Iterator ──────────────────────────────────────────────────────────


def _iter_text_files(data_path: str | Path) -> Iterator[str]:
    """Yield lines from text files in a directory or a single file."""
    path = Path(data_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.suffix in (".txt", ".jsonl", ".json", ".md", ".csv")
            and p.is_file()
        )
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if not files:
        raise ValueError(f"No text files found in {data_path}")

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
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield line


# ── Trainer ──────────────────────────────────────────────────────────────────


def train_bpe_tokenizer(
    data_path: str | Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer on a text corpus.

    Args:
        data_path:      Path to a text file or directory of text files
        vocab_size:     Target vocabulary size
        min_frequency:  Minimum frequency for a merge to be applied
        special_tokens: List of special tokens (defaults to SPECIAL_TOKENS)

    Returns:
        A trained Tokenizer instance
    """
    if special_tokens is None:
        special_tokens = list(SPECIAL_TOKENS)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Pre-tokenization: split on whitespace and punctuation (byte-level)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder matches the pre-tokenizer
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Collect file paths for the trainer
    path = Path(data_path)
    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        files = sorted(
            str(p) for p in path.rglob("*")
            if p.suffix in (".txt", ".md", ".csv")
            and p.is_file()
        )
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if not files:
        # Fall back to iterator-based training (for JSONL)
        tokenizer.train_from_iterator(
            _iter_text_files(data_path),
            trainer=trainer,
        )
    else:
        # Direct file training is faster
        tokenizer.train(files, trainer=trainer)

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_dir: str | Path) -> Path:
    """
    Save a trained tokenizer to disk.

    Saves:
      - tokenizer.json (the full tokenizer, loadable by HuggingFace)
      - config.json (metadata: vocab_size, special tokens)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Save metadata
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "pad_token": "<|pad|>",
        "eos_token": "<|eos|>",
        "bos_token": "<|bos|>",
        "unk_token": "<|unk|>",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_dir


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    """Load a tokenizer from a directory saved by save_tokenizer."""
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"No tokenizer.json found in {tokenizer_dir}")
    return Tokenizer.from_file(str(tokenizer_path))
