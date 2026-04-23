"""
ForgeAI Data Pipeline
=====================
Dataset loading, tokenization, sharding, and DataLoader creation.
"""

from __future__ import annotations

from .dataset import (
    iter_documents,
    prepare_dataset,
    ShardedTokenDataset,
    create_dataloader,
)

__all__ = [
    "iter_documents",
    "prepare_dataset",
    "ShardedTokenDataset",
    "create_dataloader",
]
