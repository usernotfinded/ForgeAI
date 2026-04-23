"""
ForgeAI — Perplexity Evaluation
=================================

Compute perplexity on a dataset — the standard metric for language model quality.

Perplexity = exp(average cross-entropy loss)

Lower is better. For reference:
  - Random (32k vocab): ~32,000
  - Untrained model: ~32,000
  - Converged 50M model on TinyStories: 20-50
  - Converged 400M model on general text: 10-30
  - GPT-2 on WikiText-103: ~22
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu",
    max_batches: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float | int]:
    """
    Compute perplexity over a dataset.

    Args:
        model:        The language model
        dataloader:   DataLoader yielding (input_ids, target_ids) pairs
        device:       Device to run on
        max_batches:  Limit evaluation to N batches (None = full dataset)
        dtype:        Compute dtype

    Returns:
        Dict with loss, perplexity, and number of tokens evaluated
    """
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    use_amp = dtype != torch.float32 and device != "cpu"
    amp_device = "cuda" if "cuda" in device else "cpu"

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating perplexity", leave=False):
            if max_batches is not None and num_batches >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type=amp_device, dtype=dtype, enabled=use_amp):
                _, loss = model(x, targets=y)

            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))  # cap to avoid overflow

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "tokens_evaluated": total_tokens,
        "batches": num_batches,
    }
