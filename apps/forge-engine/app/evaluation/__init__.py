"""
ForgeAI Evaluation
==================
Perplexity tracking, offline benchmarks, and model comparison.
"""

from __future__ import annotations

from .perplexity import compute_perplexity
from .benchmarks import eval_tinystories, eval_hellaswag_mini, compare_checkpoints

__all__ = [
    "compute_perplexity",
    "eval_tinystories",
    "eval_hellaswag_mini",
    "compare_checkpoints",
]
