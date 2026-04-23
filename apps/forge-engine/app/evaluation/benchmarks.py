"""
ForgeAI — Offline Benchmarks
==============================

Lightweight benchmarks for evaluating model quality during/after training.

Supported benchmarks:
  - TinyStories eval: coherence scoring on held-out samples
  - HellaSwag-mini: 1000-sample 0-shot accuracy (sentence completion)
  - ARC-easy: common-sense reasoning (optional)

These are not frontier-level evals. They are useful for:
  - Comparing your runs against each other
  - Catching regressions during training
  - Confirming that training actually converged
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tokenizers import Tokenizer


# ── TinyStories Evaluation ───────────────────────────────────────────────────


def eval_tinystories(
    model: nn.Module,
    tokenizer: Tokenizer,
    device: str = "cpu",
    num_samples: int = 50,
    max_gen_tokens: int = 200,
    temperature: float = 0.8,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """
    Generate short stories and score coherence.

    Uses a set of TinyStories-style prompts. For each prompt:
      1. Generate a continuation
      2. Score based on:
         - Non-repetition (ratio of unique tokens)
         - Length (did it produce meaningful output?)
         - Doesn't degenerate to a single repeated token

    Returns dict with average coherence score and sample generations.
    """
    prompts = [
        "Once upon a time, there was a little",
        "The cat sat on the",
        "One day, a boy named Tom went to",
        "Lily loved to play in the",
        "There was a big red",
        "Mom said it was time to",
        "The dog ran to the",
        "In a small house, there lived",
        "The sun was shining and",
        "A girl found a magic",
    ]

    model.eval()
    model = model.to(device)
    results = []
    total_score = 0.0

    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i % len(prompts)]
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_gen_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
            )

        generated_ids = output_ids[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        new_tokens = generated_ids[len(encoded.ids):]

        # Score: uniqueness ratio (higher = less repetitive)
        unique_ratio = len(set(new_tokens)) / max(len(new_tokens), 1)

        # Score: length penalty (very short = bad)
        length_score = min(1.0, len(new_tokens) / max_gen_tokens)

        # Score: not degenerate (top token < 50% of all tokens)
        if new_tokens:
            from collections import Counter
            counts = Counter(new_tokens)
            most_common_ratio = counts.most_common(1)[0][1] / len(new_tokens)
            diversity_score = 1.0 - most_common_ratio
        else:
            diversity_score = 0.0

        coherence = (unique_ratio + length_score + diversity_score) / 3.0
        total_score += coherence

        results.append({
            "prompt": prompt,
            "generation": generated_text[:500],
            "unique_ratio": unique_ratio,
            "length_score": length_score,
            "diversity_score": diversity_score,
            "coherence": coherence,
        })

    avg_score = total_score / max(len(results), 1)

    return {
        "benchmark": "tinystories",
        "avg_coherence": avg_score,
        "num_samples": len(results),
        "samples": results[:5],  # only return first 5 for display
    }


# ── HellaSwag-mini ───────────────────────────────────────────────────────────


def eval_hellaswag_mini(
    model: nn.Module,
    tokenizer: Tokenizer,
    data_path: Optional[str] = None,
    device: str = "cpu",
    num_samples: int = 1000,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """
    HellaSwag 0-shot evaluation on a subset.

    HellaSwag is a sentence completion benchmark: given a context,
    pick the correct ending from 4 choices.

    Expects a JSONL file where each line has:
      {"ctx": "...", "endings": ["...", "...", "...", "..."], "label": 0}

    If no data_path is provided, returns a placeholder result with instructions
    on how to download the dataset.
    """
    if data_path is None or not Path(data_path).exists():
        return {
            "benchmark": "hellaswag-mini",
            "accuracy": None,
            "num_samples": 0,
            "note": (
                "HellaSwag data not found. Download from: "
                "https://github.com/rowanz/hellaswag "
                "Place hellaswag_val.jsonl at the data_path."
            ),
        }

    # Load data
    samples: list[dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            if len(samples) >= num_samples:
                break
            obj = json.loads(line.strip())
            samples.append(obj)

    model.eval()
    model = model.to(device)
    correct = 0

    use_amp = dtype != torch.float32 and device != "cpu"
    amp_device = "cuda" if "cuda" in device else "cpu"

    for sample in tqdm(samples, desc="HellaSwag", leave=False):
        ctx = sample.get("ctx", sample.get("context", ""))
        endings = sample.get("endings", sample.get("choices", []))
        label = int(sample.get("label", sample.get("answer", 0)))

        if not endings:
            continue

        # Score each ending by average log-likelihood
        best_score = float("-inf")
        best_idx = 0

        for idx, ending in enumerate(endings):
            full_text = ctx + " " + ending
            encoded = tokenizer.encode(full_text)
            input_ids = torch.tensor([encoded.ids], device=device)

            ctx_encoded = tokenizer.encode(ctx)
            ctx_len = len(ctx_encoded.ids)

            with torch.no_grad():
                with torch.amp.autocast(device_type=amp_device, dtype=dtype, enabled=use_amp):
                    logits, _ = model(input_ids)

            # Score only the ending tokens
            logits = logits[0, ctx_len - 1:-1, :]  # shift by 1 for next-token prediction
            targets = input_ids[0, ctx_len:]
            if logits.shape[0] == 0 or targets.shape[0] == 0:
                continue

            min_len = min(logits.shape[0], targets.shape[0])
            logits = logits[:min_len]
            targets = targets[:min_len]

            log_probs = F.log_softmax(logits, dim=-1)
            token_scores = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            avg_score = token_scores.mean().item()

            if avg_score > best_score:
                best_score = avg_score
                best_idx = idx

        if best_idx == label:
            correct += 1

    accuracy = correct / max(len(samples), 1)

    return {
        "benchmark": "hellaswag-mini",
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "num_samples": len(samples),
    }


# ── Comparison Mode ──────────────────────────────────────────────────────────


def compare_checkpoints(results_a: dict[str, Any], results_b: dict[str, Any]) -> str:
    """
    Compare two evaluation results and produce a markdown summary.

    Args:
        results_a: Evaluation results from checkpoint A
        results_b: Evaluation results from checkpoint B

    Returns:
        Markdown string with side-by-side comparison
    """
    lines = ["# ForgeAI — Model Comparison", ""]
    lines.append("| Metric | Model A | Model B | Winner |")
    lines.append("|--------|---------|---------|--------|")

    def _add_row(
        name: str,
        val_a: Any,
        val_b: Any,
        lower_is_better: bool = True,
    ) -> None:
        if val_a is None or val_b is None:
            lines.append(f"| {name} | {val_a} | {val_b} | — |")
            return

        if lower_is_better:
            winner = "A" if val_a < val_b else ("B" if val_b < val_a else "Tie")
        else:
            winner = "A" if val_a > val_b else ("B" if val_b > val_a else "Tie")

        fmt_a = f"{val_a:.4f}" if isinstance(val_a, float) else str(val_a)
        fmt_b = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b)
        lines.append(f"| {name} | {fmt_a} | {fmt_b} | **{winner}** |")

    # Compare common metrics
    _add_row("Loss", results_a.get("loss"), results_b.get("loss"), lower_is_better=True)
    _add_row("Perplexity", results_a.get("perplexity"), results_b.get("perplexity"), lower_is_better=True)

    for key in ["accuracy", "avg_coherence"]:
        if key in results_a or key in results_b:
            _add_row(
                key.replace("_", " ").title(),
                results_a.get(key),
                results_b.get(key),
                lower_is_better=False,
            )

    return "\n".join(lines)
