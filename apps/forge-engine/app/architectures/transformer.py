"""
ForgeAI — GPT-style Transformer Architecture
=============================================

A clean, from-scratch implementation of a decoder-only Transformer (GPT architecture)
in PyTorch. This is the real architecture used by GPT-2, GPT-3, LLaMA, Mistral, etc.

Key design choices:
  - Pre-LayerNorm (more stable training than the original Post-LN GPT)
  - RoPE (Rotary Position Embeddings) — used by LLaMA/Mistral, better than learned embeddings
  - Grouped Query Attention (GQA) — reduces KV cache memory, used by Mistral/Gemma
  - SwiGLU activation — better than GELU for FFN layers (used by LLaMA/PaLM)
  - No bias in linear layers — modern best practice
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

RopeScalingConfig = dict[str, float | int | str]


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class GPTConfig:
    """
    Full configuration for a GPT-style Transformer.

    Sensible defaults produce a ~125M parameter model (GPT-2 small scale).
    Adjust n_layer, n_head, n_embd to scale up or down.
    """

    # Vocabulary & context
    vocab_size: int = 32000        # Tokenizer vocabulary size (e.g. 32k for BPE)
    context_length: int = 2048     # Maximum sequence length (context window)

    # Model dimensions
    n_embd: int = 768              # Embedding dimension (d_model)
    n_layer: int = 12              # Number of Transformer blocks
    n_head: int = 12               # Number of query attention heads
    n_kv_head: int = 12            # Number of KV heads (set < n_head for GQA)

    # Feed-forward network
    ffn_multiplier: float = 4.0    # FFN hidden dim = ffn_multiplier * n_embd (SwiGLU uses 2/3 of this)

    # Regularization
    dropout: float = 0.0           # Dropout rate (0.0 = no dropout, best for large models)
    attn_dropout: float = 0.0      # Attention dropout rate

    # RoPE (Rotary Position Embeddings)
    rope_theta: float = 10000.0    # RoPE base frequency (10000 = original, 500000 = LLaMA 3)
    # Optional persistent RoPE scaling config (used by `forge stretch` YaRN flow).
    rope_scaling: RopeScalingConfig | None = None

    # Misc
    bias: bool = False             # Use bias in Linear layers (False = modern practice)
    tie_embeddings: bool = True    # Tie input/output embedding weights (saves params)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0.")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be > 0.")
        if self.n_head <= 0:
            raise ValueError("n_head must be > 0.")
        if self.n_kv_head <= 0:
            raise ValueError("n_kv_head must be > 0.")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be > 0.")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})."
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})."
            )
        return self.n_embd // self.n_head

    @property
    def ffn_hidden_dim(self) -> int:
        """
        SwiGLU FFN hidden dimension.
        SwiGLU gates use 2/3 of the standard multiplier to keep param count similar to GELU FFN.
        """
        hidden = int(self.ffn_multiplier * self.n_embd * 2 / 3)
        # Round to nearest multiple of 64 for efficiency
        return (hidden + 63) // 64 * 64

    def num_parameters(self, include_embeddings: bool = True) -> int:
        """Estimate total parameter count (useful before allocating the model)."""
        # Embeddings
        emb = self.vocab_size * self.n_embd

        # Per-layer: attention + FFN + 2x LayerNorm
        attn = (
            self.n_embd * self.n_embd  # Q projection
            + 2 * self.n_kv_head * self.head_dim * self.n_embd  # KV projections
            + self.n_embd * self.n_embd  # O projection
        )
        ffn = (
            self.n_embd * self.ffn_hidden_dim  # gate
            + self.n_embd * self.ffn_hidden_dim  # up
            + self.ffn_hidden_dim * self.n_embd  # down
        )
        ln = 2 * self.n_embd  # weight + bias per LN
        per_layer = attn + ffn + 2 * ln

        # Final LayerNorm + LM head
        final = ln + (0 if self.tie_embeddings else self.vocab_size * self.n_embd)

        total = (self.n_layer * per_layer) + final
        if include_embeddings:
            total += emb
        return total


# ── Rotary Position Embeddings (RoPE) ─────────────────────────────────────────


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    rope_scaling: RopeScalingConfig | None = None,
) -> torch.Tensor:
    """
    Precompute RoPE complex frequencies for a given head dimension and sequence length.

    Returns a tensor of shape (max_seq_len, head_dim // 2) with complex freqs.
    """
    # Frequency bands: θ_i = 1 / (theta^(2i / head_dim)) for i in [0, head_dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    if rope_scaling:
        positions = _apply_rope_scaling_positions(positions, rope_scaling)
    # Outer product: (max_seq_len, head_dim // 2)
    freqs = torch.outer(positions, freqs)
    # Convert to complex exponentials
    return torch.polar(torch.ones_like(freqs), freqs)  # e^(i * freq)


def _apply_rope_scaling_positions(
    positions: torch.Tensor,
    rope_scaling: RopeScalingConfig,
) -> torch.Tensor:
    """
    Apply persistent YaRN-style position scaling.

    v1 implementation keeps short-context positions unchanged and compresses
    only the tail beyond the original context window.
    """
    scaling_type = str(rope_scaling.get("type", "")).lower()
    factor = float(rope_scaling.get("factor", 1.0))
    original_max = int(
        rope_scaling.get(
            "original_max_position_embeddings",
            len(positions),
        )
    )

    if scaling_type != "yarn" or factor <= 1.0:
        return positions

    original_max = max(1, original_max)
    boundary = float(original_max)

    scaled = positions.clone()
    mask = scaled > boundary
    if not torch.any(mask):
        return scaled

    scaled[mask] = boundary + (scaled[mask] - boundary) / factor
    return scaled


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensors.

    Args:
        x:      (batch, seq_len, n_heads, head_dim)
        freqs:  (seq_len, head_dim // 2) complex tensor

    Returns:
        Rotated tensor of the same shape as x.
    """
    # View last dim as pairs of floats → treat as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape freqs for broadcasting: (1, seq_len, 1, head_dim // 2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    # Rotate in complex space, then convert back to real
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


# ── Grouped Query Attention ────────────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """
    Multi-Head / Grouped Query Attention with RoPE.

    When n_kv_head == n_head  → standard Multi-Head Attention (MHA)
    When n_kv_head == 1       → Multi-Query Attention (MQA)
    When 1 < n_kv_head < n_head → Grouped Query Attention (GQA, as in Mistral/Gemma)

    GQA reduces the KV cache size by a factor of (n_head / n_kv_head),
    which is the main memory bottleneck during inference.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_groups = config.n_head // config.n_kv_head  # heads per KV group

        # Q gets full n_head heads, K/V only get n_kv_head heads
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, n_embd

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_freqs[:T])
        k = apply_rope(k, rope_freqs[:T])

        # Repeat K/V heads to match Q head count (GQA expansion)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)

        # Rearrange to (B, heads, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention if available in PyTorch 2.0+)
        scale = 1.0 / math.sqrt(self.head_dim)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=(mask is None),  # use causal mask if no explicit mask
            scale=scale,
        )

        # Merge heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return cast(torch.Tensor, self.resid_dropout(self.o_proj(y)))


# ── SwiGLU Feed-Forward Network ───────────────────────────────────────────────


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = SiLU(gate(x)) * up(x)
    output = down(SwiGLU(x))

    Used by LLaMA, PaLM, Gemma — consistently outperforms GELU FFN.
    The 'gated' design lets the network learn to suppress irrelevant activations.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden_dim
        self.gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up   = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) acts as a soft gate on the "up" path
        return cast(torch.Tensor, self.dropout(self.down(F.silu(self.gate(x)) * self.up(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """
    A single Pre-LayerNorm Transformer block.

    Pre-LN (normalize before attention/FFN) is more stable than the original
    Post-LN design and doesn't require the learning rate warmup tricks.

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.n_embd)  # Pre-attention norm
        self.attn  = GroupedQueryAttention(config)
        self.norm2 = nn.RMSNorm(config.n_embd)  # Pre-FFN norm
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs)
        x = x + self.ffn(self.norm2(x))
        return x


# ── GPT Model ─────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    """
    GPT-style decoder-only Transformer.

    This is the same fundamental architecture as GPT-2, GPT-3, LLaMA, Mistral, and Gemma.
    The differences between those models are in the specific config values and training data.

    Usage:
        config = GPTConfig(vocab_size=32000, n_layer=12, n_head=12, n_embd=768)
        model = GPT(config)
        print(f"Parameters: {config.num_parameters() / 1e6:.1f}M")

        # Forward pass (training)
        tokens = torch.randint(0, config.vocab_size, (2, 512))  # (batch, seq_len)
        logits, loss = model(tokens, targets=tokens)

        # Generation (inference)
        prompt = torch.tensor([[1, 2, 3]])  # tokenized prompt
        output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding (no position embedding — we use RoPE instead)
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.norm_f = nn.RMSNorm(config.n_embd)

        # Language model head (projects to vocab logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share input/output embeddings (saves ~10% params, common in GPT-2+)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies (not a parameter — register as buffer)
        rope_freqs = precompute_rope_freqs(
            config.head_dim,
            config.context_length,
            config.rope_theta,
            config.rope_scaling,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT weight initialization."""
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale down residual projections by sqrt(2 * n_layer) for stability
            if hasattr(module, "_is_residual"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                      # (batch, seq_len) token indices
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len) for training
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds max context length {self.config.context_length}."
            )

        # Embed tokens
        x = self.token_emb(idx)  # (B, T, n_embd)

        # Pass through Transformer blocks (RoPE freqs are precomputed)
        for block in self.blocks:
            x = block(x, self.rope_freqs)

        # Final norm
        x = self.norm_f(x)

        if targets is not None:
            # Training: compute logits for all positions and loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for the last token (efficient)
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,          # (batch, prompt_len) token indices
        max_new_tokens: int = 200,
        temperature: float = 1.0,   # > 1.0 = more random, < 1.0 = more focused
        top_k: int = 50,            # Keep only top-k logits (0 = disabled)
        top_p: float = 1.0,         # Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: float = 1.0,  # > 1.0 penalizes repeated tokens
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, and top-p sampling.

        Args:
            idx:               Starting token indices (tokenized prompt)
            max_new_tokens:    How many tokens to generate
            temperature:       Sampling temperature (1.0 = no scaling)
            top_k:             Keep only top-k logits before sampling
            top_p:             Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p
            repetition_penalty: Penalize tokens that have already appeared

        Returns:
            Token indices including the prompt: (batch, prompt_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context to max length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            # Forward pass (only last token logits returned during inference)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    for token_id in set(idx[b].tolist()):
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # Apply temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold (shift by 1 to keep first token)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count actual model parameters."""
        params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)
