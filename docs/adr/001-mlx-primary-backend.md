# ADR-001: Backend Priority — MLX over CUDA as Primary Development Target

**Date**: 2026-04-22
**Status**: Accepted (directional), revised 2026-04-25

## Context

ForgeAI needs a primary compute backend. The main developer works on Apple Silicon (MacBook Air M4 16GB). CUDA-first tools (Axolotl, Unsloth, LLaMA-Factory) already dominate the fine-tuning space.

## Decision

ForgeAI remains Apple Silicon/MLX-oriented as a product direction, but the currently implemented training runtime is PyTorch-based.

- Current training path: PyTorch on CUDA / MPS / CPU
- MLX is currently integrated for inference workflows (via `mlx-lm`) and environment detection
- Native MLX training backend is planned and tracked, but not yet complete
- CPU remains a fallback/testing path (slow but supported)

## Rationale

1. **Less competition**: CUDA-first LLM training tools are mature and crowded. MLX support for full pre-training from scratch is underserved.
2. **Development hardware**: The main author's machine is Apple Silicon, so Apple compatibility remains a priority.
3. **Unified memory advantage**: Apple Silicon's unified memory architecture allows larger models relative to VRAM for equivalent cost.

## Consequences

- Documentation and CLI messaging must explicitly distinguish implemented PyTorch training vs planned native MLX training.
- MLX-specific training code paths will be introduced incrementally; until then, no claims of "native MLX training complete".
- Some advanced features (Flash Attention 2, FSDP) remain CUDA-specific in the near term.
