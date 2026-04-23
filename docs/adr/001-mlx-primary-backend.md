# ADR-001: Backend Priority — MLX over CUDA as Primary Development Target

**Date**: 2026-04-22
**Status**: Accepted

## Context

ForgeAI needs a primary compute backend. The main developer works on Apple Silicon (MacBook Air M4 16GB). CUDA-first tools (Axolotl, Unsloth, LLaMA-Factory) already dominate the fine-tuning space.

## Decision

MLX (Apple's native ML framework) is the primary development target.

- All v0.1 functionality is developed and tested on MLX first
- CUDA is fully supported but not the development focus
- PyTorch MPS is a fallback for Apple Silicon without MLX installed
- CPU is inference-only

## Rationale

1. **Less competition**: CUDA-first LLM training tools are mature and crowded. MLX support for full pre-training from scratch is underserved.
2. **Development hardware**: The main author's machine is a MacBook Air M4. MLX-first ensures every feature is tested on the development machine.
3. **Unified memory advantage**: Apple Silicon's unified memory architecture allows larger models relative to VRAM for equivalent cost.

## Consequences

- MLX-specific code paths must be maintained separately from PyTorch
- Some PyTorch features (Flash Attention 2, FSDP) are CUDA-only and will have MLX equivalents or be deferred
- Multi-GPU training (FSDP) is a CUDA-only feature for v1.0
