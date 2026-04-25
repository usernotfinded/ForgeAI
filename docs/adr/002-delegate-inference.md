# ADR-002: Delegate Inference to mlx-lm and llama.cpp

**Date**: 2026-04-22
**Status**: Accepted

## Context

ForgeAI's primary purpose is training orchestration. A production-grade inference server (KV cache management, streaming, continuous batching, quantization) is a separate engineering problem of comparable complexity.

## Decision

For v0.1 and v1.0, ForgeAI delegates most inference workflows to:
- `mlx-lm` for Apple Silicon (MLX backend)
- `llama.cpp` for CUDA and CPU (via GGUF export)

ForgeAI provides `forge chat <model>` as a thin wrapper over these tools, with a PyTorch fallback path for local checkpoint/debug scenarios.

## Rationale

1. `mlx-lm` and `llama.cpp` are mature, actively maintained, and handle quantization, streaming, and efficient KV cache well.
2. Building a comparable inference server would double the codebase scope without adding unique value.
3. ForgeAI's unique value is in training orchestration, hardware auto-detection, and cost estimation — not inference optimization.

## Consequences

- ForgeAI has a dependency on `mlx-lm` (optional) and `llama.cpp` (via subprocess or Python binding) for inference
- A native PyTorch chat path exists as a compatibility fallback, not as the primary optimized inference route
- Custom inference engine with GGUF/GPTQ/AWQ is explicitly out of scope for v0.1–v1.0
- Users who need production inference use vLLM, llama.cpp, or mlx-lm directly
