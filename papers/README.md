# Research Papers

This folder contains the foundational research papers behind ForgeAI's architectures, training methods, and design decisions.

**Format**: Papers are stored as `.zip` files (PDF inside). Plain PDFs are gitignored to keep the repo lightweight — `.zip` files are tracked.

---

## 📁 Folder Structure

```
papers/
├── architectures/     # Model architecture papers (Transformer, Mamba, RWKV, MoE, ...)
├── training/          # Pre-training, scaling laws, optimization, data
├── alignment/         # RLHF, DPO, SFT, instruction tuning
├── tokenization/      # BPE, SentencePiece, tokenizer design
├── inference/         # KV cache, quantization, speculative decoding, serving
└── labs/              # Papers grouped by research lab
    ├── deepseek/      # DeepSeek series (DeepSeek-V2, V3, R1, MoE, Coder, ...)
    ├── moonshotai/    # Moonshot AI / Kimi (Kimi k1.5, MoonshotAI papers)
    ├── google/        # Google DeepMind (Gemini, Gemma, PaLM, T5, BERT, ...)
    ├── meta/          # Meta AI (LLaMA 1/2/3, OPT, Llama Guard, ...)
    └── mistral/       # Mistral AI (Mistral 7B, Mixtral MoE, ...)
```

---

## 📚 Reading List

### Architectures

| Paper | Year | What It Introduces | Folder |
|-------|------|-------------------|--------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The original Transformer | `architectures/` |
| [Language Models are Unsupervised Multitask Learners (GPT-2)](https://openai.com/research/better-language-models) | 2019 | Decoder-only LM at scale | `architectures/` |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | 2023 | State Space Model alternative to attention | `architectures/` |
| [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) | 2023 | Linear-time RNN with transformer performance | `architectures/` |
| [Outrageously Large Neural Networks (MoE)](https://arxiv.org/abs/1701.06538) | 2017 | Mixture of Experts sparse architecture | `architectures/` |
| [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | 2023 | Grouped Query Attention (KV cache reduction) | `architectures/` |
| [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) | 2021 | Rotary positional embeddings | `architectures/` |
| [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) | 2020 | SwiGLU activation for FFN layers | `architectures/` |

### Training & Scaling

| Paper | Year | What It Introduces | Folder |
|-------|------|-------------------|--------|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | How loss scales with compute/data/params | `training/` |
| [Chinchilla: Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) | 2022 | Optimal tokens-per-parameter ratio | `training/` |
| [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) | 2022 | IO-aware attention (10x faster) | `training/` |
| [ZeRO: Memory Optimizations for Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 2019 | Distributed training memory reduction | `training/` |
| [TiDAR](https://arxiv.org/abs/2501.12571) | 2025 | Training data attribution and routing | `training/` |
| [Slime: Scalable Lightweight Instruction Model Evolution](https://arxiv.org/abs/2504.07703) | 2025 | Efficient instruction dataset curation | `training/` |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | Parameter-efficient fine-tuning | `training/` |
| [QLoRA: Efficient Fine-tuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit fine-tuning on consumer GPUs | `training/` |

### Alignment

| Paper | Year | What It Introduces | Folder |
|-------|------|-------------------|--------|
| [InstructGPT: Training LMs to Follow Instructions with RLHF](https://arxiv.org/abs/2203.02155) | 2022 | RLHF for instruction following | `alignment/` |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | 2023 | Simpler alternative to RLHF | `alignment/` |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | 2022 | Generating instruction data from the model itself | `alignment/` |

### Tokenization

| Paper | Year | What It Introduces | Folder |
|-------|------|-------------------|--------|
| [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) | 2016 | Byte-Pair Encoding for tokenization | `tokenization/` |
| [SentencePiece: A Simple Language Independent Subword Tokenizer](https://arxiv.org/abs/1808.06226) | 2018 | Language-agnostic tokenizer | `tokenization/` |

### Inference & Serving

| Paper | Year | What It Introduces | Folder |
|-------|------|-------------------|--------|
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 2023 | vLLM paged KV cache | `inference/` |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) | 2022 | Draft-then-verify for faster generation | `inference/` |
| [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) | 2022 | 4-bit weight quantization | `inference/` |

---

## 🏭 By Lab

### DeepSeek (`labs/deepseek/`)

| Paper | Year | Description |
|-------|------|-------------|
| [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | 2024 | 7B/67B open models, scaling insights |
| [DeepSeek-MoE](https://arxiv.org/abs/2401.06066) | 2024 | Fine-grained expert segmentation for MoE |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 2024 | MLA attention (Multi-head Latent Attention) |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 | 671B MoE, 37B active, state-of-the-art open |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | Reasoning via RL without SFT |
| [DeepSeek-Coder](https://arxiv.org/abs/2401.14196) | 2024 | Code-specialized model series |

### Moonshot AI / Kimi (`labs/moonshotai/`)

| Paper | Year | Description |
|-------|------|-------------|
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 2025 | Long-context RL reasoning, multimodal |
| [MoonshotAI Technical Report](https://github.com/MoonshotAI/Kimi-k1.5) | 2025 | Scaling RL with long-CoT |

### Google DeepMind (`labs/google/`)

| Paper | Year | Description |
|-------|------|-------------|
| [Gemma](https://arxiv.org/abs/2403.08295) | 2024 | Lightweight open models based on Gemini research |
| [Gemma 2](https://arxiv.org/abs/2408.00118) | 2024 | Knowledge distillation, GQA |
| [PaLM: Pathways Language Model](https://arxiv.org/abs/2204.02311) | 2022 | 540B, SwiGLU, multi-task |
| [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683) | 2019 | Text-to-text unified framework |

### Meta AI (`labs/meta/`)

| Paper | Year | Description |
|-------|------|-------------|
| [LLaMA: Open and Efficient Foundation LMs](https://arxiv.org/abs/2302.13971) | 2023 | RoPE, RMSNorm, SwiGLU open models |
| [LLaMA 2](https://arxiv.org/abs/2307.09288) | 2023 | RLHF, grouped query attention |
| [LLaMA 3](https://arxiv.org/abs/2407.21783) | 2024 | 405B, extended RoPE (theta=500k), GQA |

### Mistral AI (`labs/mistral/`)

| Paper | Year | Description |
|-------|------|-------------|
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | GQA + sliding window attention |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 2024 | 8x7B sparse MoE, outperforms LLaMA 2 70B |

---

## How to Add Papers

1. Download the PDF from arXiv
2. Zip it with a clear name: `author_year_short-title.zip`
3. Drop it in the appropriate subfolder
4. Commit the `.zip` — it's tracked by git

```bash
# Example: download and zip Mamba paper
wget https://arxiv.org/pdf/2312.00752.pdf
zip gu_2023_mamba.zip 2312.00752.pdf
mv gu_2023_mamba.zip papers/architectures/
rm 2312.00752.pdf
```

> **Why .zip?** Plain PDFs (10–50MB each) bloat the repo for anyone who clones it.
> Zipping saves 5–15% and keeps the file trackable in git without needing LFS.
> Use `unzip` to read the paper normally: `unzip gu_2023_mamba.zip`
