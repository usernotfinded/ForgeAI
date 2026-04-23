# ForgeAI

**Train language models at any scale your hardware supports — from 50M on a laptop to 200B on a cluster. Your data, your hardware, your model.**

ForgeAI is an open-source, MLX-first training orchestrator for language models. It adapts to whatever hardware you have — a MacBook Air, a gaming PC with a 4090, a rack of H100s — and configures the training pipeline accordingly. There are no artificial ceilings on model size. If you have the hardware and the patience, you can train it.

ForgeAI is the Blender of AI training. Blender doesn't refuse to render a scene because your PC is slow — it estimates the time and lets you decide. ForgeAI doesn't refuse to train a 70B model because you have one GPU — it tells you it will take 14 months and lets you proceed if you accept the trade-off. Or you add more GPUs.

The core value proposition is **sovereignty**: you own the data, the training process, the weights, and the serving infrastructure. A 7B model trained on your proprietary documents can be more useful *to you* than Claude, because it is specialized and private. Nobody can revoke your access, change the terms of service, read your prompts, or decide what you're allowed to ask.

---

## Why ForgeAI Exists

There are good tools for fine-tuning existing models. ForgeAI does something different.

| Tool | Strength | ForgeAI's difference |
|------|----------|---------------------|
| **Axolotl** | Fine-tuning Llama/Mistral on CUDA | ForgeAI trains from scratch, MLX-first, scales to multi-node |
| **Unsloth** | Fast LoRA fine-tuning | ForgeAI supports full pre-training, not just fine-tuning |
| **MLX-LM** | Apple Silicon inference and fine-tuning | ForgeAI adds pre-training from scratch, auto-scaling, multi-architecture |
| **nanoGPT** | Educational, minimal code | ForgeAI adds hardware auto-tuning, multi-GPU, cost estimation, evaluation |
| **LLaMA-Factory** | Web UI for fine-tuning | ForgeAI covers the full pipeline: data → tokenizer → pre-training → eval → serving |

**ForgeAI's position**: end-to-end sovereignty (data → tokenizer → training → evaluation → serving) with hardware-adaptive orchestration, from a single laptop to a multi-node cluster, MLX-first.

---

## Scope and Honest Expectations

Read this section before anything else.

### What You Can Realistically Train

| Hardware | Realistic Max (from scratch) | Training Time | Example Use Case |
|----------|------------------------------|---------------|------------------|
| MacBook Air M4 16GB | 400M – 1B | Days – weeks | Personal assistant, domain chatbot, experimental research |
| RTX 4090 24GB | 7B – 14B | Weeks – months | Home automation AI, private coding assistant |
| 4× RTX 4090 | 14B – 30B | Months | Company domain expert, internal document Q&A |
| 8× H100 80GB | 70B – 200B | Months – year | Sovereign AI for organizations, no cloud dependency |

### What to Expect from the Models

- **400M model**: coherent text generation, basic Q&A on trained domain. Will not match ChatGPT. Can be highly specialized and surprisingly useful for narrow tasks.
- **7B model**: genuinely useful for domain-specific work. Comparable to GPT-2 / early GPT-3 quality. Good for home automation, local RAG, code completion on your codebase.
- **70B model**: approaching GPT-3.5 quality if trained well on good data. Requires serious compute and serious data curation.

### What ForgeAI Does NOT Do

- Does not ship large pre-trained models. You train your own, or fine-tune existing open models.
- Does not guarantee quality. Training is hard. Data quality, hyperparameters, and compute all matter. ForgeAI gives you the tools and guardrails, not the guarantees.
- Does not make a 400M model "as good as Claude." That is physically impossible. But it makes a 400M model that knows YOUR domain better than Claude does.

---

## Sovereignty Use Cases

This is not about doomsday scenarios. It is about control over your AI stack in normal times.

- **A medical research lab** has 300K patient records under HIPAA. Using OpenAI's API is a compliance risk they cannot take. They train a 14B model on their own 4-GPU server with ForgeAI. The data never leaves their network. The model is specialized in their diagnostic ontology and outperforms general-purpose models on their internal benchmarks.

- **A robotics hobbyist** running Home Assistant on a Raspberry Pi cluster has an RTX 4090 in their workstation. They train a 7B model to understand their Zigbee device topology, interpret Frigate NVR camera feeds, and learn household patterns. The model runs 24/7 on a Jetson Orin. No cloud, no subscription, no latency.

- **A computational biologist** wants a domain model for protein function prediction. Trains a 14B model on curated UniProt entries and PubMed abstracts, iterates on tokenizer vocabulary and data mix. No API rate limits, no per-token billing, full control over the training corpus.

- **A hobbyist** with a MacBook Air wants to understand how language models actually work. Trains a 50M model from scratch on TinyStories, inspects every gradient, experiments with learning rate schedules, and reads the source code line by line.

- **A national defense research institute** in a mid-size country needs sovereign AI capability without dependency on US or Chinese cloud providers. They operate an air-gapped 8× H100 cluster and use ForgeAI to train a 70B model on classified intelligence documents. This is a real and growing market.

ForgeAI also works fully offline — but that is a side effect of being local, not the main pitch.

---

## How It Works

### Hardware-Adaptive Training

ForgeAI auto-detects your hardware and configures optimal training settings. You do not need to know what FSDP is, what mixed-precision dtype to use, or how to set the batch size for your VRAM.

```bash
$ forge plan --arch transformer --params 7B --data ./my-corpus/
┌──────────────────────────────────────────────────────┐
│  ForgeAI — Training Plan                            │
│                                                      │
│  Hardware   : NVIDIA RTX 4090 (24GB VRAM)            │
│  Backend    : CUDA (PyTorch)                         │
│  Model      : Transformer 7B (32 layers, GQA, RoPE) │
│  Precision  : bfloat16                               │
│  Batch size : 4 (auto, grad accumulation: 8)         │
│  Optimizer  : AdamW (8-bit via bitsandbytes)         │
│  Tokens     : ~12B (estimated from corpus)           │
│                                                      │
│  Estimated time  : ~45 days                          │
│  Estimated cost  : ~€85 electricity (€0.30/kWh)     │
│  Checkpoint size : ~14 GB per save                   │
│  Total disk      : ~180 GB (checkpoints + data)      │
│                                                      │
│  Proceed? [y/N]                                      │
└──────────────────────────────────────────────────────┘
```

Key features:
- **Cost and time estimation before training starts** — no surprises at hour 200
- **Auto-configuration** — dtype, batch size, gradient accumulation, optimizer chosen for your hardware
- **No artificial limits** — if you ask for 70B on a 4090, ForgeAI says "max ~14B on your VRAM, want to proceed with 14B?" instead of crashing
- **Override everything** — advanced users can set any parameter manually

### Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                      ForgeAI Pipeline                         │
│                                                                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Corpus  │─▶│ Tokenizer │─▶│ Training │─▶│  Evaluation  │  │
│  │  (text)  │  │ (BPE/SPM) │  │  Loop    │  │  (online +   │  │
│  └──────────┘  └───────────┘  └────┬─────┘  │   offline)   │  │
│                                    │        └──────┬───────┘  │
│                              ┌─────▼─────┐         │          │
│                              │Checkpoint │◀────────┘          │
│                              └─────┬─────┘                    │
│                                    │                          │
│                              ┌─────▼───────────┐             │
│                              │   Inference     │             │
│                              │  (mlx-lm /      │             │
│                              │   llama.cpp)    │             │
│                              └─────────────────┘             │
└────────────────────────────────────────────────────────────────┘
```

### Forge Stretch (v1)

`forge stretch` persistently extends the context window of an **existing compatible model**.

- Method v1: **YaRN scaling only**
- Persistence v1: **`adapter_plus_manifest` only** (not `full_checkpoint`)
- The target context length must be **strictly greater** than the model's native context
- Validation v1 is **local/proxy**: `forge stretch` includes specific checks for its own workflow, but the repository as a whole does not yet feature a general, reusable long-context benchmark suite (like Needle-in-a-Haystack) in the official `forge eval` module.

Full documentation, examples, and artifact list:
- [`docs/stretch.md`](./docs/stretch.md)

---

## System Design

ForgeAI is a local, single-user tool. No auth service, no API gateway, no microservices. You are the only user.

```
┌──────────────────────────────────────────────────┐
│                  ForgeAI Stack                   │
│                                                  │
│  ┌───────────┐          ┌──────────────────────┐ │
│  │   forge   │          │  Local Web UI        │ │
│  │   (CLI)   │          │  (Next.js :3000)     │ │
│  └─────┬─────┘          └──────────┬───────────┘ │
│        │                           │             │
│        └─────────┬─────────────────┘             │
│                  │                               │
│        ┌─────────▼──────────┐                    │
│        │    forge-engine    │                    │
│        │    (Python core)   │                    │
│        │                    │                    │
│        │  - architectures   │                    │
│        │  - tokenizer       │                    │
│        │  - training loop   │                    │
│        │  - evaluation      │                    │
│        │  - checkpoints     │                    │
│        │  - auto-tuning     │                    │
│        └─────────┬──────────┘                    │
│                  │                               │
│     ┌────────────┼────────────┐                  │
│     │            │            │                  │
│   ┌─▼──┐   ┌────▼──────┐  ┌──▼────┐             │
│   │MLX │   │CUDA/FSDP  │  │ CPU   │             │
│   └────┘   └───────────┘  └───────┘             │
└──────────────────────────────────────────────────┘
```

| Component | Role |
|-----------|------|
| **forge (CLI)** | `forge train`, `forge eval`, `forge serve`, `forge plan` — scriptable, no UI required |
| **forge-engine** | Python core: architectures, tokenizer, training loop, evaluation, checkpoints, hardware auto-tuning |
| **web (local UI)** | Optional Next.js dashboard (MVP): basic training curves and cost tracking (chat and eval routes are still WIP) |
| **Inference** | Delegated to `mlx-lm` (Apple Silicon) and `llama.cpp` (cross-platform) for v0.1 |

---

## Architecture

### v0.1: Transformer (decoder-only)

One architecture, done properly. Modern GPT-style defaults following LLaMA/Mistral design:

```
Decoder-only Transformer
  ├── Token embedding (no learned positional — RoPE instead)
  ├── N × TransformerBlock
  │     ├── RMSNorm (pre-norm)
  │     ├── Grouped Query Attention (GQA) + RoPE
  │     ├── RMSNorm
  │     └── SwiGLU FFN
  └── RMSNorm → LM head (weight-tied to embedding)
```

The implementation in `forge-engine/architectures/transformer.py` is ~300 lines with inline comments explaining each design decision. The codebase is structured to be pluggable — adding an architecture means implementing a `BaseModel` interface and registering it.

### Architecture Comparison

| Architecture | Status | Best For | Trade-off |
|---|---|---|---|
| **Transformer** | ✅ v0.1 | General purpose, most tooling/research support | Quadratic attention cost with sequence length |
| **Mamba (SSM)** | 🚧 Planned v1.0+ | Long documents, lower memory | Newer, less tooling, MLX support still maturing |
| **RWKV** | 🚧 Planned v1.0+ | Fast inference, CPU-friendly | Less proven at scale |
| **MoE** | 🚧 Community-driven | Efficient scaling, multi-domain | Complex routing, harder to train |

Only one additional architecture will be added in v1.0, chosen based on community demand. The others ship when someone builds them and the PR passes review.

---

## Backend

MLX is the primary development target. CUDA is fully supported but not the development focus. This is a deliberate niche choice — Axolotl, Unsloth, and LLaMA-Factory already dominate CUDA fine-tuning.

| Priority | Backend | Trigger | Training | Inference | Multi-GPU |
|----------|---------|---------|----------|-----------|-----------|
| 1 | **MLX** | Apple Silicon + `mlx` installed | ✅ Native | via `mlx-lm` | N/A (unified memory) |
| 2 | **CUDA** | NVIDIA GPU | ✅ PyTorch | via `llama.cpp` | ✅ FSDP (v1.0) |
| 3 | **MPS** | Apple Silicon, no MLX | ⚠ Slower | via `llama.cpp` | N/A |
| 4 | **CPU** | No GPU | ❌ Too slow for >10M | via `llama.cpp` | N/A |

Auto-detection runs at startup: backend, dtype (bfloat16 on MLX/Ampere+, float16 on older CUDA, float32 on CPU), and recommended model size are set automatically.

### Hardware Examples

| Hardware | Backend | Max Model (training) | Notes |
|----------|---------|---------------------|-------|
| MacBook Air M4 16GB | MLX | 400M – 1B | Development machine. All v0.1 claims tested here. |
| Mac Studio M2 Ultra 192GB | MLX | 7B – 14B | Unified memory makes large models feasible. |
| RTX 4090 24GB | CUDA | 7B – 14B | Single GPU, gradient checkpointing required for 14B. |
| 4× RTX 4090 | CUDA + FSDP | 14B – 30B | v1.0 target. |
| 8× H100 80GB | CUDA + FSDP | 70B – 200B | v2.0 target, multi-node. |

---

## Starter Models

ForgeAI is a training tool, not a model distributor. The primary workflow is: bring your data, train your model.

For testing the pipeline, ForgeAI can import existing open-source models into ForgeAI checkpoint format:

| Model | Source | Params | License | Notes |
|-------|--------|--------|---------|-------|
| `smollm-135m` | HuggingFace SmolLM | 135M | Apache 2.0 | Good for testing the pipeline |
| `qwen2.5-0.5b` | Alibaba Qwen team | 494M | Apache 2.0 | Strong multilingual |
| `tinyllama-1b` | TinyLlama project | 1.1B | Apache 2.0 | Needs ≥16GB RAM |

These are not ForgeAI models. Full credit goes to the upstream authors. ForgeAI provides a converter and a consistent interface for fine-tuning.

ForgeAI may also include 1–2 tiny demo models (50M–150M) trained on TinyStories for instant pipeline testing. These are labeled as toy demos — do not expect useful output from them.

---

## Evaluation and Benchmarking

Training without evaluation is flying blind. ForgeAI treats evaluation as a first-class part of every training run.

### During Training (online)

- **Validation perplexity** — logged every N steps on a held-out split
- **Loss and learning rate curves** — real-time in the local web UI
- **Gradient norm** — tracked to catch instabilities before they waste hours of compute

### Offline Benchmarks

- **TinyStories eval** — coherence score on held-out samples (~2 min, good for models <1B)
- **HellaSwag-mini** — 1000-sample subset, 0-shot accuracy (useful for ≥100M params)
- **ARC-easy** (optional) — common-sense reasoning, lightweight

These benchmarks are not impressive by frontier standards. They are useful for comparing your runs against each other, catching regressions, and confirming that training actually converged.

### Cost Tracking

- **GPU-hours consumed** per run
- **Estimated electricity cost** (configure your €/kWh rate)
- **Checkpoint size** on disk, total storage consumed

### Comparison Mode

Train two models with different configs and compare metrics side by side:
```bash
forge eval compare ./checkpoints/run-a/latest ./checkpoints/run-b/latest
```

Exports comparison as markdown, viewable in the web UI or as a file.

---

## Quick Start

### Requirements

- Python 3.11+
- `pip install mlx mlx-lm` (Apple Silicon) or `pip install torch` (CUDA)
- Node.js 20+ (optional, for the local web UI)

### Install

```bash
git clone https://github.com/usernotfounded/forgeai.git
cd forgeai
pip install -e apps/forge-engine
```

### Chat with a pre-existing model

```bash
# Import an open-source model into ForgeAI format
forge model pull smollm-135m

# Chat via mlx-lm
forge chat smollm-135m
```

### Train from scratch

```bash
# 1. See what your hardware can handle
forge plan --arch transformer --params 400M --data ./my-corpus/

# 2. Prepare data
forge data prepare ./my-corpus/ --output ./data/processed/

# 3. Train a tokenizer
forge tokenizer train --data ./data/processed/ --vocab-size 8000

# 4. Train
forge train \
  --arch transformer \
  --params 400M \
  --data ./data/processed/ \
  --tokenizer ./tokenizers/my-tokenizer/ \
  --output ./checkpoints/my-run/

# 5. Evaluate
forge eval --checkpoint ./checkpoints/my-run/latest --benchmark tinystories hellaswag-mini

# 6. Chat
forge chat ./checkpoints/my-run/latest
```

### Local Web UI (MVP)

```bash
cd apps/web && npm install && npm run dev
# Open http://localhost:3000 — view training curves, run evaluations, chat
# Note: The dashboard and API are functional, but some secondary pages (e.g. /eval) are still WIP.
```

### Minimal Smoke Test

If you just cloned the repository, you can verify that the core engine works on your system by running the minimal smoke test. It runs the entire pipeline (data → tokenizer → train → chat) on a toy text file and finishes in seconds:

```bash
./scripts/smoke_test.sh
```

---

## Project Structure

```
forgeai/
├── apps/
│   ├── forge-engine/              # Python core (pip install -e .)
│   │   ├── app/
│   │   │   ├── architectures/     # Transformer (v0.1) + registry for future archs
│   │   │   │   ├── transformer.py # GPT-style decoder-only (~300 lines, commented)
│   │   │   │   └── registry.py    # Architecture registry + preset configs
│   │   │   ├── core/
│   │   │   │   └── backend.py     # Auto-detect MLX / CUDA / MPS / CPU
│   │   │   ├── tokenizer/         # BPE and SentencePiece training
│   │   │   ├── training/          # Training loop, optimizer, scheduler, auto-tuning
│   │   │   ├── evaluation/        # Perplexity, HellaSwag-mini, TinyStories eval
│   │   │   ├── data/              # Dataset loading, preprocessing, batching
│   │   │   └── checkpoints/       # Save / load / convert / import from HF
│   │   └── cli/                   # forge CLI entry points
│   └── web/                       # Local Next.js dashboard (optional)
│       └── src/
│           ├── app/               # Training monitor, loss curves, cost tracker, chat
│           └── components/
├── papers/                        # Research papers reading list (see papers/README.md)
│   ├── architectures/
│   ├── training/
│   ├── alignment/
│   ├── tokenization/
│   ├── inference/
│   └── labs/                      # By lab: deepseek, moonshotai, google, meta, mistral
├── docs/
│   └── adr/                       # Architecture Decision Records
├── scripts/                       # Dev helpers
├── .gitignore
└── turbo.json
```

---

## Research Papers

The `papers/` directory is a learning resource organized by topic and lab. See [papers/README.md](papers/README.md) for the annotated reading list (~35 papers covering architectures, training, alignment, tokenization, and inference).

PDFs are gitignored — they are already compressed, zipping them saves only 5–15% and adds friction. The reading list in `papers/README.md` has direct arXiv links for every paper. Download what you need. If you want PDFs in the repo, use [git-lfs](https://git-lfs.com).

---

## Roadmap

Timelines assume P90 (worst-case) planning. Each phase has a binary "done" definition and a kill-switch condition.

---

### Phase 1 (v0.1) — Single-GPU Training

**Timeline**: 9 months (P50: 6 months, P90: 12 months)

**Done when**: A Transformer model can be trained from scratch on TinyStories, reach <50 perplexity on the validation set, and generate grammatical text. The core path (data → tokenizer → train → eval → chat) works end-to-end as a technical MVP on a MacBook Air M4 16GB and on a single CUDA GPU, with minimal manual configuration required.

**Kill-switch**: If training loop is not functional on both MLX and CUDA by month 12, pivot to fine-tuning-only tool and drop from-scratch pre-training.

| Deliverable | Status |
|-------------|--------|
| Transformer architecture (RoPE, GQA, SwiGLU) | ✅ |
| Architecture registry + presets | ✅ |
| Backend auto-detection (MLX / CUDA / MPS / CPU) | ✅ |
| BPE tokenizer training | ✅ |
| Training loop (MLX native + PyTorch CUDA) | ✅ (PyTorch; MLX native planned) |
| Mixed precision + gradient checkpointing | ✅ |
| Validation perplexity tracking | ✅ |
| Checkpoint save / load / resume | ✅ |
| Hardware auto-tuning (batch size, dtype, optimizer) | ✅ |
| `forge plan` — cost/time estimation | ✅ |
| `forge train` CLI | ✅ |
| `forge eval` — TinyStories + HellaSwag-mini | ✅ |
| `forge chat` via `mlx-lm` / `llama.cpp` | ✅ |
| Basic web UI (loss curves, chat) | ✅ (Dashboard & API present; secondary pages WIP) |

---

### Phase 2 (v1.0) — Multi-GPU + Fine-tuning + Second Architecture

**Timeline**: +12 months after v0.1 (P50: +9 months, P90: +18 months)

**Done when**: Multi-GPU training (FSDP, 2–8 GPUs) works end-to-end. LoRA fine-tuning works on a redistributed model on 8GB unified RAM. A second architecture (Mamba or RWKV) passes the same end-to-end test as Transformer. A 7B model is successfully trained on a multi-GPU setup.

**Kill-switch**: If multi-GPU FSDP is not working 18 months into Phase 2, stay single-GPU and focus on quality, UX, and fine-tuning depth instead of scaling.

| Deliverable | Status |
|-------------|--------|
| Multi-GPU training (FSDP, 2–8 CUDA GPUs) | ⬜ |
| LoRA / QLoRA fine-tuning (MLX + CUDA) | ⬜ |
| Second architecture (Mamba or RWKV, based on community demand) | ⬜ |
| Model converter (HuggingFace → ForgeAI format) | ✅ |
| `forge model pull` for open-source models | ✅ |
| Advanced cost estimation (electricity, GPU-hours, disk) | ✅ |
| Model comparison mode (`forge eval compare`) | ✅ |
| 7B model trained on multi-GPU and evaluated | ⬜ |

---

### Phase 3 (v2.0+) — Multi-Node + Production Hardening

**Done when**: Training can span multiple machines. Fault tolerance handles node failures mid-training. At least 3 external developers can clone the repo and complete a training run on their own data within 2 hours without asking for help.

**Kill-switch**: If multi-node is not feasible with the contributor base by v2.0 target, scope down to single-server multi-GPU and focus on documentation and usability.

| Deliverable | Status |
|-------------|--------|
| Multi-node training (2+ machines) | ⬜ |
| Fault tolerance (checkpoint resume on node failure) | ⬜ |
| Third architecture (community-driven) | ⬜ |
| End-to-end documentation and tutorials | ⬜ |
| GitHub CI (lint, unit tests, smoke test) | 🚧 (Local smoke test script added) |
| 2-hour onboarding benchmark (3 external testers) | ⬜ |

---

## Out of Scope

These are explicitly not planned for v0.1 – v1.0. They may happen later or as community contributions.

| Feature | Reason |
|---------|--------|
| Custom inference server (KV cache, streaming, quantization) | `mlx-lm` and `llama.cpp` do this well. Not worth reinventing. |
| Kubernetes / Terraform deployment | ForgeAI is a local tool. Cloud orchestration is a different product. |
| Auth, 2FA, CSRF, rate limiting, GDPR | Single-user local tool. No multi-tenancy, no server exposure. |
| RLHF / DPO alignment | Requires reward models, preference data, and significant compute. Out of reach for v0.x. |
| MoE architecture | Complex routing, harder to train correctly. Community-contributed if demand exists. |
| TensorRT / ONNX export | Worth exploring later. Not for v0.x – v1.0. |
| Pre-trained models above 400M | Requires compute budget not currently available. Users train their own. |
| Competing with Claude/GPT-4 on general capability | Not the goal. ForgeAI optimizes for sovereignty and specialization, not frontier performance. If someone opens an issue asking "why isn't this as good as ChatGPT," the answer is in this table. |

---

## Community-Driven Exploration

Not committed deliverables. These happen if contributors build them.

- Additional architectures beyond Transformer + one other
- Advanced alignment (PPO, GRPO, constitutional AI)
- Model merging and ensembling
- Distributed training across heterogeneous hardware (mixed MLX + CUDA)
- Dataset curation and deduplication tools
- Quantization-aware training

---

## Security

ForgeAI is a local tool. There is no auth layer. Do not expose the web UI to the public internet. It binds to `localhost:3000` by default.

If you want to expose the inference endpoint on your LAN, use a basic API key via environment variable. This is optional and not the default.

---

## Papers

See [papers/README.md](papers/README.md) for an annotated reading list organized by topic (architectures, training, alignment, tokenization, inference) and by lab (DeepSeek, Moonshot/Kimi, Google, Meta, Mistral).

PDFs are gitignored. Download what you need via the arXiv links in the reading list.

---

## FAQ

**Q: Why would I train from scratch instead of fine-tuning Llama?**

Fine-tuning is faster and usually sufficient. Train from scratch when: (1) you need a custom tokenizer for a low-resource language or specialized domain vocabulary, (2) you want complete control over the architecture and training dynamics, (3) you are learning how LLMs work from the ground up, or (4) licensing compliance requires no derivative weights from a specific base model.

**Q: Can I really train a useful model on a MacBook?**

Yes, at the 50M–400M scale. "Useful" depends on your use case. A 400M model will not write code like Claude, but it can learn your writing style, summarize domain-specific documents, classify support tickets, or control simple automation tasks. Many practical applications do not need a frontier model.

**Q: Is this legal? Can I train on copyrighted data?**

ForgeAI is a tool. What you train on is your responsibility. Many jurisdictions have fair use or text-and-data-mining (TDM) exemptions for research (EU AI Act Article 4, US fair use doctrine). Consult a lawyer if you plan to train on non-permissive data for commercial use. ForgeAI does not include any datasets — you bring your own.

**Q: How is this different from just running nanoGPT?**

nanoGPT is a minimal educational implementation (~300 lines). ForgeAI builds on the same pedagogical spirit but adds: hardware auto-detection and configuration, cost/time estimation before training, evaluation benchmarks, checkpoint management, a local web UI for monitoring, and a roadmap toward multi-GPU scaling. Think of it as nanoGPT grown up into a usable tool.

---

## License

MIT — see [LICENSE](LICENSE) for details.
