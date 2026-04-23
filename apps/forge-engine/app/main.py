"""
ForgeAI — Forge Engine
=======================
FastAPI service exposing model architecture info, hardware backend detection,
training status, and evaluation endpoints.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.architectures import get_architecture, list_architectures, GPTConfig
from app.core.backend import detect_backend, get_backend

SERVICE_NAME = os.getenv("SERVICE_NAME", "forge-engine")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    backend = detect_backend()
    print(f"[{SERVICE_NAME}] Starting up")
    print(f"[{SERVICE_NAME}] Backend: {backend.type.value.upper()} — {backend.device_name}")
    if backend.vram_gb:
        mem_label = "Unified RAM" if backend.unified_memory else "VRAM"
        print(f"[{SERVICE_NAME}] {mem_label}: {backend.vram_gb:.1f} GB")
    print(f"[{SERVICE_NAME}] Recommended preset: {backend.recommended_preset}")
    yield
    print(f"[{SERVICE_NAME}] Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="ForgeAI — Forge Engine",
    description="Model architecture registry and training orchestration service.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────


class BuildRequest(BaseModel):
    architecture: str = "transformer"
    preset: str | None = "forge-tiny"
    vocab_size: int | None = None
    context_length: int | None = None
    n_layer: int | None = None
    n_head: int | None = None
    n_kv_head: int | None = None
    n_embd: int | None = None
    dropout: float | None = None


class HardwareInfo(BaseModel):
    backend: str
    torch_device: str
    device_name: str
    vram_gb: float | None
    unified_memory: bool
    bf16_supported: bool
    flash_attention: bool
    mlx_available: bool
    recommended_preset: str
    recommended_dtype: str
    notes: list[str]


class PlanRequest(BaseModel):
    architecture: str = "transformer"
    params: str = "400M"
    data_path: str = "./data"
    kwh_cost: float = 0.30


class TrainingLogEntry(BaseModel):
    step: int
    loss: float | None = None
    val_loss: float | None = None
    val_perplexity: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None
    tokens_per_second: float | None = None
    total_tokens: int | None = None
    timestamp: float | None = None


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": SERVICE_NAME}


@app.get("/architectures", summary="List all available architectures and presets")
async def get_architectures() -> dict[str, Any]:
    return {"architectures": list_architectures()}


@app.post("/architectures/build", summary="Instantiate a model and return its stats")
async def build_model(req: BuildRequest) -> dict[str, Any]:
    try:
        overrides = {k: v for k, v in req.model_dump().items()
                     if k not in ("architecture", "preset") and v is not None}
        model = get_architecture(req.architecture, preset=req.preset, **overrides)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    config: GPTConfig = model.config
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(trainable_only=True)

    return {
        "architecture": req.architecture,
        "preset": req.preset,
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "n_embd": config.n_embd,
            "head_dim": config.head_dim,
            "ffn_hidden_dim": config.ffn_hidden_dim,
            "rope_theta": config.rope_theta,
            "tie_embeddings": config.tie_embeddings,
        },
        "parameters": {
            "total": total_params,
            "total_millions": round(total_params / 1e6, 1),
            "trainable": trainable_params,
            "trainable_millions": round(trainable_params / 1e6, 1),
        },
    }


@app.get("/hardware", response_model=HardwareInfo, summary="Detect available hardware and best backend")
async def get_hardware() -> HardwareInfo:
    b = get_backend()
    return HardwareInfo(
        backend=b.type.value,
        torch_device=b.torch_device,
        device_name=b.device_name,
        vram_gb=b.vram_gb,
        unified_memory=b.unified_memory,
        bf16_supported=b.bf16_supported,
        flash_attention=b.flash_attention,
        mlx_available=b.mlx_available,
        recommended_preset=b.recommended_preset,
        recommended_dtype=b.recommended_dtype,
        notes=b.notes,
    )


@app.post("/plan", summary="Estimate training time and cost")
async def create_plan(req: PlanRequest) -> dict[str, Any]:
    from app.training.planner import estimate_training
    b = get_backend()
    plan = estimate_training(
        arch=req.architecture,
        params=req.params,
        data_path=req.data_path,
        backend=b,
        kwh_cost=req.kwh_cost,
    )
    return {
        "architecture": plan.arch,
        "params": plan.params,
        "params_millions": round(plan.params / 1e6, 1),
        "backend": plan.backend_name,
        "device": plan.device_name,
        "estimated_tokens": plan.estimated_tokens,
        "estimated_hours": round(plan.estimated_hours, 1),
        "estimated_days": round(plan.estimated_hours / 24, 1),
        "electricity_kwh": round(plan.electricity_kwh, 1),
        "electricity_cost_eur": round(plan.electricity_kwh * plan.kwh_cost, 1),
        "checkpoint_size_gb": round(plan.checkpoint_size_gb, 1),
        "recommended_dtype": plan.recommended_dtype,
        "recommended_batch_size": plan.recommended_batch_size,
        "warnings": plan.warnings,
    }


@app.get("/training/logs", summary="Get training log entries for the web UI")
async def get_training_logs(
    log_file: str = "./checkpoints/run/train_log.jsonl",
    last_n: int = 500,
) -> dict[str, Any]:
    """Read training log entries for live monitoring in the web UI."""
    path = Path(log_file)
    if not path.exists():
        return {"entries": [], "note": "No training log found. Start a training run first."}

    entries: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Return last N entries
    return {"entries": entries[-last_n:], "total": len(entries)}


@app.get("/checkpoints", summary="List saved checkpoints")
async def list_saved_checkpoints(checkpoint_dir: str = "./checkpoints/run") -> dict[str, Any]:
    from app.checkpoints import list_checkpoints
    path = Path(checkpoint_dir)
    if not path.exists():
        return {"checkpoints": [], "note": "No checkpoint directory found."}
    return {"checkpoints": list_checkpoints(checkpoint_dir)}


# ── Chat / Inference ─────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    checkpoint_dir: str
    message: str
    conversation: list[str] = []
    tokenizer_dir: str | None = None
    temperature: float = 0.8
    max_tokens: int = 200
    top_k: int = 50
    top_p: float = 0.9


# Cache loaded models to avoid reloading on every request
_loaded_models: dict[str, tuple[Any, Any, dict[str, Any], str]] = {}


@app.post("/chat", summary="Generate a chat response from a loaded model")
async def chat_generate(req: ChatRequest) -> dict[str, Any]:
    """
    Generate a response from a ForgeAI checkpoint.

    Loads the model on first request and caches it for subsequent calls.
    The conversation history is passed in the request body so the server
    remains stateless (state lives in the frontend).
    """
    import torch

    cache_key = req.checkpoint_dir
    if cache_key not in _loaded_models:
        # Load model + tokenizer
        ckpt_path = Path(req.checkpoint_dir)
        if not ckpt_path.exists():
            raise HTTPException(404, f"Checkpoint not found: {req.checkpoint_dir}")

        resolved = ckpt_path
        if (ckpt_path / "latest").exists():
            resolved = (ckpt_path / "latest").resolve()

        meta_path = resolved / "metadata.json"
        if not meta_path.exists():
            raise HTTPException(400, f"No metadata.json in {resolved}")

        with open(meta_path) as f:
            meta = json.load(f)

        model_config = meta.get("model_config", {})
        arch = meta.get("architecture", "transformer")

        model = get_architecture(arch, **model_config)
        from app.checkpoints.manager import load_checkpoint
        b = get_backend()
        load_checkpoint(req.checkpoint_dir, model, device=b.torch_device)
        model = model.to(b.torch_device)
        model.eval()

        # Load tokenizer
        tokenizer = None
        if req.tokenizer_dir:
            from app.tokenizer import load_tokenizer
            tokenizer = load_tokenizer(req.tokenizer_dir)

        _loaded_models[cache_key] = (model, tokenizer, model_config, b.torch_device)

    model, tokenizer, model_config, device = _loaded_models[cache_key]

    if tokenizer is None and req.tokenizer_dir:
        from app.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(req.tokenizer_dir)
        _loaded_models[cache_key] = (model, tokenizer, model_config, device)

    if tokenizer is None:
        raise HTTPException(400, "No tokenizer loaded. Pass tokenizer_dir on first request.")

    # Build prompt from conversation history + current message
    conversation = list(req.conversation) + [req.message]
    prompt = "\n".join(conversation)

    encoded = tokenizer.encode(prompt)
    context_length = model_config.get("context_length", 2048)

    # Truncate if needed
    input_ids_list = encoded.ids
    max_input = context_length - req.max_tokens
    if len(input_ids_list) > max_input:
        input_ids_list = input_ids_list[-max_input:]

    input_ids = torch.tensor([input_ids_list], device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

    generated_ids = output_ids[0].tolist()[len(input_ids_list):]
    response = tokenizer.decode(generated_ids).strip()

    return {
        "response": response,
        "tokens_generated": len(generated_ids),
        "conversation": conversation + [response],
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


class EvalRequest(BaseModel):
    checkpoint_dir: str
    tokenizer_dir: str
    data_path: str
    max_batches: int = 50
    batch_size: int = 4


@app.post("/eval/perplexity")
async def eval_perplexity(req: EvalRequest) -> dict[str, Any]:
    """Compute perplexity on a binary data shard."""
    import torch
    from app.checkpoints.manager import load_checkpoint
    from app.evaluation.perplexity import compute_perplexity
    from app.tokenizer import load_tokenizer
    from app.data.dataset import create_dataloader

    b = get_backend()

    # Load tokenizer
    try:
        load_tokenizer(req.tokenizer_dir)
    except Exception as exc:
        raise HTTPException(400, f"Cannot load tokenizer: {exc}") from exc

    # Load model
    ckpt_path = Path(req.checkpoint_dir)
    meta_path = ckpt_path / "metadata.json"
    if not meta_path.exists():
        # Try latest symlink
        latest = ckpt_path / "latest"
        if latest.exists():
            meta_path = latest.resolve() / "metadata.json"
        if not meta_path.exists():
            raise HTTPException(400, f"No metadata.json in {req.checkpoint_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    model_config = meta.get("model_config", {})
    arch = meta.get("architecture", "transformer")
    model = get_architecture(arch, **model_config)
    load_checkpoint(req.checkpoint_dir, model, device=b.torch_device)
    model = model.to(b.torch_device)
    model.eval()

    # Build dataloader
    try:
        dataloader = create_dataloader(
            data_dir=req.data_path,
            batch_size=req.batch_size,
            context_length=model_config.get("context_length", 512),
        )
    except Exception as exc:
        raise HTTPException(400, f"Cannot load data: {exc}") from exc

    dtype = torch.bfloat16 if b.bf16_supported else torch.float16 if b.torch_device != "cpu" else torch.float32

    result = compute_perplexity(
        model=model,
        dataloader=dataloader,
        device=b.torch_device,
        max_batches=req.max_batches,
        dtype=dtype,
    )
    return result
