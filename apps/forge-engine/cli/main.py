"""
forge — ForgeAI CLI

Usage:
    forge wizard --data ./corpus/
    forge stretch --model ./models/qwen2.5-0.5b --target-context 131072
    forge plan  --arch transformer --params 400M --data ./corpus/
    forge train --arch transformer --params 400M --data ./corpus/ --tokenizer ./tokenizers/tok --output ./checkpoints/run-1/
    forge eval  ./checkpoints/run-1/latest --benchmark tinystories hellaswag-mini --tokenizer ./tokenizers/tok
    forge chat  ./checkpoints/run-1/latest
    forge model pull smollm-135m
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="forge",
    help="ForgeAI — local-first CLI workbench for planning, training, and evaluating language models.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ── forge plan ────────────────────────────────────────────────────────────────


@app.command()
def plan(
    arch: str = typer.Option("transformer", "--arch", help="Model architecture"),
    params: str = typer.Option(..., "--params", help="Approximate parameter count (e.g. 400M, 7B)"),
    data: str = typer.Option(
        ...,
        "--data",
        help=(
            "Path to dataset (raw file/dir or prepared dataset directory). "
            "If metadata.json exists, planner reports dataset facts."
        ),
    ),
    kwh_cost: float = typer.Option(0.30, "--kwh-cost", help="Electricity cost in €/kWh"),
):
    """
    Estimate training time, cost, and hardware requirements BEFORE committing to a run.
    Detects your runtime backend and proposes conservative defaults.
    """
    from app.core.backend import get_backend
    from app.training.planner import estimate_training

    backend = get_backend()
    console.print("\n[dim]Detected backend:[/dim]")
    console.print(str(backend))
    console.print()

    try:
        plan_result = estimate_training(
            arch=arch, params=params, data_path=data, backend=backend, kwh_cost=kwh_cost
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(
        "[dim]Planner note: tempo/costo sono stime euristiche. "
        "I metadati dataset (se presenti) sono usati come contesto informativo.[/dim]"
    )
    console.print()
    plan_result.print_summary(console)


# ── forge wizard ──────────────────────────────────────────────────────────────


@app.command()
def wizard(
    data: str | None = typer.Option(
        None,
        "--data",
        help="Path al dataset (file o cartella). Se omesso, il wizard lo chiede interattivamente.",
    ),
    session_dir: str = typer.Option(
        "./.forge/wizard",
        "--session-dir",
        help="Directory dove salvare stato e artefatti del wizard.",
    ),
    auto_start: bool = typer.Option(
        False,
        "--auto-start",
        help="Esegue subito i passi locali (tokenizer + preparazione dati) dopo la conferma finale.",
    ),
):
    """
    Wizard semi-automatico locale con resume affidabile e consenso stratificato.
    Nota v1 adattamento: supporta solo checkpoint ForgeAI nativi compatibili con preset forge-small.
    """
    from app.wizard import run_wizard

    run_wizard(
        console=console,
        data_path=data,
        session_dir=session_dir,
        auto_start=auto_start,
    )


# ── forge stretch ─────────────────────────────────────────────────────────────


@app.command()
def stretch(
    model: str | None = typer.Option(
        None,
        "--model",
        help="Percorso a una directory checkpoint compatibile ForgeAI.",
    ),
    target_context: str | None = typer.Option(
        None,
        "--target-context",
        help=(
            "Target context in token (es. 65536 o 64k). "
            "Deve essere strettamente maggiore del contesto nativo."
        ),
    ),
    aggressiveness: str | None = typer.Option(
        None,
        "--aggressiveness",
        help="Profilo: prudent | balanced | ambitious (stesso metodo YaRN, cambia solo aggressività).",
    ),
    session_dir: str = typer.Option(
        "./.forge/stretch",
        "--session-dir",
        help="Directory per stato sessione stretch e report.",
    ),
    output_dir: str = typer.Option(
        "./models/stretched",
        "--output-dir",
        help="Directory dove salvare le varianti persistenti stretched.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Esegue senza prompt quando gli input forniti sono sufficienti.",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        help="Percorso a config JSON locale per uso non interattivo/batch.",
    ),
):
    """
    Estende in modo persistente un modello compatibile a un contesto più lungo con YaRN (v1).
    Nota: v1 produce una variante `adapter_plus_manifest` (mapping artifact + manifest, non `full_checkpoint`).
    """
    from app.stretch import run_stretch

    run_stretch(
        console=console,
        model_path=model,
        target_context=target_context,
        aggressiveness=aggressiveness,
        session_dir=session_dir,
        output_dir=output_dir,
        non_interactive=non_interactive,
        config_path=config,
    )


# ── forge train ───────────────────────────────────────────────────────────────


def _compute_train_val_samples(
    total_samples: int,
    *,
    batch_size: int,
    val_split: float,
) -> tuple[int, int]:
    if total_samples <= 0:
        raise ValueError(
            "Dataset privo di sequenze utili per il training. "
            "Riduci --context-length o prepara più dati."
        )
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split deve essere strettamente tra 0 e 1.")

    val_samples = max(1, int(total_samples * val_split))
    train_samples = total_samples - val_samples

    if train_samples <= 0:
        raise ValueError(
            "Split non valido: train set vuoto. Riduci --val-split o usa più dati."
        )
    if train_samples < batch_size:
        raise ValueError(
            "Train loader vuoto con i parametri correnti: "
            f"{train_samples} sample train < --batch-size {batch_size} (drop_last=True). "
            "Riduci --batch-size o usa più dati."
        )
    if val_samples < batch_size:
        raise ValueError(
            "Validation loader vuoto con i parametri correnti: "
            f"{val_samples} sample val < --batch-size {batch_size} (drop_last=True). "
            "Riduci --batch-size, riduci --val-split o usa più dati."
        )
    return train_samples, val_samples


def _validate_training_data_preflight(
    *,
    total_tokens: int,
    total_samples: int,
    context_length: int,
    batch_size: int,
    val_split: float,
) -> tuple[int, int]:
    if context_length <= 1:
        raise ValueError("--context-length deve essere > 1.")
    if total_tokens <= context_length:
        raise ValueError(
            "Dataset troppo piccolo per il context_length scelto: "
            f"token totali={total_tokens}, context_length={context_length}. "
            "Riduci --context-length o usa più dati."
        )
    return _compute_train_val_samples(
        total_samples,
        batch_size=batch_size,
        val_split=val_split,
    )


@app.command()
def train(
    arch: str = typer.Option("transformer", "--arch", help="Model architecture"),
    params: str = typer.Option(None, "--params", help="Parameter count (e.g. 400M) — or use --preset"),
    preset: str = typer.Option(None, "--preset", help="Preset name (e.g. forge-nano, forge-tiny)"),
    data: str = typer.Option(..., "--data", help="Path to processed data directory (with shard_*.bin)"),
    tokenizer_path: str = typer.Option(..., "--tokenizer", help="Path to trained tokenizer directory"),
    output: str = typer.Option("./checkpoints/run", "--output", help="Directory to save checkpoints"),
    resume: str | None = typer.Option(None, "--resume", help="Resume from checkpoint path"),
    lr: float = typer.Option(3e-4, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(4, "--batch-size", help="Micro batch size"),
    grad_accum: int = typer.Option(1, "--grad-accum", help="Gradient accumulation steps"),
    max_steps: int = typer.Option(10000, "--max-steps", help="Maximum training steps"),
    val_split: float = typer.Option(0.05, "--val-split", help="Fraction of data for validation"),
    save_every: int = typer.Option(1000, "--save-every", help="Save checkpoint every N steps"),
    val_every: int = typer.Option(200, "--val-every", help="Run validation every N steps"),
    context_length: int = typer.Option(None, "--context-length", help="Override context length"),
    gradient_checkpointing: bool = typer.Option(False, "--gradient-checkpointing", help="Enable gradient checkpointing"),
):
    """
    Train a language model from scratch using the current PyTorch training path (CUDA/MPS/CPU).
    Native MLX training is planned but not yet implemented.
    """
    import torch
    from app.core.backend import BackendType, get_backend
    from app.architectures import get_architecture
    from app.tokenizer import load_tokenizer
    from app.data import ShardedTokenDataset
    from app.training.trainer import train as run_training, TrainConfig

    backend = get_backend()
    console.print("\n[bold]ForgeAI — Starting training run[/bold]")
    console.print(f"  Backend : {backend.type.value.upper()} — {backend.device_name}")
    console.print(f"  Dtype   : {backend.recommended_dtype}")
    if backend.type == BackendType.MLX:
        console.print(
            "[yellow]MLX rilevato, ma il training nativo MLX non è ancora disponibile.[/yellow]"
        )
        console.print("[yellow]Questo run userà il percorso PyTorch su CPU (molto lento).[/yellow]")
    elif backend.type == BackendType.MPS and backend.mlx_available:
        console.print(
            "[dim]Nota: MLX è disponibile per alcuni flussi di inferenza (`forge chat --engine mlx`), "
            "ma il training usa PyTorch su MPS in questa versione.[/dim]"
        )

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    console.print(f"  Tokenizer : {tokenizer_path} (vocab={vocab_size})")

    # Build model
    config_overrides = {"vocab_size": vocab_size}
    if context_length is not None:
        config_overrides["context_length"] = context_length

    if preset:
        model = get_architecture(arch, preset=preset, **config_overrides)
    elif params:
        # Map param count to a reasonable preset or custom config
        from app.training.planner import _parse_params
        n_params = _parse_params(params)
        if n_params <= 80_000_000:
            model = get_architecture(arch, preset="forge-nano", **config_overrides)
        elif n_params <= 200_000_000:
            model = get_architecture(arch, preset="forge-tiny", **config_overrides)
        else:
            model = get_architecture(arch, preset="forge-small", **config_overrides)
    else:
        console.print("[red]Specify --params or --preset[/red]")
        raise typer.Exit(1)

    model_config = {
        "vocab_size": model.config.vocab_size,
        "context_length": model.config.context_length,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_embd": model.config.n_embd,
    }
    ctx_len = model.config.context_length

    total_params = model.num_parameters()
    console.print(f"  Model   : {arch} — {total_params / 1e6:.1f}M parameters")
    console.print(f"  Context : {ctx_len}")

    # Create data loaders
    try:
        dataset = ShardedTokenDataset(data, context_length=ctx_len)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    total_tokens = int(getattr(dataset, "total_tokens", 0))
    total_samples = len(dataset)
    try:
        train_samples, val_samples = _validate_training_data_preflight(
            total_tokens=total_tokens,
            total_samples=total_samples,
            context_length=ctx_len,
            batch_size=batch_size,
            val_split=val_split,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_samples, val_samples]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True
    )

    console.print(f"  Data    : {total_samples} sequences ({train_samples} train, {val_samples} val)")
    console.print()

    # Training config
    train_config = TrainConfig(
        learning_rate=lr,
        batch_size=batch_size,
        grad_accumulation_steps=grad_accum,
        max_steps=max_steps,
        dtype=backend.recommended_dtype,
        checkpoint_dir=output,
        save_every_steps=save_every,
        val_every_steps=val_every,
        architecture=arch,
        backend=backend.torch_device,
        gradient_checkpointing=gradient_checkpointing,
        log_file=str(Path(output) / "train_log.jsonl"),
    )

    # Run training
    result = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_config=model_config,
        resume_from=resume,
    )

    console.print()
    console.print("[bold green]Training complete![/bold green]")
    console.print(f"  Final step : {result['final_step']}")
    console.print(f"  Final loss : {result['final_loss']:.4f}")
    if result.get("best_val_loss") is not None:
        console.print(f"  Best val loss : {result['best_val_loss']:.4f}")
    console.print(f"  Checkpoints : {output}")


# ── forge eval ────────────────────────────────────────────────────────────────


@app.command("eval")
def evaluate(
    checkpoint: str = typer.Argument(..., help="Path to checkpoint directory"),
    benchmark: list[str] = typer.Option(
        ["perplexity"],
        "--benchmark", "-b",
        help=(
            "Eval helpers: perplexity (real local metric), "
            "tinystories (proxy qualitative check), "
            "hellaswag-mini (requires local JSONL file)."
        ),
    ),
    tokenizer_path: str = typer.Option(None, "--tokenizer", help="Path to tokenizer directory"),
    data: str = typer.Option(None, "--data", help="Path to eval data (for perplexity)"),
    hellaswag_data: str = typer.Option(None, "--hellaswag-data", help="Path to hellaswag_val.jsonl"),
    batch_size: int = typer.Option(4, "--batch-size"),
    max_batches: int = typer.Option(50, "--max-batches", help="Max batches for perplexity eval"),
):
    """
    Evaluate a checkpoint with local metrics/checks.
    Perplexity is the primary quantitative metric in this workflow.
    """
    import torch
    from app.core.backend import get_backend
    from app.architectures import get_architecture
    from app.tokenizer import load_tokenizer

    backend = get_backend()

    console.print("\n[bold]ForgeAI — Evaluation[/bold]")
    console.print(f"  Checkpoint : {checkpoint}")
    console.print(f"  Benchmarks : {', '.join(benchmark)}")

    # Load checkpoint metadata to reconstruct model
    ckpt_path = Path(checkpoint)
    if (ckpt_path / "latest").exists():
        resolved = (ckpt_path / "latest").resolve()
    else:
        resolved = ckpt_path

    meta_path = resolved / "metadata.json"
    if not meta_path.exists():
        console.print(f"[red]No metadata.json found in {resolved}[/red]")
        raise typer.Exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    model_config = meta.get("model_config", {})
    arch = meta.get("architecture", "transformer")

    # Build model from config
    model = get_architecture(arch, **model_config)

    # Load weights
    from app.checkpoints.manager import load_checkpoint as _load_ckpt
    _load_ckpt(checkpoint, model, device=backend.torch_device)

    device = backend.torch_device
    dtype_str = meta.get("dtype", backend.recommended_dtype)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str, torch.float32)

    results = {}

    # Perplexity
    if "perplexity" in benchmark:
        if not data:
            console.print(
                "\n  [yellow]Perplexity requested but --data is missing.[/yellow] "
                "Pass a prepared dataset directory."
            )
        else:
            from app.evaluation import compute_perplexity
            from app.data import create_dataloader

            ctx_len = model_config.get("context_length", 2048)
            eval_loader = create_dataloader(data, context_length=ctx_len, batch_size=batch_size, shuffle=False)

            ppl_result = compute_perplexity(model, eval_loader, device=device, max_batches=max_batches, dtype=dtype)
            results["perplexity"] = ppl_result

            console.print("\n  [bold]Perplexity (primary local metric)[/bold]")
            console.print(f"    Loss       : {ppl_result['loss']:.4f}")
            console.print(f"    Perplexity : {ppl_result['perplexity']:.2f}")
            console.print(f"    Tokens     : {ppl_result['tokens_evaluated']}")

    # TinyStories
    if "tinystories" in benchmark:
        if not tokenizer_path:
            console.print("[red]--tokenizer required for tinystories proxy check[/red]")
        else:
            from app.evaluation import eval_tinystories
            tokenizer = load_tokenizer(tokenizer_path)

            ts_result = eval_tinystories(model, tokenizer, device=device, dtype=dtype)
            results["tinystories"] = ts_result

            console.print("\n  [bold]TinyStories (proxy qualitative check)[/bold]")
            console.print(f"    Proxy score: {ts_result['avg_coherence']:.4f}")
            console.print(f"    Samples    : {ts_result['num_samples']}")
            console.print("    [dim]Nota: non è un benchmark TinyStories ufficiale.[/dim]")
            if ts_result.get("samples"):
                console.print("\n    Sample generation:")
                sample = ts_result["samples"][0]
                console.print(f"    [dim]{sample['generation'][:200]}[/dim]")

    # HellaSwag
    if "hellaswag-mini" in benchmark:
        if not tokenizer_path:
            console.print("[red]--tokenizer required for hellaswag-mini local check[/red]")
        else:
            from app.evaluation import eval_hellaswag_mini
            tokenizer = load_tokenizer(tokenizer_path)

            hs_result = eval_hellaswag_mini(
                model, tokenizer, data_path=hellaswag_data, device=device, dtype=dtype
            )
            results["hellaswag-mini"] = hs_result

            console.print("\n  [bold]HellaSwag-mini (local-file dependent)[/bold]")
            if hs_result.get("accuracy") is not None:
                console.print(f"    Accuracy   : {hs_result['accuracy']:.4f} ({hs_result['correct']}/{hs_result['total']})")
            else:
                console.print(f"    [yellow]{hs_result.get('note', 'No data available')}[/yellow]")

    # Save results
    results_path = resolved / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n  Results saved to {results_path}")


# ── forge eval compare ────────────────────────────────────────────────────────


@app.command("eval-compare")
def eval_compare(
    checkpoint_a: str = typer.Argument(..., help="Path to first checkpoint"),
    checkpoint_b: str = typer.Argument(..., help="Path to second checkpoint"),
):
    """
    Compare evaluation results from two checkpoints side by side.
    """
    from app.evaluation import compare_checkpoints

    # Load eval results
    def _load_results(ckpt_path: str) -> dict:
        path = Path(ckpt_path)
        if (path / "latest").exists():
            path = (path / "latest").resolve()
        results_file = path / "eval_results.json"
        if not results_file.exists():
            console.print(f"[red]No eval_results.json in {path}. Run 'forge eval' first.[/red]")
            raise typer.Exit(1)
        with open(results_file) as f:
            return json.load(f)

    results_a = _load_results(checkpoint_a)
    results_b = _load_results(checkpoint_b)

    # Flatten nested results for comparison
    flat_a = {}
    flat_b = {}
    for key, val in results_a.items():
        if isinstance(val, dict):
            flat_a.update(val)
        else:
            flat_a[key] = val
    for key, val in results_b.items():
        if isinstance(val, dict):
            flat_b.update(val)
        else:
            flat_b[key] = val

    markdown = compare_checkpoints(flat_a, flat_b)
    console.print(markdown)

    # Also save to file
    output_path = Path("comparison.md")
    output_path.write_text(markdown)
    console.print(f"\nSaved to {output_path}")


# ── forge chat ────────────────────────────────────────────────────────────────


def _check_mlx_lm() -> bool:
    """Return True if mlx_lm is importable."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def _check_llama_cpp() -> bool:
    """Return True if llama_cpp (Python binding) is importable."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _find_llama_cli() -> str | None:
    """Look for the llama-cli (llama.cpp) executable in PATH."""
    import shutil
    for name in ("llama-cli", "llama", "main"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _resolve_checkpoint(model_path: str) -> tuple[Path, dict]:
    """Resolve a checkpoint path and return (resolved_dir, metadata_dict)."""
    ckpt_path = Path(model_path)
    resolved = ckpt_path
    if (ckpt_path / "latest").exists():
        resolved = (ckpt_path / "latest").resolve()

    meta_path = resolved / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {resolved}")

    with open(meta_path) as f:
        meta = json.load(f)
    return resolved, meta


def _chat_mlx_lm(
    model_path: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> None:
    """
    Chat loop using mlx_lm (Apple Silicon).

    Uses mlx_lm.load() for the model and mlx_lm.generate() with streaming
    for efficient token-by-token output on Apple Silicon.
    """
    import mlx_lm

    console.print("  Engine  : [blue]mlx-lm[/blue] (Apple Silicon native)")

    # mlx_lm.load can load HF format or mlx-converted directories
    model, tokenizer = mlx_lm.load(model_path)

    console.print()
    console.print("[dim]Type your message and press Enter. Type 'quit' to exit.[/dim]")
    console.print()

    conversation: list[str] = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("Bye!")
            break

        if not user_input.strip():
            continue

        conversation.append(user_input)
        prompt = "\n".join(conversation) + "\n"

        console.print("[bold green]Model:[/bold green] ", end="")

        response_tokens: list[str] = []
        
        # Robust handling for different mlx-lm versions
        generator = None
        if hasattr(mlx_lm, "stream_generate"):
            generator = mlx_lm.stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            )
        else:
            # Fallback for newer versions that might merge streaming into generate
            generator = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                stream=True,
            )

        for chunk in generator:
            # Some versions yield strings, others yield objects with a .text attribute
            token_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            print(token_text, end="", flush=True)
            response_tokens.append(token_text)

        full_response = "".join(response_tokens).strip()
        conversation.append(full_response)
        console.print()
        console.print()


def _chat_llama_cpp(
    model_path: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> None:
    """
    Chat loop using llama-cpp-python binding.

    Loads a GGUF model and runs streaming inference via the Python API.
    """
    from llama_cpp import Llama

    console.print("  Engine  : [green]llama.cpp[/green] (Python binding)")

    # Find GGUF file
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        console.print(f"[red]No .gguf file found in {model_path}[/red]")
        console.print("[dim]Convert with: llama-quantize or use a GGUF model.[/dim]")
        raise typer.Exit(1)

    gguf_path = str(gguf_files[0])
    console.print(f"  GGUF    : {gguf_path}")

    llm = Llama(model_path=gguf_path, n_ctx=2048, verbose=False)

    console.print()
    console.print("[dim]Type your message and press Enter. Type 'quit' to exit.[/dim]")
    console.print()

    conversation: list[str] = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("Bye!")
            break

        if not user_input.strip():
            continue

        conversation.append(user_input)
        prompt = "\n".join(conversation) + "\n"

        console.print("[bold green]Model:[/bold green] ", end="")

        response_text = ""
        for chunk in llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stream=True,
        ):
            token_text = chunk["choices"][0].get("text", "")
            print(token_text, end="", flush=True)
            response_text += token_text

        conversation.append(response_text.strip())
        console.print()
        console.print()


def _chat_llama_cli(
    model_path: str,
    llama_cli_path: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> None:
    """
    Chat loop using the llama-cli executable (llama.cpp CLI).

    Falls back to subprocess-based generation with llama-cli --interactive.
    """
    import subprocess as sp

    console.print(f"  Engine  : [green]llama-cli[/green] ({llama_cli_path})")

    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        console.print(f"[red]No .gguf file found in {model_path}[/red]")
        raise typer.Exit(1)

    gguf_path = str(gguf_files[0])

    console.print(f"  GGUF    : {gguf_path}")
    console.print()
    console.print("[dim]Launching llama-cli interactive mode...[/dim]")
    console.print()

    try:
        sp.run(
            [
                llama_cli_path,
                "-m", gguf_path,
                "--interactive",
                "--temp", str(temperature),
                "--top-k", str(top_k),
                "--top-p", str(top_p),
                "-n", str(max_tokens),
            ],
        )
    except FileNotFoundError:
        console.print(f"[red]Could not run {llama_cli_path}[/red]")
        raise typer.Exit(1)


def _chat_native_pytorch(
    model_path: str,
    tokenizer_path: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    top_p: float,
) -> None:
    """
    Chat loop using ForgeAI's native PyTorch generate().

    Last-resort fallback when neither mlx_lm nor llama.cpp are available.
    Maintains conversation memory by accumulating context tokens.
    """
    import torch
    from app.core.backend import get_backend
    from app.tokenizer import load_tokenizer
    from app.architectures import get_architecture
    from app.checkpoints.manager import load_checkpoint

    backend = get_backend()
    device = backend.torch_device

    console.print("  Engine  : [yellow]PyTorch native[/yellow] (fallback — slower than mlx-lm/llama.cpp)")

    tokenizer = load_tokenizer(tokenizer_path)

    resolved, meta = _resolve_checkpoint(model_path)
    model_config = meta.get("model_config", {})
    arch = meta.get("architecture", "transformer")
    model = get_architecture(arch, **model_config)
    load_checkpoint(model_path, model, device=device)
    model = model.to(device)
    model.eval()

    context_length = model_config.get("context_length", 2048)

    console.print(f"  Model   : {arch} ({model.num_parameters() / 1e6:.1f}M params)")
    console.print(f"  Loaded  : {resolved}")
    console.print()
    console.print("[dim]Type your message and press Enter. Type 'quit' to exit.[/dim]")
    console.print()

    conversation_ids: list[int] = []
    eos_id = tokenizer.token_to_id("<|eos|>")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("Bye!")
            break

        if not user_input.strip():
            continue

        # Append user turn to conversation context
        encoded = tokenizer.encode(user_input)
        conversation_ids.extend(encoded.ids)
        if eos_id is not None:
            conversation_ids.append(eos_id)

        # Truncate to fit context window (keep most recent tokens)
        if len(conversation_ids) > context_length - max_tokens:
            conversation_ids = conversation_ids[-(context_length - max_tokens):]

        input_ids = torch.tensor([conversation_ids], device=device)

        console.print("[bold green]Model:[/bold green] ", end="")

        # Stream tokens one by one
        generated_ids: list[int] = []
        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = input_ids if input_ids.size(1) <= context_length else input_ids[:, -context_length:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]

                # Temperature
                logits = logits / max(temperature, 1e-8)

                # Top-k
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < topk_vals[:, [-1]]] = float("-inf")

                # Top-p
                if top_p < 1.0:
                    import torch.nn.functional as F
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                # Stop on EOS
                if eos_id is not None and token_id == eos_id:
                    break

                generated_ids.append(token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stream: decode and print each token
                token_text = tokenizer.decode([token_id])
                print(token_text, end="", flush=True)

        # Add generated tokens to conversation memory
        conversation_ids.extend(generated_ids)

        console.print()
        console.print()


@app.command()
def chat(
    model_path: str = typer.Argument(..., help="Checkpoint path, HF model dir, or GGUF file dir"),
    tokenizer_path: str = typer.Option(None, "--tokenizer", help="Path to tokenizer directory (for native PyTorch)"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    max_tokens: int = typer.Option(200, "--max-tokens"),
    top_k: int = typer.Option(50, "--top-k"),
    top_p: float = typer.Option(0.9, "--top-p"),
    engine: str = typer.Option(
        "auto", "--engine", "-e",
        help="Inference engine: auto, mlx, llama-cpp, llama-cli, pytorch",
    ),
):
    """
    Chat with a model interactively.

    Inference is delegated to the best available engine:
      1. mlx-lm   (Apple Silicon — fastest on Mac)
      2. llama.cpp (CUDA / CPU — via Python binding or CLI)
      3. PyTorch   (fallback — ForgeAI's native generate())

    Use --engine to force a specific backend.
    """
    from app.core.backend import get_backend, BackendType

    backend = get_backend()

    console.print("\n[bold]ForgeAI — Chat[/bold]")
    console.print(f"  Backend : {backend.type.value.upper()} — {backend.device_name}")

    resolved_path = Path(model_path)
    if not resolved_path.exists():
        console.print(f"[red]Path not found: {model_path}[/red]")
        console.print("[dim]Use 'forge model pull <name>' to download a model first.[/dim]")
        raise typer.Exit(1)

    # ── Engine selection ─────────────────────────────────────────────────
    has_mlx = _check_mlx_lm()
    has_llama_cpp = _check_llama_cpp()
    llama_cli_path = _find_llama_cli()
    has_gguf = bool(list(resolved_path.glob("*.gguf")))
    has_forge_ckpt = (resolved_path / "metadata.json").exists() or (resolved_path / "latest").exists()

    if engine == "auto":
        # Pick the best engine for this hardware + model format
        if backend.type == BackendType.MLX and has_mlx:
            engine = "mlx"
        elif has_gguf and has_llama_cpp:
            engine = "llama-cpp"
        elif has_gguf and llama_cli_path:
            engine = "llama-cli"
        elif has_forge_ckpt:
            engine = "pytorch"
        elif has_mlx:
            # MLX can load HF dirs directly
            engine = "mlx"
        else:
            engine = "pytorch"

    console.print(f"  Model   : {model_path}")

    # ── Dispatch to engine ───────────────────────────────────────────────
    if engine == "mlx":
        if not has_mlx:
            console.print("[red]mlx-lm not installed.[/red] Install with: pip install mlx-lm")
            raise typer.Exit(1)
        _chat_mlx_lm(model_path, temperature, max_tokens, top_p)

    elif engine == "llama-cpp":
        if not has_llama_cpp:
            console.print("[red]llama-cpp-python not installed.[/red] Install with: pip install llama-cpp-python")
            raise typer.Exit(1)
        _chat_llama_cpp(model_path, temperature, max_tokens, top_p, top_k)

    elif engine == "llama-cli":
        if not llama_cli_path:
            console.print("[red]llama-cli not found in PATH.[/red]")
            console.print("[dim]Build llama.cpp and ensure the binary is in your PATH.[/dim]")
            raise typer.Exit(1)
        _chat_llama_cli(model_path, llama_cli_path, temperature, max_tokens, top_p, top_k)

    elif engine == "pytorch":
        if not has_forge_ckpt:
            console.print("[red]No ForgeAI checkpoint found at this path.[/red]")
            console.print("[dim]PyTorch engine requires a ForgeAI checkpoint (with metadata.json).[/dim]")
            raise typer.Exit(1)
        if not tokenizer_path:
            console.print("[red]--tokenizer required for PyTorch native engine[/red]")
            raise typer.Exit(1)
        _chat_native_pytorch(model_path, tokenizer_path, temperature, max_tokens, top_k, top_p)

    else:
        console.print(f"[red]Unknown engine '{engine}'.[/red] Options: auto, mlx, llama-cpp, llama-cli, pytorch")
        raise typer.Exit(1)


# ── forge model ───────────────────────────────────────────────────────────────


model_app = typer.Typer(help="Manage redistributed open-source models.")
app.add_typer(model_app, name="model")

AVAILABLE_MODELS = {
    "smollm-135m": {
        "source": "HuggingFace/HuggingFaceTB/SmolLM-135M",
        "hf_id": "HuggingFaceTB/SmolLM-135M",
        "params": "135M",
        "license": "Apache 2.0",
    },
    "qwen2.5-0.5b": {
        "source": "HuggingFace/Qwen/Qwen2.5-0.5B",
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "params": "494M",
        "license": "Apache 2.0",
    },
    "tinyllama-1b": {
        "source": "HuggingFace/TinyLlama/TinyLlama_v1.1",
        "hf_id": "TinyLlama/TinyLlama_v1.1",
        "params": "1.1B",
        "license": "Apache 2.0",
    },
}


@model_app.command("pull")
def model_pull(
    name: str = typer.Argument(..., help="Model name (run 'forge model list' to see options)"),
    output: str = typer.Option("./models", "--output", "-o"),
    skip_convert: bool = typer.Option(False, "--skip-convert", help="Skip conversion to ForgeAI format"),
):
    """
    Download a redistributed open-source model and convert to ForgeAI checkpoint format.
    """
    if name not in AVAILABLE_MODELS:
        console.print(f"[red]Unknown model '{name}'.[/red] Run 'forge model list' to see available models.")
        raise typer.Exit(1)

    info = AVAILABLE_MODELS[name]
    console.print(f"[bold]Pulling[/bold] {name}")
    console.print(f"  Source  : {info['source']}")
    console.print(f"  Params  : {info['params']}")
    console.print(f"  License : {info['license']}")
    console.print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[yellow]huggingface_hub not installed.[/yellow]")
        console.print("Install it with: pip install huggingface_hub")
        console.print(f"Or download manually from: https://huggingface.co/{info['hf_id']}")
        raise typer.Exit(1)

    hf_dir = Path(output) / f"{name}-hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    console.print("Downloading from HuggingFace Hub...")
    try:
        snapshot_download(
            repo_id=info["hf_id"],
            local_dir=str(hf_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Downloaded to {hf_dir}[/green]")

    if skip_convert:
        console.print()
        console.print("[dim]Skipping conversion. HuggingFace weights saved as-is.[/dim]")
        return

    # ── Convert HF weights to ForgeAI format ────────────────────────────
    console.print()
    console.print("[bold]Converting to ForgeAI format...[/bold]")

    forge_dir = Path(output) / name
    try:
        from app.checkpoints.converter import convert_hf_to_forge

        result_dir = convert_hf_to_forge(hf_dir, forge_dir)

        # Read back metadata for summary
        meta_path = result_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        mc = meta.get("model_config", {})
        conv = meta.get("conversion_info", {})

        console.print("[green]Conversion complete![/green]")
        console.print(f"  ForgeAI checkpoint : {result_dir}")
        console.print("  Architecture       : transformer")
        console.print(f"  Parameters         : {mc.get('n_layer', '?')}L / {mc.get('n_head', '?')}H / {mc.get('n_embd', '?')}D")
        console.print(f"  Vocab size         : {mc.get('vocab_size', '?')}")
        console.print(f"  Context length     : {mc.get('context_length', '?')}")
        console.print(f"  Keys converted     : {conv.get('total_forge_keys', '?')} / {conv.get('total_hf_keys', '?')}")
        if conv.get("skipped_keys"):
            console.print(f"  Skipped (recomputed): {len(conv['skipped_keys'])} keys (rotary embeddings, etc.)")
        console.print()
        console.print(f"  Use with: [bold]forge chat {result_dir} --tokenizer <tokenizer_path>[/bold]")
        console.print(f"  Or eval:  [bold]forge eval {result_dir} --tokenizer <tokenizer_path>[/bold]")

    except ImportError as e:
        console.print(f"[yellow]Conversion requires additional packages:[/yellow] {e}")
        console.print("Install with: pip install safetensors")
        console.print(f"[dim]HuggingFace weights are still available at {hf_dir}[/dim]")
    except ValueError as e:
        console.print(f"[red]Conversion failed:[/red] {e}")
        console.print(f"[dim]HuggingFace weights are still available at {hf_dir}[/dim]")
    except Exception as e:
        console.print(f"[red]Unexpected error during conversion:[/red] {e}")
        console.print(f"[dim]HuggingFace weights are still available at {hf_dir}[/dim]")


@model_app.command("list")
def model_list():
    """List available redistributed models."""
    from rich.table import Table
    table = Table(title="Available Models")
    table.add_column("Name")
    table.add_column("Params")
    table.add_column("License")
    table.add_column("Source")
    for name, info in AVAILABLE_MODELS.items():
        table.add_row(name, info["params"], info["license"], info["source"])
    console.print(table)


# ── forge tokenizer ───────────────────────────────────────────────────────────


tokenizer_app = typer.Typer(help="Train and manage tokenizers.")
app.add_typer(tokenizer_app, name="tokenizer")


@tokenizer_app.command("train")
def tokenizer_train(
    data: str = typer.Option(..., "--data", help="Path to text corpus (file or directory)"),
    vocab_size: int = typer.Option(8000, "--vocab-size"),
    output: str = typer.Option("./tokenizers/my-tokenizer", "--output", "-o"),
):
    """Train a BPE tokenizer on your corpus."""
    from app.tokenizer import train_bpe_tokenizer, save_tokenizer

    console.print("[bold]Training BPE tokenizer[/bold]")
    console.print(f"  Data       : {data}")
    console.print(f"  Vocab size : {vocab_size}")
    console.print(f"  Output     : {output}")
    console.print()

    tokenizer = train_bpe_tokenizer(
        data_path=data,
        vocab_size=vocab_size,
    )

    save_tokenizer(tokenizer, output)

    console.print(f"[green]Tokenizer trained and saved to {output}[/green]")
    console.print(f"  Vocab size : {tokenizer.get_vocab_size()}")

    # Quick test
    test_text = "Hello, world! This is ForgeAI."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    console.print(f"\n  Test encode: \"{test_text}\"")
    console.print(f"  Tokens     : {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
    console.print(f"  Decoded    : \"{decoded}\"")


# ── forge data ────────────────────────────────────────────────────────────────


data_app = typer.Typer(help="Prepare and inspect training data.")
app.add_typer(data_app, name="data")


@data_app.command("prepare")
def data_prepare(
    source: str = typer.Argument(..., help="Path to raw text corpus (directory or file)"),
    output: str = typer.Option(..., "--output", "-o"),
    tokenizer_dir: str = typer.Option(..., "--tokenizer", help="Path to trained tokenizer directory"),
    context_length: int = typer.Option(2048, "--context-length"),
):
    """Tokenize and shard a corpus into training-ready binary format."""
    from app.tokenizer import load_tokenizer
    from app.data import prepare_dataset

    console.print("[bold]Preparing data[/bold]")
    console.print(f"  Source         : {source}")
    console.print(f"  Output         : {output}")
    console.print(f"  Tokenizer      : {tokenizer_dir}")
    console.print(f"  Context length : {context_length}")
    console.print()

    tokenizer = load_tokenizer(tokenizer_dir)

    try:
        metadata = prepare_dataset(
            data_path=source,
            tokenizer=tokenizer,
            output_dir=output,
            context_length=context_length,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print("[green]Data prepared![/green]")
    console.print(f"  Total tokens : {metadata['total_tokens']:,}")
    console.print(f"  Shards       : {metadata['num_shards']}")
    console.print(f"  Vocab size   : {metadata['vocab_size']}")
    console.print(f"  Token dtype  : {metadata.get('token_dtype', 'uint16')}")
    console.print(f"  Output       : {output}")


if __name__ == "__main__":
    app()
