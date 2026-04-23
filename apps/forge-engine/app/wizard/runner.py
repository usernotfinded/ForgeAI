from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from app.core.backend import BackendInfo, get_backend
from app.data import prepare_dataset
from app.tokenizer import load_tokenizer, save_tokenizer, train_bpe_tokenizer
from app.training.planner import estimate_training

from .analysis import DatasetAnalysis, analyze_dataset
from .catalog import BaseModelCandidate, select_compatible_models
from .recommendation import (
    OBJECTIVE_LABELS,
    ObjectiveCategory,
    RouteRecommendation,
    StrategyPreset,
    WizardPath,
    build_strategy_config,
    recommend_route,
    stratified_consent,
)
from .session import SessionStore, WizardSessionState


def run_wizard(
    console: Console,
    data_path: str | None = None,
    session_dir: str = "./.forge/wizard",
    auto_start: bool = False,
) -> None:
    store = SessionStore(session_dir)
    state = _load_or_initialize_session(console, store)
    if state is None:
        return

    if state.current_step <= 1 and not _step_1_objective(console, state, store):
        return
    if state.current_step <= 2 and not _step_2_data(console, state, store, data_path):
        return
    if state.current_step <= 3 and not _step_3_hardware(console, state, store):
        return
    if state.current_step <= 4 and not _step_4_route_selection(console, state, store):
        return
    if state.current_step <= 5 and not _step_5_strategy(console, state, store):
        return
    if state.current_step <= 6 and not _step_6_review_and_consent(console, state, store):
        return

    _step_7_execution(console, state, store, auto_start=auto_start)


def _load_or_initialize_session(console: Console, store: SessionStore) -> WizardSessionState | None:
    existing_state = store.load()
    if existing_state is None:
        state = WizardSessionState(session_id=_new_session_id())
        store.save(state)
        return state

    options = ["Riprendi", "Ricomincia", "Visualizza ultimo riepilogo"]
    choice = _prompt_choice(
        "È stata trovata una sessione precedente. Cosa vuoi fare?",
        options,
        default=1,
    )

    if choice == 3:
        summary = store.read_summary()
        if summary:
            console.print("\n[bold]Ultimo riepilogo salvato[/bold]\n")
            console.print(summary)
        else:
            console.print("[yellow]Nessun riepilogo finale trovato per questa sessione.[/yellow]")
        return None

    if choice == 2:
        store.reset()
        state = WizardSessionState(session_id=_new_session_id())
        store.save(state)
        return state

    return existing_state


def _step_1_objective(console: Console, state: WizardSessionState, store: SessionStore) -> bool:
    console.rule("[bold]Passo 1/7 — Obiettivo[/bold]")
    console.print("Perché: definire il risultato atteso aiuta a scegliere un piano realistico.")

    objectives = [
        ObjectiveCategory.PERSONAL_ASSISTANT,
        ObjectiveCategory.DOC_ASSISTANT,
        ObjectiveCategory.CODE_ASSISTANT,
        ObjectiveCategory.CLASSIFICATION,
        ObjectiveCategory.NEW_LANGUAGE,
        ObjectiveCategory.RESEARCH,
        ObjectiveCategory.OTHER,
    ]

    labels = [OBJECTIVE_LABELS[item] for item in objectives]
    selection = _prompt_choice("Scegli il tuo obiettivo principale:", labels, default=1)
    objective = objectives[selection - 1]

    expectation = typer.prompt(
        "In una frase, cosa ti aspetti dal modello?",
        default="",
        show_default=False,
    ).strip()

    warnings = _assess_expectation(expectation)
    for warning in warnings:
        console.print(f"[yellow]Nota aspettative:[/yellow] {warning}")

    state.answers["objective"] = objective.value
    state.answers["objective_label"] = OBJECTIVE_LABELS[objective]
    state.answers["expectation"] = expectation
    state.answers["expectation_notes"] = warnings
    state.mark_step_completed(1)
    store.save(state)
    return True


def _step_2_data(
    console: Console,
    state: WizardSessionState,
    store: SessionStore,
    data_path_arg: str | None,
) -> bool:
    console.rule("[bold]Passo 2/7 — Dati[/bold]")
    console.print("Perché: quantità e formato dei dati cambiano completamente il percorso consigliato.")

    data_path = data_path_arg or str(state.answers.get("data_path", "")).strip()

    while not data_path:
        data_path = typer.prompt("Percorso dataset (file o cartella)").strip()

    while True:
        try:
            analysis = analyze_dataset(data_path)
            break
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            data_path = typer.prompt("Inserisci un percorso valido").strip()
        except Exception as exc:  # pragma: no cover - defensive branch
            console.print(f"[red]Analisi dataset fallita:[/red] {exc}")
            return False

    state.answers["data_path"] = str(Path(data_path).resolve())
    state.data_analysis = analysis.to_dict()
    state.mark_step_completed(2)
    store.save(state)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Campo", style="dim")
    table.add_column("Valore")
    table.add_row("Documenti analizzati", f"{analysis.documents_scanned}")
    table.add_row("Token stimati", f"{analysis.estimated_tokens:,}")
    table.add_row("Tipo dataset", analysis.dataset_type)
    table.add_row("Lingua dominante", analysis.dominant_language)
    table.add_row("Duplicati", f"{analysis.duplicate_ratio * 100:.1f}%")
    table.add_row("Qualità", f"{analysis.quality_score:.1f}/100")
    console.print(table)

    for note in analysis.quality_notes:
        console.print(f"  - {note}")

    return True


def _step_3_hardware(console: Console, state: WizardSessionState, store: SessionStore) -> bool:
    console.rule("[bold]Passo 3/7 — Hardware[/bold]")
    console.print("Perché: stimiamo tempi/costi reali prima di scegliere il percorso.")

    backend = get_backend()
    data_path = str(state.answers.get("data_path", ""))

    scenarios = [
        ("Scenario rapido", "50M"),
        ("Scenario bilanciato", "120M"),
        ("Scenario ambizioso", "310M"),
    ]

    scenario_plans: list[dict[str, Any]] = []
    table = Table(title="Piano realistico (stima)")
    table.add_column("Scenario")
    table.add_column("Parametri")
    table.add_column("Tempo")
    table.add_column("Costo energia")

    for label, params in scenarios:
        plan = estimate_training(
            arch="transformer",
            params=params,
            data_path=data_path,
            backend=backend,
        )
        scenario_plans.append(
            {
                "label": label,
                "params": params,
                "estimated_hours": round(plan.estimated_hours, 2),
                "estimated_days": round(plan.estimated_hours / 24, 2),
                "electricity_cost": round(plan.electricity_kwh * plan.kwh_cost, 2),
            }
        )
        table.add_row(
            label,
            params,
            f"~{plan.estimated_hours:.0f}h",
            f"~€{plan.electricity_kwh * plan.kwh_cost:.0f}",
        )

    console.print(f"Backend rilevato: [bold]{backend.type.value.upper()}[/bold] — {backend.device_name}")
    if backend.vram_gb is not None:
        memory_label = "RAM unificata" if backend.unified_memory else "VRAM"
        console.print(f"{memory_label}: {backend.vram_gb:.1f} GB")
    console.print(table)

    state.hardware = {
        "backend": backend.type.value,
        "device_name": backend.device_name,
        "vram_gb": backend.vram_gb,
        "unified_memory": backend.unified_memory,
        "recommended_dtype": backend.recommended_dtype,
        "recommended_batch_size": _recommended_batch_from_backend(backend),
        "scenario_plans": scenario_plans,
    }
    state.mark_step_completed(3)
    store.save(state)
    return True


def _step_4_route_selection(console: Console, state: WizardSessionState, store: SessionStore) -> bool:
    console.rule("[bold]Passo 4/7 — Scelta del Percorso[/bold]")
    console.print("Perché: il wizard propone il percorso più realistico in base a dati + hardware + obiettivo.")

    analysis = DatasetAnalysis.from_dict(state.data_analysis or {})
    objective = _objective_from_state(state)
    backend = get_backend()

    recommendation = recommend_route(
        objective=objective,
        analysis=analysis,
        available_memory_gb=backend.vram_gb,
        unified_memory=backend.unified_memory,
    )

    state.recommendation = recommendation.to_dict()

    recommended_label = _path_label(recommendation.recommended_path)
    alternate_path = (
        WizardPath.TRAIN_FROM_SCRATCH
        if recommendation.recommended_path == WizardPath.ADAPT_EXISTING
        else WizardPath.ADAPT_EXISTING
    )

    console.print(f"Consiglio automatico: [bold]{recommended_label}[/bold]")
    for reason in recommendation.reasons:
        console.print(f"  - {reason}")

    if recommendation.unsupported_adaptation_modes:
        unsupported = ", ".join(recommendation.unsupported_adaptation_modes)
        console.print(
            "[yellow]Nota modalità:[/yellow] "
            f"{unsupported} non sono ancora disponibili in v1. "
            "Alternativa supportata: adattamento completo del modello base (continued pretraining)."
        )

    for warning in recommendation.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    choice = _prompt_choice(
        "Quale percorso vuoi usare?",
        [
            f"Segui consiglio: {recommended_label}",
            f"Percorso alternativo: {_path_label(alternate_path)}",
        ],
        default=1,
    )

    selected_path = recommendation.recommended_path if choice == 1 else alternate_path
    state.selected_path = selected_path.value

    if selected_path == WizardPath.ADAPT_EXISTING:
        compatible_models = select_compatible_models(backend.vram_gb, limit=5)
        if not compatible_models:
            console.print(
                "[red]Nessun modello base compatibile trovato con l'hardware rilevato.[/red]"
            )
            if typer.confirm(
                "Vuoi passare al percorso 'Addestra da zero' con preset piccolo?",
                default=True,
            ):
                state.selected_path = WizardPath.TRAIN_FROM_SCRATCH.value
                state.selected_base_model = None
            else:
                state.execution = {
                    "status": "blocked",
                    "reason": "no_compatible_base_model",
                    "next_action": "upgrade_hardware_or_choose_train_from_scratch",
                }
                store.save(state)
                return False
        else:
            _print_base_model_table(console, compatible_models)
            model_labels = [
                f"{model.name} ({model.params}, raccomandazione: {model.recommendation_level})"
                for model in compatible_models
            ]
            model_choice = _prompt_choice(
                "Seleziona il modello base da adattare:",
                model_labels,
                default=1,
            )
            selected_model = compatible_models[model_choice - 1]
            state.selected_base_model = selected_model.name

    state.mark_step_completed(4)
    store.save(state)
    return True


def _step_5_strategy(console: Console, state: WizardSessionState, store: SessionStore) -> bool:
    console.rule("[bold]Passo 5/7 — Strategia[/bold]")
    console.print("Perché: scegli il compromesso tra velocità, costo e qualità finale.")

    selected_path = _path_from_state(state)
    backend = get_backend()
    batch_hint = _recommended_batch_from_backend(backend)

    presets = [StrategyPreset.FAST, StrategyPreset.BALANCED, StrategyPreset.MAX_QUALITY]
    labels = ["Veloce", "Bilanciato", "Massima Qualità"]
    choice = _prompt_choice("Scegli il preset:", labels, default=2)
    selected_preset = presets[choice - 1]

    strategy = build_strategy_config(selected_path, selected_preset, batch_hint)
    console.print(f"Preset scelto: [bold]{strategy['label']}[/bold]")
    console.print(f"Trade-off: {strategy['tradeoff']}")

    state.preset = selected_preset.value
    state.answers["strategy"] = strategy
    state.mark_step_completed(5)
    store.save(state)
    return True


def _step_6_review_and_consent(console: Console, state: WizardSessionState, store: SessionStore) -> bool:
    console.rule("[bold]Passo 6/7 — Review Finale + Conferma[/bold]")
    console.print("Perché: conferma consapevole prima di generare ed eseguire il piano.")

    analysis = DatasetAnalysis.from_dict(state.data_analysis or {})
    recommendation = RouteRecommendation.from_dict(state.recommendation or {})
    selected_path = _path_from_state(state)
    backend = get_backend()

    strategy = state.answers.get("strategy", {})
    objective_label = str(state.answers.get("objective_label", "Obiettivo non specificato"))

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Voce", style="dim")
    summary_table.add_column("Valore")
    summary_table.add_row("Obiettivo", objective_label)
    summary_table.add_row("Percorso consigliato", _path_label(recommendation.recommended_path))
    summary_table.add_row("Percorso scelto", _path_label(selected_path))
    summary_table.add_row("Tipo dataset", analysis.dataset_type)
    summary_table.add_row("Token stimati", f"{analysis.estimated_tokens:,}")
    summary_table.add_row("Preset", str(strategy.get("label", "n/d")))
    if state.selected_base_model:
        summary_table.add_row("Modello base", state.selected_base_model)
    console.print(summary_table)

    consent_level, consent_messages = stratified_consent(
        chosen_path=selected_path,
        recommendation=recommendation,
        analysis=analysis,
        available_memory_gb=backend.vram_gb,
    )

    console.print("\n[bold]Stratified consent[/bold]")
    console.print("Livello 1 — Consiglio: il wizard suggerisce la scelta con minor rischio.")

    if consent_level >= 2:
        console.print("Livello 2 — Warning forte:")
        for message in consent_messages:
            console.print(f"  - {message}")

    if consent_level >= 3:
        phrase = typer.prompt(
            "Livello 3 — Override consapevole: scrivi CONFERMO per continuare",
            default="",
            show_default=False,
        ).strip()
        if phrase.upper() != "CONFERMO":
            console.print("[yellow]Operazione annullata: override non confermato.[/yellow]")
            state.execution = {
                "status": "stopped",
                "reason": "override_not_confirmed",
                "next_action": "repeat_step_6",
            }
            store.save(state)
            return False

    proceed = typer.confirm("Confermi il piano finale?", default=True)
    if not proceed:
        console.print("[yellow]Wizard fermato prima dell'esecuzione.[/yellow]")
        state.execution = {
            "status": "stopped",
            "reason": "user_cancelled_at_step_6",
            "next_action": "repeat_step_6",
        }
        store.save(state)
        return False

    state.mark_step_completed(6)
    store.save(state)
    return True


def _step_7_execution(
    console: Console,
    state: WizardSessionState,
    store: SessionStore,
    auto_start: bool,
) -> None:
    console.rule("[bold]Passo 7/7 — Esecuzione Guidata[/bold]")
    console.print("Perché: generiamo artefatti tracciabili ed eseguiamo i passi locali in sicurezza.")

    config = _build_generated_config(state, store)
    store.ensure_dirs()
    _write_json(store.generated_config_path, config)
    _write_execution_plan(store.execution_plan_path, config["commands"])

    summary_text = _build_final_summary(state, config, store)
    store.summary_path.write_text(summary_text, encoding="utf-8")

    state.generated_config = config
    artifact_paths = {
        str(store.state_path.resolve()),
        str(store.generated_config_path.resolve()),
        str(store.execution_plan_path.resolve()),
        str(store.summary_path.resolve()),
    }
    state.artifacts = sorted(artifact_paths)
    state.mark_step_completed(7)
    state.execution = {
        "status": "ready",
        "next_action": "run_local_steps_or_training",
    }
    store.save(state)

    console.print("Artefatti generati:")
    console.print(f"  - Stato sessione: {store.state_path}")
    console.print(f"  - Config generata: {store.generated_config_path}")
    console.print(f"  - Piano comandi: {store.execution_plan_path}")
    console.print(f"  - Riepilogo finale: {store.summary_path}")

    should_start = auto_start or typer.confirm(
        "Vuoi eseguire adesso i passi locali (tokenizer + preparazione dati)?",
        default=False,
    )
    if not should_start:
        return

    _run_local_execution(console, state, store)

    if typer.confirm("Vuoi avviare anche il comando di training ora?", default=False):
        training_command = _find_training_command(config["commands"])
        if training_command:
            console.print(f"Avvio: [dim]{training_command}[/dim]")
            state.execution = {"status": "running", "stage": "training_command"}
            store.save(state)
            completed = subprocess.run(training_command, shell=True, check=False)
            if completed.returncode == 0:
                state.execution = {"status": "completed", "stage": "training_command"}
            else:
                state.execution = {
                    "status": "failed",
                    "stage": "training_command",
                    "returncode": completed.returncode,
                }
            store.save(state)


def _run_local_execution(console: Console, state: WizardSessionState, store: SessionStore) -> None:
    config = state.generated_config or {}
    data_path = Path(str(config.get("paths", {}).get("source_data", "")))
    tokenizer_path = Path(str(config.get("paths", {}).get("tokenizer_dir", "")))
    prepared_data_path = Path(str(config.get("paths", {}).get("prepared_data_dir", "")))

    if not data_path.exists():
        console.print(f"[red]Dataset non trovato:[/red] {data_path}")
        state.execution = {
            "status": "blocked",
            "reason": "missing_dataset",
            "next_action": "fix_dataset_path",
        }
        store.save(state)
        return

    try:
        state.execution = {"status": "running", "stage": "tokenizer"}
        store.save(state)

        if not (tokenizer_path / "tokenizer.json").exists():
            vocab_size = int(config.get("tokenizer", {}).get("vocab_size", 8000))
            console.print(f"Training tokenizer locale (vocab={vocab_size})...")
            tokenizer = train_bpe_tokenizer(data_path=data_path, vocab_size=vocab_size)
            save_tokenizer(tokenizer, tokenizer_path)
        else:
            console.print("Tokenizer già presente, riuso quello esistente.")

        state.execution = {"status": "running", "stage": "prepare_data"}
        store.save(state)

        tokenizer = load_tokenizer(tokenizer_path)
        context_length = int(config.get("training", {}).get("context_length", 2048))
        metadata = prepare_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            output_dir=prepared_data_path,
            context_length=context_length,
        )

        state.execution = {
            "status": "prepared",
            "stage": "prepare_data",
            "prepared_tokens": int(metadata.get("total_tokens", 0)),
            "next_action": "optional_training_command",
        }
        store.save(state)
        console.print(f"Preparazione completata: {metadata.get('total_tokens', 0):,} token pronti.")

    except KeyboardInterrupt:
        state.execution = {
            "status": "stopped",
            "reason": "keyboard_interrupt",
            "next_action": "resume_step_7",
        }
        store.save(state)
        console.print("[yellow]Esecuzione interrotta. Stato salvato, puoi riprendere.[/yellow]")
    except Exception as exc:  # pragma: no cover - defensive branch
        state.execution = {
            "status": "failed",
            "reason": str(exc),
            "next_action": "inspect_logs_and_resume",
        }
        store.save(state)
        console.print(f"[red]Errore durante esecuzione guidata:[/red] {exc}")


def _build_generated_config(state: WizardSessionState, store: SessionStore) -> dict[str, Any]:
    selected_path = _path_from_state(state)
    strategy = dict(state.answers.get("strategy", {}))

    source_data = str(state.answers.get("data_path", ""))

    outputs_root = store.session_dir / "outputs"
    tokenizer_dir = outputs_root / "tokenizer"
    prepared_data_dir = outputs_root / "prepared_data"
    checkpoints_dir = outputs_root / "checkpoints"

    selected_base_model = state.selected_base_model
    default_base_path = str((Path("./models") / selected_base_model).resolve()) if selected_base_model else ""

    base_model_path = ""
    if selected_path == WizardPath.ADAPT_EXISTING:
        base_model_path = typer.prompt(
            "Percorso checkpoint modello base (ForgeAI format)",
            default=default_base_path,
        ).strip()

    tokenizer_override = typer.prompt(
        "Se hai già un tokenizer, inserisci il path (altrimenti lascia vuoto)",
        default="",
        show_default=False,
    ).strip()
    if tokenizer_override:
        tokenizer_dir = Path(tokenizer_override).resolve()

    context_length = int(strategy.get("context_length", 2048))

    commands = _build_command_plan(
        selected_path=selected_path,
        source_data=source_data,
        tokenizer_dir=str(tokenizer_dir),
        prepared_data_dir=str(prepared_data_dir),
        checkpoints_dir=str(checkpoints_dir),
        base_model_name=selected_base_model,
        base_model_path=base_model_path,
        strategy=strategy,
    )

    return {
        "version": 1,
        "session_id": state.session_id,
        "selected_path": selected_path.value,
        "selected_base_model": selected_base_model,
        "objective": state.answers.get("objective"),
        "objective_label": state.answers.get("objective_label"),
        "preset": state.preset,
        "strategy": strategy,
        "analysis": state.data_analysis,
        "hardware": state.hardware,
        "recommendation": state.recommendation,
        "tokenizer": {
            "path": str(tokenizer_dir),
            "vocab_size": 8000,
        },
        "training": {
            "context_length": context_length,
            "max_steps": int(strategy.get("max_steps", 3000)),
            "batch_size": int(strategy.get("batch_size", 2)),
            "grad_accum": int(strategy.get("grad_accum", 1)),
            "learning_rate": float(strategy.get("learning_rate", 3e-4)),
            "val_split": float(strategy.get("val_split", 0.05)),
            "save_every": int(strategy.get("save_every", 500)),
            "val_every": int(strategy.get("val_every", 200)),
            "gradient_checkpointing": bool(strategy.get("gradient_checkpointing", False)),
            "model_preset": strategy.get("model_preset", "forge-tiny"),
        },
        "paths": {
            "source_data": source_data,
            "tokenizer_dir": str(tokenizer_dir),
            "prepared_data_dir": str(prepared_data_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "base_model_path": base_model_path,
        },
        "commands": commands,
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def _build_command_plan(
    selected_path: WizardPath,
    source_data: str,
    tokenizer_dir: str,
    prepared_data_dir: str,
    checkpoints_dir: str,
    base_model_name: str | None,
    base_model_path: str,
    strategy: dict[str, Any],
) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []

    if not (Path(tokenizer_dir) / "tokenizer.json").exists():
        commands.append(
            {
                "name": "train_tokenizer",
                "reason": "Costruisce il tokenizer locale se non esiste già.",
                "command": f"forge tokenizer train --data {source_data} --vocab-size 8000 --output {tokenizer_dir}",
            }
        )

    commands.append(
        {
            "name": "prepare_data",
            "reason": "Converte i dati in shard binari per training veloce.",
            "command": (
                f"forge data prepare {source_data} --output {prepared_data_dir} "
                f"--tokenizer {tokenizer_dir} --context-length 2048"
            ),
        }
    )

    if selected_path == WizardPath.ADAPT_EXISTING:
        if base_model_name:
            commands.append(
                {
                    "name": "optional_pull_base_model",
                    "reason": "Scarica e converte il modello base se non lo hai già locale.",
                    "command": f"forge model pull {base_model_name} --output ./models",
                }
            )

        commands.append(
            {
                "name": "train_adapt_existing",
                "reason": "Adattamento del modello base (v1 supporta continued pretraining).",
                "command": (
                    "forge train --arch transformer "
                    f"--preset forge-small --data {prepared_data_dir} --tokenizer {tokenizer_dir} "
                    f"--output {checkpoints_dir} --resume {base_model_path} "
                    f"--lr {strategy.get('learning_rate', 1.8e-4)} "
                    f"--batch-size {strategy.get('batch_size', 2)} "
                    f"--grad-accum {strategy.get('grad_accum', 2)} "
                    f"--max-steps {strategy.get('max_steps', 3500)} "
                    f"--val-split {strategy.get('val_split', 0.06)} "
                    f"--save-every {strategy.get('save_every', 500)} "
                    f"--val-every {strategy.get('val_every', 200)}"
                ),
            }
        )
    else:
        commands.append(
            {
                "name": "train_from_scratch",
                "reason": "Training completo da zero con preset coerente con la strategia scelta.",
                "command": (
                    "forge train --arch transformer "
                    f"--preset {strategy.get('model_preset', 'forge-tiny')} "
                    f"--data {prepared_data_dir} --tokenizer {tokenizer_dir} "
                    f"--output {checkpoints_dir} "
                    f"--lr {strategy.get('learning_rate', 2.8e-4)} "
                    f"--batch-size {strategy.get('batch_size', 2)} "
                    f"--grad-accum {strategy.get('grad_accum', 2)} "
                    f"--max-steps {strategy.get('max_steps', 12000)} "
                    f"--val-split {strategy.get('val_split', 0.06)} "
                    f"--save-every {strategy.get('save_every', 800)} "
                    f"--val-every {strategy.get('val_every', 250)}"
                ),
            }
        )

    return commands


def _build_final_summary(state: WizardSessionState, config: dict[str, Any], store: SessionStore) -> str:
    selected_path = _path_from_state(state)
    recommendation = RouteRecommendation.from_dict(state.recommendation or {})
    analysis = DatasetAnalysis.from_dict(state.data_analysis or {})

    lines = [
        "# Forge Wizard Summary",
        "",
        f"Sessione: `{state.session_id}`",
        f"Percorso consigliato: `{recommendation.recommended_path.value}`",
        f"Percorso scelto: `{selected_path.value}`",
        f"Obiettivo: `{state.answers.get('objective_label', 'n/d')}`",
        f"Tipo dataset: `{analysis.dataset_type}`",
        f"Token stimati: `{analysis.estimated_tokens}`",
        f"Preset: `{state.preset}`",
        f"Modello base: `{state.selected_base_model or 'n/d'}`",
        "",
        "## Artefatti",
        f"- Stato sessione: `{store.state_path}`",
        f"- Config generata: `{store.generated_config_path}`",
        f"- Piano comandi: `{store.execution_plan_path}`",
        f"- Riepilogo: `{store.summary_path}`",
        "",
        "## Nota modalità",
        "- In v1 LoRA/QLoRA non sono disponibili; percorso adattamento usa continued pretraining.",
        "",
        "## Comandi principali",
    ]

    for item in config.get("commands", []):
        lines.append(f"- `{item.get('command', '')}`")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=True)


def _write_execution_plan(path: Path, commands: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in commands:
        lines.append(f"# {item.get('name', 'step')}: {item.get('reason', '')}")
        lines.append(item.get("command", ""))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_training_command(commands: list[dict[str, str]]) -> str | None:
    for item in commands:
        name = item.get("name", "")
        if name.startswith("train_"):
            return item.get("command")
    return None


def _print_base_model_table(console: Console, models: list[BaseModelCandidate]) -> None:
    table = Table(title="Modelli base compatibili (catalogo locale)")
    table.add_column("Nome")
    table.add_column("Dimensione")
    table.add_column("Memoria richiesta")
    table.add_column("Punto di forza")
    table.add_column("Lingua prevalente")
    table.add_column("Licenza")
    table.add_column("Raccomandazione")

    for model in models:
        table.add_row(
            model.name,
            model.params,
            f"{model.min_memory_gb:.1f} GB",
            model.strength,
            model.dominant_language,
            model.license,
            model.recommendation_level,
        )
    console.print(table)


def _objective_from_state(state: WizardSessionState) -> ObjectiveCategory:
    raw = str(state.answers.get("objective", ObjectiveCategory.OTHER.value))
    try:
        return ObjectiveCategory(raw)
    except ValueError:
        return ObjectiveCategory.OTHER


def _path_from_state(state: WizardSessionState) -> WizardPath:
    raw = str(state.selected_path or WizardPath.ADAPT_EXISTING.value)
    try:
        return WizardPath(raw)
    except ValueError:
        return WizardPath.ADAPT_EXISTING


def _path_label(path: WizardPath) -> str:
    if path == WizardPath.ADAPT_EXISTING:
        return "Adatta un modello esistente"
    return "Addestra da zero"


def _prompt_choice(prompt: str, options: list[str], default: int = 1) -> int:
    typer.echo(prompt)
    for idx, option in enumerate(options, start=1):
        typer.echo(f"  {idx}. {option}")

    while True:
        raw = typer.prompt("Selezione", default=str(default)).strip()
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return value
        typer.echo(f"Inserisci un numero tra 1 e {len(options)}.")


def _new_session_id() -> str:
    return datetime.utcnow().strftime("wizard-%Y%m%d-%H%M%S")


def _recommended_batch_from_backend(backend: BackendInfo) -> int:
    memory = float(backend.vram_gb or 0.0)
    if memory >= 24.0:
        return 8
    if memory >= 12.0:
        return 4
    if memory >= 8.0:
        return 2
    return 1


def _assess_expectation(expectation: str) -> list[str]:
    text = expectation.lower()
    notes: list[str] = []
    unrealistic_keywords = ("gpt-4", "claude", "superare", "perfetto", "100%")
    if any(keyword in text for keyword in unrealistic_keywords):
        notes.append(
            "Obiettivo molto ambizioso: conviene puntare prima a un modello utile su un compito specifico."
        )
    if "subito" in text or "in pochi minuti" in text:
        notes.append("I primi risultati arrivano presto, ma la qualità richiede iterazioni e tempo.")
    return notes
