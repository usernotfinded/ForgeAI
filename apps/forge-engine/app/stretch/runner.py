from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from .executor import create_persistent_variant
from .model_inspector import ModelInspectionError, ModelType, inspect_model
from .planner import (
    AggressivenessProfile,
    StretchCompatibility,
    StretchPlan,
    analyze_compatibility,
    build_stretch_plan,
    stratified_consent_level,
    validate_target_context,
)
from .reconstructor import reconstruct_variant_from_manifest, run_minimal_reconstruction_demo
from .session import StretchSessionState, StretchSessionStore
from .validator import validate_variant


def run_stretch(
    console: Console,
    model_path: str | None = None,
    target_context: str | None = None,
    aggressiveness: str | None = None,
    session_dir: str = "./.forge/stretch",
    output_dir: str = "./models/stretched",
    non_interactive: bool = False,
    config_path: str | None = None,
) -> None:
    store = StretchSessionStore(session_dir)
    state = _load_or_initialize_session(console, store, non_interactive=non_interactive)
    if state is None:
        return

    config_overrides = _load_non_interactive_config(config_path) if config_path else {}

    if state.current_step <= 1 and not _step_1_load_model(
        console,
        state,
        store,
        model_path=model_path or config_overrides.get("model_path"),
        non_interactive=non_interactive,
    ):
        return

    if state.current_step <= 2 and not _step_2_compatibility(
        console,
        state,
        store,
        non_interactive=non_interactive,
    ):
        return

    if state.current_step <= 3 and not _step_3_target_context(
        console,
        state,
        store,
        target_context=target_context or config_overrides.get("target_context"),
        non_interactive=non_interactive,
    ):
        return

    if state.current_step <= 4 and not _step_4_aggressiveness(
        console,
        state,
        store,
        aggressiveness=aggressiveness or config_overrides.get("aggressiveness"),
        non_interactive=non_interactive,
    ):
        return

    if state.current_step <= 5 and not _step_5_review_consent(
        console,
        state,
        store,
        output_dir=output_dir,
        non_interactive=non_interactive,
    ):
        return

    if state.current_step <= 6 and not _step_6_execute(
        console,
        state,
        store,
        non_interactive=non_interactive,
    ):
        return

    if state.current_step <= 7 and not _step_7_validate(
        console,
        state,
        store,
    ):
        return

    _step_8_finalize(console, state, store)


def _load_or_initialize_session(
    console: Console,
    store: StretchSessionStore,
    non_interactive: bool,
) -> StretchSessionState | None:
    existing_state = store.load()
    if existing_state is None:
        state = StretchSessionState(session_id=_new_session_id())
        store.save(state)
        return state

    if non_interactive:
        return existing_state

    options = ["Riprendi", "Ricomincia", "Mostra ultimo riepilogo"]
    choice = _prompt_choice(
        "Trovata una sessione stretch precedente. Come vuoi procedere?",
        options,
        default=1,
    )

    if choice == 3:
        summary = store.read_summary()
        if summary:
            console.print("\n[bold]Ultimo riepilogo stretch salvato[/bold]\n")
            console.print(summary)
        else:
            console.print("[yellow]Nessun riepilogo disponibile per questa sessione.[/yellow]")
        return None

    if choice == 2:
        store.reset()
        state = StretchSessionState(session_id=_new_session_id())
        store.save(state)
        return state

    return existing_state


def _step_1_load_model(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
    model_path: str | None,
    non_interactive: bool,
) -> bool:
    console.rule("[bold]Step 1/8 — Caricamento Modello[/bold]")
    console.print("Perché: lo stretch richiede un checkpoint persistente compatibile come input.")

    selected_model_path = model_path or state.model_path
    if not selected_model_path and non_interactive:
        console.print("[red]In modalità non interattiva devi fornire --model o config.model_path.[/red]")
        return False

    while not selected_model_path:
        selected_model_path = typer.prompt("Percorso checkpoint modello").strip()

    try:
        inspection = inspect_model(selected_model_path)
    except ModelInspectionError as exc:
        console.print(f"[red]{exc}[/red]")
        return False

    state.model_path = inspection.resolved_path
    state.model_type = inspection.model_type.value
    state.architecture = inspection.architecture
    state.native_context = inspection.native_context
    state.add_log(
        f"Modello caricato '{inspection.model_name}' ({inspection.architecture}, contesto nativo {inspection.native_context})."
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value")
    table.add_row("Modello", inspection.model_name)
    table.add_row("Percorso", inspection.resolved_path)
    table.add_row("Architettura", inspection.architecture)
    table.add_row("Tipo modello", _model_type_label(inspection.model_type))
    table.add_row("Contesto nativo", str(inspection.native_context))
    table.add_row("Compatibile RoPE", "sì" if inspection.rope_based else "no")
    console.print(table)

    if inspection.limits:
        for limit in inspection.limits:
            console.print(f"[yellow]Limite:[/yellow] {limit}")

    state.mark_step_completed(1)
    store.save(state)
    return True


def _step_2_compatibility(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
    non_interactive: bool,
) -> bool:
    console.rule("[bold]Step 2/8 — Analisi Compatibilità[/bold]")
    console.print("Perché: la v1 supporta solo stretch locale persistente YaRN su checkpoint RoPE Transformer compatibili.")

    if not state.model_path:
        console.print("[red]Nessun modello caricato nello stato della sessione.[/red]")
        return False

    inspection = inspect_model(state.model_path)

    backend_name, vram_gb, unified_memory = _detect_backend_info(console)
    if backend_name is None:
        return False

    compatibility = analyze_compatibility(
        inspection=inspection,
        backend=backend_name,
        vram_gb=vram_gb,
        unified_memory=unified_memory,
    )
    state.compatibility = compatibility.to_dict()

    compatibility_table = Table(show_header=False, box=None, padding=(0, 2))
    compatibility_table.add_column("Field", style="dim")
    compatibility_table.add_column("Value")
    compatibility_table.add_row("Backend", backend_name)
    compatibility_table.add_row("Metodo", compatibility.method or "n/d")
    compatibility_table.add_row(
        "Target massimo realistico",
        str(compatibility.max_realistic_target) if compatibility.max_realistic_target else "n/d",
    )
    compatibility_table.add_row(
        "Target validi",
        ", ".join(str(item) for item in compatibility.valid_targets) or "nessuno",
    )
    console.print(compatibility_table)

    if compatibility.warnings:
        for warning in compatibility.warnings:
            console.print(f"[yellow]Avviso:[/yellow] {warning}")

    if compatibility.errors:
        for error in compatibility.errors:
            console.print(f"[red]Errore di compatibilità:[/red] {error}")
        state.process_status = "blocked"
        state.add_log("Analisi compatibilità fallita.")
        store.save(state)
        return False

    state.add_log("Analisi compatibilità superata.")
    state.mark_step_completed(2)
    store.save(state)
    return True


def _step_3_target_context(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
    target_context: str | int | None,
    non_interactive: bool,
) -> bool:
    console.rule("[bold]Step 3/8 — Proposta Target Context[/bold]")
    console.print("Perché: il target deve essere strettamente maggiore del contesto nativo e supportato dalla ricetta v1.")

    native_context = int(state.native_context or 0)
    compatibility = _compatibility_from_state(state)
    valid_targets = compatibility.valid_targets

    if not valid_targets:
        console.print("[red]Nessun target context valido per questa configurazione.[/red]")
        return False

    _print_target_overview(console, compatibility)

    selected_target: int | None = None

    if target_context is not None:
        selected_target = _parse_context_value(target_context)
        if selected_target is None:
            console.print("[red]Impossibile interpretare il target context richiesto.[/red]")
            if non_interactive:
                return False
        else:
            validation = validate_target_context(
                native_context=native_context,
                target_context=selected_target,
                valid_targets=valid_targets,
            )
            if not validation.is_valid:
                console.print(f"[red]{validation.reason}[/red]")
                _print_suggested_targets(console, validation.suggested_targets)
                if non_interactive:
                    return False
                selected_target = None

    if selected_target is None:
        if non_interactive:
            selected_target = compatibility.recommended_target
        else:
            default_target = str(compatibility.recommended_target or valid_targets[0])
            raw = typer.prompt(
                "Scegli il target context (token)",
                default=default_target,
            )
            selected_target = _parse_context_value(raw)

        if selected_target is None:
            console.print("[red]Impossibile interpretare il target context richiesto.[/red]")
            return False

    validation = validate_target_context(
        native_context=native_context,
        target_context=selected_target,
        valid_targets=valid_targets,
    )
    if not validation.is_valid:
        console.print(f"[red]{validation.reason}[/red]")
        _print_suggested_targets(console, validation.suggested_targets)
        return False

    state.target_context = selected_target
    state.add_log(f"Target context selezionato: {selected_target}.")
    state.mark_step_completed(3)
    store.save(state)
    return True


def _step_4_aggressiveness(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
    aggressiveness: str | None,
    non_interactive: bool,
) -> bool:
    console.rule("[bold]Step 4/8 — Profilo di Aggressività[/bold]")
    console.print("Perché: i profili modulano rischio/costo restando nello stesso metodo YaRN.")

    chosen_profile: AggressivenessProfile | None = None

    if aggressiveness:
        chosen_profile = _parse_profile(aggressiveness)
        if chosen_profile is None and non_interactive:
            console.print("[red]Profilo di aggressività non valido in modalità non interattiva.[/red]")
            return False

    if chosen_profile is None:
        if non_interactive:
            chosen_profile = AggressivenessProfile.BALANCED
        else:
            options = [
                "Prudente — rischio minore, estensione moderata.",
                "Bilanciato — compromesso migliore.",
                "Ambizioso — target più spinto e rischio drift più alto.",
            ]
            selection = _prompt_choice("Seleziona il profilo:", options, default=2)
            chosen_profile = {
                1: AggressivenessProfile.PRUDENT,
                2: AggressivenessProfile.BALANCED,
                3: AggressivenessProfile.AMBITIOUS,
            }[selection]

    native_context = int(state.native_context or 0)
    target_context = int(state.target_context or 0)
    plan = build_stretch_plan(
        native_context=native_context,
        target_context=target_context,
        profile=chosen_profile,
    )

    state.aggressiveness = chosen_profile.value
    state.plan = plan.to_dict()
    state.add_log(f"Profilo selezionato: {chosen_profile.value}.")
    state.mark_step_completed(4)
    store.save(state)

    console.print(f"Profilo: [bold]{chosen_profile.value}[/bold]")
    console.print(f"Metodo: {plan.method} (unico metodo supportato in v1)")
    console.print(f"Moltiplicatore costo computazionale stimato: x{plan.estimated_cost_multiplier}")
    console.print(f"Moltiplicatore tempo stimato: x{plan.estimated_time_multiplier}")

    return True


def _step_5_review_consent(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
    output_dir: str,
    non_interactive: bool,
) -> bool:
    console.rule("[bold]Step 5/8 — Review Finale e Consenso Stratificato[/bold]")
    console.print("Perché: lo stretch cambia affidabilità/latenza; il consenso esplicito evita esecuzioni rischiose non intenzionali.")

    compatibility = _compatibility_from_state(state)
    plan = _plan_from_state(state)

    review = Table(show_header=False, box=None, padding=(0, 2))
    review.add_column("Field", style="dim")
    review.add_column("Value")
    review.add_row("Modello sorgente", str(state.model_path))
    review.add_row("Tipo modello", str(state.model_type))
    review.add_row("Contesto nativo", str(state.native_context))
    review.add_row("Target context", str(state.target_context))
    review.add_row("Profilo", str(state.aggressiveness))
    review.add_row("Metodo", plan.method)
    review.add_row("Stima costo", f"x{plan.estimated_cost_multiplier}")
    review.add_row("Stima tempo", f"x{plan.estimated_time_multiplier}")
    review.add_row("Cartella output", output_dir)
    console.print(review)

    level, messages = stratified_consent_level(plan, compatibility)

    console.print("\n[bold]Consenso stratificato[/bold]")
    console.print("Livello 1 — Consiglio: un contesto più lungo può aiutare la copertura, ma non garantisce qualità migliore.")

    if level >= 2:
        console.print("Livello 2 — Warning forte:")
        for message in messages:
            console.print(f"  - {message}")

    if level >= 3:
        if non_interactive:
            console.print("[red]Serve override livello 3 in modalità interattiva. Esecuzione non interattiva fermata.[/red]")
            return False
        typed = typer.prompt(
            "Override livello 3 richiesto. Scrivi 'PROCEDI COMUNQUE' per continuare",
            default="",
            show_default=False,
        ).strip()
        if typed != "PROCEDI COMUNQUE":
            console.print("[yellow]Stretch annullato: override esplicito non confermato.[/yellow]")
            state.process_status = "stopped"
            state.add_log("Annullato dall'utente al consenso stratificato livello 3.")
            store.save(state)
            return False

    if not non_interactive:
        proceed = typer.confirm(
            "Confermi l'esecuzione persistente? (v1 produce mapping artifact + manifest, non full checkpoint)",
            default=True,
        )
        if not proceed:
            console.print("[yellow]Stretch annullato dall'utente prima dell'esecuzione.[/yellow]")
            state.process_status = "stopped"
            state.add_log("Annullato dall'utente prima dell'esecuzione.")
            store.save(state)
            return False

    state.generated_config = _build_generated_config(state, output_dir)
    state.expected_outputs = _expected_outputs_from_config(state.generated_config)
    state.add_log("Review finale accettata.")
    state.mark_step_completed(5)
    store.save(state)
    return True


def _step_6_execute(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
    non_interactive: bool,
) -> bool:
    del non_interactive  # kept for future extension

    console.rule("[bold]Step 6/8 — Esecuzione Persistente[/bold]")
    console.print("Perché: crea una variante long-context riusabile su disco, non un override solo runtime.")

    if not state.model_path or not state.plan or not state.generated_config:
        console.print("[red]Prerequisiti mancanti nello stato di sessione.[/red]")
        return False

    inspection = inspect_model(state.model_path)
    plan = _plan_from_state(state)
    output_root = state.generated_config.get("output_root")

    try:
        state.process_status = "running"
        state.add_log("Esecuzione stretch persistente avviata.")
        store.save(state)

        artifacts = create_persistent_variant(
            inspection=inspection,
            plan=plan,
            output_root=str(output_root),
        )

        state.produced_outputs = sorted(set(state.produced_outputs + artifacts.output_paths))
        state.generated_config["variant"] = artifacts.to_dict()
        state.add_log(f"Variante persistente creata: {artifacts.variant_dir}")

        console.print(f"Variante creata: [bold]{artifacts.variant_name}[/bold]")
        console.print(f"Percorso: {artifacts.variant_dir}")
        console.print(
            "Nota v1: la persistenza è `adapter_plus_manifest` "
            "(artefatto mapping YaRN + manifest deterministico), non `full_checkpoint`."
        )

        state.mark_step_completed(6)
        store.save(state)
        return True
    except KeyboardInterrupt:
        state.process_status = "interrupted"
        state.add_log("Esecuzione interrotta dall'utente. Resume disponibile.")
        store.save(state)
        console.print("[yellow]Esecuzione interrotta. Stato salvato, puoi riprendere.[/yellow]")
        return False
    except Exception as exc:  # pragma: no cover - defensive branch
        state.process_status = "failed"
        state.add_log(f"Esecuzione fallita: {exc}")
        store.save(state)
        console.print(f"[red]Esecuzione stretch fallita:[/red] {exc}")
        return False


def _step_7_validate(
    console: Console,
    state: StretchSessionState,
    store: StretchSessionStore,
) -> bool:
    console.rule("[bold]Step 7/8 — Validazione[/bold]")
    console.print("Perché: verifica regressioni a contesto corto e comportamento a contesto lungo dopo lo stretch.")

    plan = _plan_from_state(state)
    variant_info = (state.generated_config or {}).get("variant", {})
    variant_dir = variant_info.get("variant_dir")
    if not variant_dir:
        console.print("[red]Percorso variante mancante: impossibile validare.[/red]")
        return False

    try:
        validation = validate_variant(variant_dir=variant_dir, plan=plan)
    except Exception as exc:  # pragma: no cover - defensive branch
        state.process_status = "failed"
        state.add_log(f"Validazione fallita con eccezione: {exc}")
        store.save(state)
        console.print(f"[red]Validazione fallita:[/red] {exc}")
        return False

    reconstruction_check = _verify_variant_reconstruction(variant_dir=Path(variant_dir))
    state.validation = validation.to_dict()
    state.validation["reconstruction_check"] = reconstruction_check

    report_path = Path(variant_dir) / "validation_report.json"
    report_path.write_text(json.dumps(state.validation, indent=2), encoding="utf-8")
    state.produced_outputs = sorted(set(state.produced_outputs + [str(report_path.resolve())]))

    short_status = "PASS" if validation.short_context_check.passed else "FAIL"
    long_status = "PASS" if validation.long_context_check.passed else "FAIL"
    structural_status = "PASS" if validation.structural_check.passed else "FAIL"
    persistence_status = "PASS" if validation.persistence_check.passed else "FAIL"
    reconstruction_status = "PASS" if reconstruction_check.get("passed") else "FAIL"
    console.print(f"Controllo strutturale: [bold]{structural_status}[/bold]")
    console.print(f"Controllo persistenza: [bold]{persistence_status}[/bold]")
    console.print(f"Controllo ricostruzione: [bold]{reconstruction_status}[/bold]")
    console.print(f"Validazione short-context: [bold]{short_status}[/bold]")
    console.print(f"Validazione long-context: [bold]{long_status}[/bold]")
    console.print(validation.summary)
    console.print(
        "Nota: questa validazione v1 è locale/proxy. È utile per regressioni tecniche, ma non sostituisce benchmark applicativi completi."
    )

    if not validation.overall_passed or not bool(reconstruction_check.get("passed", False)):
        state.process_status = "validation_failed"
        state.add_log("Validazione fallita.")
        store.save(state)
        return False

    state.add_log("Validazione superata.")
    state.mark_step_completed(7)
    store.save(state)
    return True


def _step_8_finalize(console: Console, state: StretchSessionState, store: StretchSessionStore) -> None:
    console.rule("[bold]Step 8/8 — Artefatti Finali[/bold]")
    console.print("Perché: salva output e metadati deterministici per riuso e tracciabilità.")

    store.ensure_dirs()
    state.process_status = "completed"
    state.mark_step_completed(8)
    state.add_log("Sessione stretch completata.")

    if state.generated_config is not None:
        store.generated_config_path.write_text(
            json.dumps(state.generated_config, indent=2),
            encoding="utf-8",
        )

    final_report = _build_final_report(state, store)
    store.final_report_path.write_text(final_report, encoding="utf-8")
    store.summary_path.write_text(final_report, encoding="utf-8")

    state.produced_outputs = sorted(
        set(
            state.produced_outputs
            + [
                str(store.state_path.resolve()),
                str(store.generated_config_path.resolve()),
                str(store.final_report_path.resolve()),
            ]
        )
    )
    store.save(state)

    console.print("Artefatti finali:")
    console.print(f"  - Stato sessione: {store.state_path}")
    console.print(f"  - Config generata: {store.generated_config_path}")
    console.print(f"  - Report finale: {store.final_report_path}")


def _detect_backend_info(console: Console) -> tuple[str | None, float | None, bool]:
    try:
        from app.core.backend import get_backend
    except Exception as exc:  # pragma: no cover - depends on runtime env
        console.print(
            "[red]Rilevamento backend fallito. Verifica che le dipendenze di forge-engine siano installate.[/red]"
        )
        console.print(f"[dim]{exc}[/dim]")
        return None, None, False

    backend = get_backend()
    return backend.type.value, backend.vram_gb, backend.unified_memory


def _build_generated_config(state: StretchSessionState, output_dir: str) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "version": 1,
        "session_id": state.session_id,
        "model_path": state.model_path,
        "model_type": state.model_type,
        "architecture": state.architecture,
        "native_context": state.native_context,
        "target_context": state.target_context,
        "aggressiveness": state.aggressiveness,
        "method": "yarn",
        "compatibility": state.compatibility,
        "plan": state.plan,
        "output_root": str(Path(output_dir).resolve()),
        "generated_at": timestamp,
    }


def _expected_outputs_from_config(config: dict[str, Any]) -> list[str]:
    output_root = Path(str(config.get("output_root", "./models/stretched")))
    target = int(config.get("target_context", 0) or 0)
    model_path = Path(str(config.get("model_path", "model")))
    variant_name = f"{model_path.name}-{_format_context_label(target)}-yarn"
    variant_dir = output_root / variant_name

    return [
        str((variant_dir / "model.pt").resolve()),
        str((variant_dir / "metadata.json").resolve()),
        str((variant_dir / "stretch_manifest.json").resolve()),
        str((variant_dir / "stretch_metadata.json").resolve()),
        str((variant_dir / "stretch_adapter.bin").resolve()),
        str((variant_dir / "validation_report.json").resolve()),
    ]


def _build_final_report(state: StretchSessionState, store: StretchSessionStore) -> str:
    variant_info = (state.generated_config or {}).get("variant", {})
    final_artifact_type = variant_info.get("final_artifact_type", "unknown")
    variant_dir = variant_info.get("variant_dir", "n/d")
    lines = [
        "# Report Forge Stretch",
        "",
        f"Sessione: `{state.session_id}`",
        f"Modello sorgente: `{state.model_path}`",
        f"Tipo modello: `{state.model_type}`",
        f"Architettura: `{state.architecture}`",
        f"Contesto nativo: `{state.native_context}`",
        f"Target context scelto: `{state.target_context}`",
        f"Aggressività: `{state.aggressiveness}`",
        "Metodo usato: `yarn`",
        f"Tipo persistenza finale: `{final_artifact_type}`",
        f"Percorso variante finale: `{variant_dir}`",
        f"Stato processo: `{state.process_status}`",
        "",
        "## Trade-off e Limiti v1",
        "- Un contesto più lungo non significa automaticamente qualità migliore.",
        "- Target aggressivi aumentano latenza e costo computazionale.",
        "- La v1 salva una variante persistente come `adapter_plus_manifest` (non `full_checkpoint`).",
        "- In `adapter_plus_manifest`, `stretch_adapter.bin` è un artefatto di mapping YaRN per ricostruzione/validazione locale.",
        "- La validazione è locale/proxy e non sostituisce benchmark semantici applicativi completi.",
        "",
        "## Output prodotti",
    ]

    for output_path in state.produced_outputs:
        lines.append(f"- `{output_path}`")

    validation = state.validation or {}
    if validation:
        structural = validation.get("structural_check", {})
        persistence = validation.get("persistence_check", {})
        reconstruction = validation.get("reconstruction_check", {})
        short_ctx = validation.get("short_context_check", {})
        long_ctx = validation.get("long_context_check", {})
        lines.extend(
            [
                "",
                "## Esito Validazioni",
                f"- Controlli strutturali: `{'PASS' if structural.get('passed') else 'FAIL'}`",
                f"- Controlli persistenza: `{'PASS' if persistence.get('passed') else 'FAIL'}`",
                f"- Controllo ricostruzione variante: `{'PASS' if reconstruction.get('passed') else 'FAIL'}`",
                f"- Validazione short-context: `{'PASS' if short_ctx.get('passed') else 'FAIL'}`",
                f"- Validazione long-context: `{'PASS' if long_ctx.get('passed') else 'FAIL'}`",
            ]
        )

    lines.extend(["", "## Stato Sessione", f"- `{store.state_path.resolve()}`", ""])
    return "\n".join(lines)


def _model_type_label(model_type: ModelType) -> str:
    labels = {
        ModelType.BASE_SUPPORTED: "modello base (supportato)",
        ModelType.ADAPTED: "modello già adattato/fine-tuned",
        ModelType.ADAPTER_SEPARATED: "modello con adapter separati",
        ModelType.ALREADY_STRETCHED: "modello già esteso",
        ModelType.UNKNOWN: "sconosciuto",
    }
    return labels.get(model_type, model_type.value)


def _compatibility_from_state(state: StretchSessionState) -> StretchCompatibility:
    if not state.compatibility:
        raise RuntimeError("Compatibility data missing from session state.")

    recipe_stub = None
    from .registry import get_stretch_recipe

    architecture = str(state.architecture or "")
    recipe_stub = get_stretch_recipe(architecture)

    payload = state.compatibility
    return StretchCompatibility(
        is_supported=bool(payload.get("is_supported", False)),
        backend=str(payload.get("backend", "unknown")),
        method=str(payload.get("method") or "yarn"),
        errors=[str(item) for item in payload.get("errors", [])],
        warnings=[str(item) for item in payload.get("warnings", [])],
        valid_targets=[int(item) for item in payload.get("valid_targets", [])],
        recommended_target=(
            int(payload["recommended_target"]) if payload.get("recommended_target") is not None else None
        ),
        prudent_target=(
            int(payload["prudent_target"]) if payload.get("prudent_target") is not None else None
        ),
        ambitious_target=(
            int(payload["ambitious_target"]) if payload.get("ambitious_target") is not None else None
        ),
        max_realistic_target=(
            int(payload["max_realistic_target"])
            if payload.get("max_realistic_target") is not None
            else None
        ),
        recipe=recipe_stub,
    )


def _plan_from_state(state: StretchSessionState) -> StretchPlan:
    if not state.plan:
        raise RuntimeError("Stretch plan missing from state.")

    payload = state.plan
    profile = _parse_profile(str(payload.get("profile", AggressivenessProfile.BALANCED.value)))
    if profile is None:
        profile = AggressivenessProfile.BALANCED

    return StretchPlan(
        method=str(payload.get("method", "yarn")),
        profile=profile,
        native_context=int(payload.get("native_context", state.native_context or 0)),
        target_context=int(payload.get("target_context", state.target_context or 0)),
        context_ratio=float(payload.get("context_ratio", 1.0)),
        yarn_config=dict(payload.get("yarn_config", {})),
        estimated_cost_multiplier=float(payload.get("estimated_cost_multiplier", 1.0)),
        estimated_time_multiplier=float(payload.get("estimated_time_multiplier", 1.0)),
        risk_level=int(payload.get("risk_level", 0)),
        risk_notes=[str(item) for item in payload.get("risk_notes", [])],
    )


def _load_non_interactive_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Percorso config non trovato: {config_path}")

    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError("La config stretch deve essere un oggetto JSON.")
    return payload


def _print_target_overview(console: Console, compatibility: StretchCompatibility) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Item", style="dim")
    table.add_column("Value")
    table.add_row("Target prudente", str(compatibility.prudent_target))
    table.add_row("Target consigliato", str(compatibility.recommended_target))
    table.add_row("Target ambizioso", str(compatibility.ambitious_target))
    table.add_row("Tutti i target validi", ", ".join(str(item) for item in compatibility.valid_targets))
    console.print(table)


def _print_suggested_targets(console: Console, targets: list[int]) -> None:
    if not targets:
        return
    console.print("Target validi suggeriti:")
    for item in targets:
        console.print(f"  - {item}")


def _parse_context_value(value: str | int) -> int | None:
    if isinstance(value, int):
        return value

    raw = str(value).strip().lower()
    if not raw:
        return None

    if raw.endswith("k"):
        try:
            return int(float(raw[:-1]) * 1024)
        except ValueError:
            return None

    try:
        return int(raw)
    except ValueError:
        return None


def _parse_profile(value: str) -> AggressivenessProfile | None:
    normalized = value.strip().lower()
    mapping = {
        "prudent": AggressivenessProfile.PRUDENT,
        "prudente": AggressivenessProfile.PRUDENT,
        "balanced": AggressivenessProfile.BALANCED,
        "bilanciato": AggressivenessProfile.BALANCED,
        "ambitious": AggressivenessProfile.AMBITIOUS,
        "ambizioso": AggressivenessProfile.AMBITIOUS,
    }
    return mapping.get(normalized)


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


def _verify_variant_reconstruction(variant_dir: Path) -> dict[str, Any]:
    manifest_path = variant_dir / "stretch_manifest.json"
    if not manifest_path.exists():
        return {
            "passed": False,
            "details": "Manifest stretch assente: verifica ricostruzione non eseguita.",
            "metrics": {"manifest_path": str(manifest_path.resolve())},
        }

    try:
        reconstructed = reconstruct_variant_from_manifest(manifest_path)
        demo = run_minimal_reconstruction_demo(reconstructed)
        passed = bool(demo.get("stretch_retrieved_all", False))
        return {
            "passed": passed,
            "details": (
                "Ricostruzione da manifest riuscita e demo minima long-context completata."
                if passed
                else "Ricostruzione eseguita ma demo minima long-context non superata."
            ),
            "metrics": demo,
        }
    except Exception as exc:  # pragma: no cover - defensive branch
        return {
            "passed": False,
            "details": f"Errore durante la verifica ricostruzione: {exc}",
            "metrics": {"manifest_path": str(manifest_path.resolve())},
        }


def _new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("stretch-%Y%m%d-%H%M%S")


def _format_context_label(target_context: int) -> str:
    if target_context % 1024 == 0:
        return f"{target_context // 1024}k"
    return str(target_context)
