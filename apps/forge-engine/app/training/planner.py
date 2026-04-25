"""
ForgeAI Training Planner
========================
Estimates training time, electricity cost, disk usage, and
recommends hardware configuration before the user starts a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from app.core.backend import BackendInfo


def _parse_params(params_str: str) -> int:
    """Parse '400M', '7B', '1.3B' etc. to integer count."""
    s = params_str.upper().strip()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _estimate_flops_per_token(n_params: int) -> float:
    """
    Approximate FLOPs per forward pass token.
    Rule of thumb: ~6 * n_params FLOPs for a transformer (forward + backward = 2x).
    Chinchilla scaling: total training FLOPs ≈ 6 * N * D where D = token count.
    """
    return 6 * n_params


def _chinchilla_optimal_tokens(n_params: int) -> int:
    """
    Chinchilla (Hoffmann et al. 2022) optimal token count: D = 20 * N.
    Returns optimal number of training tokens for a given parameter count.
    """
    return 20 * n_params


BACKEND_TFLOPS = {
    # Approximate peak bf16 TFLOPS per device
    "mlx_m4_16gb": 15.0,      # MacBook Air M4 (estimated)
    "mlx_m2_ultra": 110.0,    # Mac Studio M2 Ultra
    "cuda_rtx4090": 165.0,    # RTX 4090 bf16
    "cuda_a100_40gb": 312.0,
    "cuda_h100_80gb": 989.0,
    "cpu": 0.05,
    "mps": 5.0,
}


@dataclass
class TrainingPlan:
    arch: str
    params: int
    data_path: str
    backend_name: str
    device_name: str
    estimated_tokens: int
    batch_tokens_per_second: float
    estimated_hours: float
    electricity_kwh: float
    kwh_cost: float
    checkpoint_size_gb: float
    recommended_dtype: str
    recommended_batch_size: int
    data_inspection: dict[str, Any]
    warnings: list[str]

    def print_summary(self, console: Console) -> None:
        days = self.estimated_hours / 24
        cost_eur = self.electricity_kwh * self.kwh_cost

        console.print()
        console.print("[bold]ForgeAI — Training Plan[/bold]")
        console.rule()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("Hardware", self.device_name)
        table.add_row("Backend", self.backend_name.upper())
        table.add_row("Architecture", self.arch)
        table.add_row("Parameters", f"{self.params / 1e6:.0f}M ({self.params / 1e9:.2f}B)")
        table.add_row("Tokens (Chinchilla optimal)", f"~{self.estimated_tokens / 1e9:.1f}B")
        table.add_row("Data path", self.data_inspection.get("resolved_path", self.data_path))
        if bool(self.data_inspection.get("prepared_metadata_found", False)):
            prepared_tokens = self.data_inspection.get("prepared_total_tokens")
            prepared_shards = self.data_inspection.get("prepared_num_shards")
            prepared_context = self.data_inspection.get("prepared_context_length")
            prepared_dtype = self.data_inspection.get("prepared_token_dtype")
            if isinstance(prepared_tokens, int):
                table.add_row("Prepared dataset tokens", f"{prepared_tokens:,}")
            if isinstance(prepared_shards, int):
                table.add_row("Prepared shards", str(prepared_shards))
            if isinstance(prepared_context, int):
                table.add_row("Prepared context length", str(prepared_context))
            if isinstance(prepared_dtype, str) and prepared_dtype:
                table.add_row("Prepared token dtype", prepared_dtype)
        table.add_row("Dtype", self.recommended_dtype)
        table.add_row("Batch size (auto)", str(self.recommended_batch_size))
        table.add_row("", "")
        table.add_row("Estimated time", f"~{self.estimated_hours:.0f}h ({days:.1f} days)")
        table.add_row("Electricity", f"~{self.electricity_kwh:.0f} kWh → ~€{cost_eur:.0f}")
        table.add_row("Checkpoint size", f"~{self.checkpoint_size_gb:.1f} GB per save")

        console.print(table)
        console.rule()

        if self.warnings:
            for w in self.warnings:
                console.print(f"  [yellow]⚠[/yellow]  {w}")
            console.print()

        console.print("  Run [bold]forge train[/bold] with the same flags to start.")
        console.print()


def estimate_training(
    arch: str,
    params: str,
    data_path: str,
    backend: "BackendInfo",
    kwh_cost: float = 0.30,
) -> TrainingPlan:
    """
    Estimate training time and cost for a given model and hardware.
    Uses Chinchilla scaling laws for token count estimation.
    """
    data_inspection = inspect_data_path(data_path)
    n_params = _parse_params(params)
    optimal_tokens = _chinchilla_optimal_tokens(n_params)
    flops_per_token = _estimate_flops_per_token(n_params)

    # Rough TFLOPS estimate per backend
    backend_tflops = {
        "mlx": 15.0,
        "cuda": 165.0,
        "mps": 5.0,
        "cpu": 0.05,
    }.get(backend.type.value, 10.0)

    # MFU (Model FLOP Utilization) — realistic is 30-50% of peak
    mfu = 0.35
    effective_tflops = backend_tflops * mfu * 1e12

    total_flops = flops_per_token * optimal_tokens
    estimated_seconds = total_flops / effective_tflops
    estimated_hours = estimated_seconds / 3600

    # Power estimate: GPU TDP (rough)
    tdp_watts = {
        "mlx": 30,
        "cuda": 450,
        "mps": 30,
        "cpu": 15,
    }.get(backend.type.value, 100)
    electricity_kwh = (tdp_watts / 1000) * estimated_hours

    # Checkpoint size: ~2 bytes/param for bf16
    checkpoint_size_gb = (n_params * 2) / 1e9

    # Recommended batch size (very rough heuristic based on VRAM/RAM)
    vram = backend.vram_gb or 8.0
    batch_size = max(1, int(vram / (n_params / 1e8)))

    warnings = []
    if vram < (n_params * 2 / 1e9):
        warnings.append(
            f"Model requires ~{n_params * 2 / 1e9:.1f} GB for weights alone. "
            f"Your hardware has {vram:.0f} GB. Gradient checkpointing will be required."
        )
    if estimated_hours > 24 * 30:
        warnings.append(
            f"Estimated training time is {estimated_hours / 24:.0f} days. "
            "Consider reducing model size or token count."
        )
    warnings.append(
        "Planner outputs are heuristic estimates based on model size + backend throughput. "
        "Use prepared dataset metadata as guidance, not as a guarantee."
    )
    prepared_tokens = data_inspection.get("prepared_total_tokens")
    if isinstance(prepared_tokens, int) and prepared_tokens > 0 and prepared_tokens < optimal_tokens:
        warnings.append(
            f"Prepared dataset has ~{prepared_tokens / 1e6:.1f}M tokens, below "
            f"the Chinchilla target (~{optimal_tokens / 1e6:.1f}M)."
        )

    return TrainingPlan(
        arch=arch,
        params=n_params,
        data_path=data_path,
        backend_name=backend.type.value,
        device_name=backend.device_name,
        estimated_tokens=optimal_tokens,
        batch_tokens_per_second=effective_tflops / flops_per_token,
        estimated_hours=estimated_hours,
        electricity_kwh=electricity_kwh,
        kwh_cost=kwh_cost,
        checkpoint_size_gb=checkpoint_size_gb,
        recommended_dtype=backend.recommended_dtype,
        recommended_batch_size=batch_size,
        data_inspection=data_inspection,
        warnings=warnings,
    )


def inspect_data_path(data_path: str | Path) -> dict[str, Any]:
    path = Path(data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    resolved = path.resolve()
    inspection: dict[str, Any] = {
        "resolved_path": str(resolved),
        "path_kind": "directory" if resolved.is_dir() else "file",
        "prepared_metadata_found": False,
        "prepared_metadata_path": None,
        "prepared_total_tokens": None,
        "prepared_num_shards": None,
        "prepared_context_length": None,
        "prepared_token_dtype": None,
    }

    metadata_path = _resolve_prepared_metadata_path(resolved)
    if metadata_path is None:
        return inspection

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        inspection["prepared_metadata_path"] = str(metadata_path.resolve())
        inspection["prepared_metadata_error"] = "invalid_json"
        return inspection

    if not isinstance(payload, dict):
        inspection["prepared_metadata_path"] = str(metadata_path.resolve())
        inspection["prepared_metadata_error"] = "invalid_payload_type"
        return inspection

    inspection["prepared_metadata_found"] = True
    inspection["prepared_metadata_path"] = str(metadata_path.resolve())
    inspection["prepared_total_tokens"] = _coerce_optional_int(payload.get("total_tokens"))
    inspection["prepared_num_shards"] = _coerce_optional_int(payload.get("num_shards"))
    inspection["prepared_context_length"] = _coerce_optional_int(payload.get("context_length"))
    token_dtype = payload.get("token_dtype")
    if isinstance(token_dtype, str) and token_dtype.strip():
        inspection["prepared_token_dtype"] = token_dtype.strip().lower()
    return inspection


def _resolve_prepared_metadata_path(path: Path) -> Path | None:
    if path.is_dir():
        candidate = path / "metadata.json"
        return candidate if candidate.exists() else None

    if path.is_file() and path.name == "metadata.json":
        return path

    if path.is_file() and path.suffix == ".bin":
        candidate = path.parent / "metadata.json"
        return candidate if candidate.exists() else None

    return None


def _coerce_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
