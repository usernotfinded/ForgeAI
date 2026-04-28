"""Availability checks for the optional experimental MLX backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
import importlib
import importlib.util
import platform
import subprocess
import sys
from types import ModuleType
from typing import Any


INSTALL_HINT = 'pip install -e "apps/forge-engine[mlx]"'


class MlxUnavailableError(RuntimeError):
    """Raised when the optional MLX backend cannot be used."""


@dataclass(frozen=True)
class MlxAvailability:
    mlx_installed: bool
    mlx_importable: bool
    apple_silicon: bool
    platform: str
    machine: str
    mlx_version: str | None
    default_device: str | None
    import_error: str | None
    message: str


@dataclass(frozen=True)
class _MlxRuntimeProbe:
    importable: bool
    mlx_version: str | None
    default_device: str | None
    error: str | None


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _import_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _probe_mlx_runtime(timeout_seconds: float = 5.0) -> _MlxRuntimeProbe:
    """Probe MLX in a child process so native import crashes do not kill the CLI."""
    probe_code = """
import json

try:
    import mlx
    import mlx.core as mx

    default_device_fn = getattr(mx, "default_device", None)
    default_device = str(default_device_fn()) if callable(default_device_fn) else None
    print(json.dumps({
        "importable": True,
        "mlx_version": getattr(mlx, "__version__", None),
        "default_device": default_device,
        "error": None,
    }))
except BaseException as exc:
    print(json.dumps({
        "importable": False,
        "mlx_version": None,
        "default_device": None,
        "error": f"{type(exc).__name__}: {exc}",
    }))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _MlxRuntimeProbe(
            importable=False,
            mlx_version=None,
            default_device=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        return _MlxRuntimeProbe(
            importable=False,
            mlx_version=None,
            default_device=None,
            error=stderr or f"MLX runtime probe exited with code {completed.returncode}",
        )

    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        return _MlxRuntimeProbe(
            importable=False,
            mlx_version=None,
            default_device=None,
            error="MLX runtime probe produced no output.",
        )

    try:
        payload = json.loads(stdout_lines[-1])
    except json.JSONDecodeError as exc:
        return _MlxRuntimeProbe(
            importable=False,
            mlx_version=None,
            default_device=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return _MlxRuntimeProbe(
        importable=bool(payload.get("importable")),
        mlx_version=(
            str(payload["mlx_version"]) if payload.get("mlx_version") is not None else None
        ),
        default_device=(
            str(payload["default_device"]) if payload.get("default_device") is not None else None
        ),
        error=str(payload["error"]) if payload.get("error") is not None else None,
    )


def platform_appears_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _availability(
    *,
    mlx_installed: bool,
    mlx_importable: bool,
    mlx_version: str | None,
    default_device: str | None,
    import_error: str | None,
    message: str,
) -> MlxAvailability:
    return MlxAvailability(
        mlx_installed=mlx_installed,
        mlx_importable=mlx_importable,
        apple_silicon=platform_appears_apple_silicon(),
        platform=sys.platform,
        machine=platform.machine(),
        mlx_version=mlx_version,
        default_device=default_device,
        import_error=import_error,
        message=message,
    )


def check_availability() -> MlxAvailability:
    """Return MLX availability without requiring MLX to be installed."""
    if not _module_available("mlx"):
        return _availability(
            mlx_installed=False,
            mlx_importable=False,
            mlx_version=None,
            default_device=None,
            import_error=None,
            message=f"MLX is not installed. Install optional dependencies with: {INSTALL_HINT}",
        )

    probe = _probe_mlx_runtime()
    if not probe.importable:
        return _availability(
            mlx_installed=True,
            mlx_importable=False,
            mlx_version=probe.mlx_version,
            default_device=probe.default_device,
            import_error=probe.error,
            message=(
                "MLX is installed but its runtime probe failed. "
                f"Reinstall optional dependencies with: {INSTALL_HINT}"
            ),
        )

    notes: list[str] = ["MLX is installed and importable."]
    if not platform_appears_apple_silicon():
        notes.append("This platform does not appear to be Apple Silicon.")

    if probe.default_device is not None:
        notes.append(f"Default device: {probe.default_device}.")

    return _availability(
        mlx_installed=True,
        mlx_importable=True,
        mlx_version=probe.mlx_version,
        default_device=probe.default_device,
        import_error=None,
        message=" ".join(notes),
    )


def require_mlx() -> tuple[ModuleType, ModuleType, ModuleType]:
    availability = check_availability()
    if not availability.mlx_importable:
        raise MlxUnavailableError(availability.message)

    try:
        mx = _import_module("mlx.core")
        nn = _import_module("mlx.nn")
        optimizers = _import_module("mlx.optimizers")
    except Exception as exc:
        raise MlxUnavailableError(
            "MLX is importable, but required MLX runtime modules are unavailable. "
            f"Reinstall optional dependencies with: {INSTALL_HINT}"
        ) from exc

    return mx, nn, optimizers


def require_attr(module: ModuleType, name: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise MlxUnavailableError(f"MLX module {module.__name__!r} is missing {name!r}.") from exc
