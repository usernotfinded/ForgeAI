"""Optional experimental MLX backend foundation helpers."""

from app.experimental.mlx_backend.availability import (
    INSTALL_HINT,
    MlxAvailability,
    MlxUnavailableError,
    check_availability,
    platform_appears_apple_silicon,
)
from app.experimental.mlx_backend.smoke import MlxSmokeResult, run_smoke

__all__ = [
    "INSTALL_HINT",
    "MlxAvailability",
    "MlxSmokeResult",
    "MlxUnavailableError",
    "check_availability",
    "platform_appears_apple_silicon",
    "run_smoke",
]

