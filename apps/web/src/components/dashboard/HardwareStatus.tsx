/**
 * HardwareStatus
 * ==============
 * Fetches /hardware from forge-engine and displays:
 *   - Backend (MLX / CUDA / MPS / CPU)
 *   - Device name and VRAM / unified RAM
 *   - Recommended starter preset
 *   - Any notes (e.g. "install flash-attn for 2-3x speedup")
 */

"use client";

import { useEffect, useState } from "react";

interface HardwareInfo {
  backend: string;
  device_name: string;
  vram_gb: number | null;
  unified_memory: boolean;
  bf16_supported: boolean;
  flash_attention: boolean;
  mlx_available: boolean;
  recommended_preset: string;
  recommended_dtype: string;
  notes: string[];
}

const BACKEND_COLOR: Record<string, string> = {
  mlx: "text-blue-400",
  cuda: "text-green-400",
  mps: "text-yellow-400",
  cpu: "text-zinc-400",
};

export function HardwareStatus() {
  const [hw, setHw] = useState<HardwareInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/hardware")
      .then((r) => r.json())
      .then((data) => {
        if (data.error) setError("forge-engine not running — start it with: uvicorn app.main:app");
        else setHw(data);
      })
      .catch(() => setError("forge-engine not running — start it with: uvicorn app.main:app"));
  }, []);

  if (error) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 text-sm text-zinc-400">
        <span className="text-yellow-500">⚠</span> {error}
      </div>
    );
  }

  if (!hw) {
    return <div className="h-16 bg-zinc-900 rounded-lg animate-pulse" />;
  }

  const memLabel = hw.unified_memory ? "Unified RAM" : "VRAM";
  const backendColor = BACKEND_COLOR[hw.backend] ?? "text-zinc-300";

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-5 py-4 flex flex-wrap gap-6 items-center text-sm">
      <div>
        <p className="text-xs text-zinc-500 mb-0.5">Backend</p>
        <p className={`font-semibold uppercase tracking-wide ${backendColor}`}>{hw.backend}</p>
      </div>
      <div>
        <p className="text-xs text-zinc-500 mb-0.5">Device</p>
        <p className="text-zinc-200">{hw.device_name}</p>
      </div>
      {hw.vram_gb && (
        <div>
          <p className="text-xs text-zinc-500 mb-0.5">{memLabel}</p>
          <p className="text-zinc-200">{hw.vram_gb} GB</p>
        </div>
      )}
      <div>
        <p className="text-xs text-zinc-500 mb-0.5">Dtype</p>
        <p className="text-zinc-200 font-mono text-xs">{hw.recommended_dtype}</p>
      </div>
      <div>
        <p className="text-xs text-zinc-500 mb-0.5">Recommended preset</p>
        <p className="text-zinc-200 font-mono text-xs">{hw.recommended_preset}</p>
      </div>
      <div className="flex gap-3 ml-auto">
        {hw.flash_attention && (
          <span className="text-xs bg-green-900/40 text-green-400 px-2 py-0.5 rounded">Flash Attn</span>
        )}
        {hw.mlx_available && (
          <span className="text-xs bg-blue-900/40 text-blue-400 px-2 py-0.5 rounded">MLX</span>
        )}
        {hw.bf16_supported && (
          <span className="text-xs bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded">BF16</span>
        )}
      </div>
    </div>
  );
}
