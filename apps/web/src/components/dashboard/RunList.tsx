/**
 * RunList
 * =======
 * Shows training runs with status, loss, step count, and quick actions.
 * Fetches data from /api/training-status and /api/checkpoints.
 */

"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

interface RunInfo {
  step: number;
  loss: number;
  val_loss: number | null;
  epoch: number;
  architecture: string;
  path: string;
  isLive: boolean;
}

export function RunList() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchRuns = useCallback(async () => {
    try {
      // Fetch checkpoints
      const ckptRes = await fetch("/api/checkpoints?dir=./checkpoints/run");
      const ckptData = await ckptRes.json();

      if (ckptData.error) {
        setError(ckptData.error);
        return;
      }

      const checkpoints: RunInfo[] = (ckptData.checkpoints ?? []).map(
        (c: Record<string, unknown>) => ({
          step: c.step as number,
          loss: c.loss as number,
          val_loss: c.val_loss as number | null,
          epoch: c.epoch as number,
          architecture: (c.architecture as string) ?? "transformer",
          path: c.path as string,
          isLive: false,
        })
      );

      // Check if training is active by looking at training logs
      try {
        const logRes = await fetch("/api/training-status?last_n=1");
        const logData = await logRes.json();
        const entries = logData.entries ?? [];
        if (entries.length > 0) {
          const lastTs = entries[entries.length - 1].timestamp;
          const isLive = lastTs && Date.now() / 1000 - lastTs < 30;
          if (checkpoints.length > 0) {
            checkpoints[checkpoints.length - 1].isLive = isLive;
          }
        }
      } catch {
        // Ignore — log fetch is optional
      }

      setError(null);
      setRuns(checkpoints);
    } catch {
      setError("forge-engine not reachable");
    }
  }, []);

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 10000);
    return () => clearInterval(interval);
  }, [fetchRuns]);

  if (error) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-8 text-center">
        <p className="text-zinc-400 text-sm">{error}</p>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-12 text-center">
        <p className="text-zinc-400 text-sm mb-2">No training runs yet.</p>
        <p className="text-zinc-600 text-xs">
          Start one with:{" "}
          <code className="font-mono bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-300">
            forge train --arch transformer --preset forge-nano --data ./data/ --tokenizer
            ./tokenizers/my-tokenizer/
          </code>
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {runs.map((run) => (
        <div
          key={run.path}
          className="bg-zinc-900 border border-zinc-800 rounded-lg px-5 py-4 flex items-center gap-6 hover:border-zinc-700 transition-colors"
        >
          {/* Status dot */}
          <div className="flex-shrink-0">
            <span
              className={`w-2.5 h-2.5 rounded-full inline-block ${
                run.isLive ? "bg-green-500 animate-pulse" : "bg-zinc-600"
              }`}
            />
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-zinc-200 truncate">
              Step {run.step.toLocaleString()}
              <span className="text-zinc-500 font-normal ml-2">
                epoch {run.epoch}
              </span>
            </p>
            <p className="text-xs text-zinc-500 truncate font-mono">
              {run.path}
            </p>
          </div>

          {/* Metrics */}
          <div className="flex gap-6 text-sm flex-shrink-0">
            <div>
              <p className="text-2xs text-zinc-500">Loss</p>
              <p className="font-mono text-zinc-200">
                {run.loss.toFixed(4)}
              </p>
            </div>
            {run.val_loss !== null && (
              <div>
                <p className="text-2xs text-zinc-500">Val Loss</p>
                <p className="font-mono text-zinc-200">
                  {run.val_loss.toFixed(4)}
                </p>
              </div>
            )}
            <div>
              <p className="text-2xs text-zinc-500">Arch</p>
              <p className="text-zinc-300 capitalize">{run.architecture}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
