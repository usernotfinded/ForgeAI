/**
 * CostTracker
 * ===========
 * Live electricity cost and resource usage estimator.
 *
 * Reads training log entries and computes:
 *   - Elapsed wall-clock time
 *   - Estimated electricity consumption (kWh)
 *   - Estimated cost based on configurable €/kWh rate
 *   - Total tokens processed
 *   - Checkpoint disk usage
 */

"use client";

import { useEffect, useState, useCallback } from "react";

interface LogEntry {
  step: number;
  timestamp?: number;
  total_tokens?: number;
  tokens_per_second?: number;
}

interface CostTrackerProps {
  logFile?: string;
  kwhCost?: number;    // €/kWh — default 0.30
  tdpWatts?: number;   // GPU/system TDP in watts — default 30 (Mac) or 450 (CUDA)
  pollInterval?: number;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

function formatTokens(n: number): string {
  if (n < 1_000) return String(n);
  if (n < 1_000_000) return `${(n / 1_000).toFixed(1)}K`;
  if (n < 1_000_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  return `${(n / 1_000_000_000).toFixed(2)}B`;
}

export function CostTracker({
  logFile = "./checkpoints/run/train_log.jsonl",
  kwhCost = 0.30,
  tdpWatts = 30,
  pollInterval = 5000,
}: CostTrackerProps) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/training-status?log_file=${encodeURIComponent(logFile)}&last_n=1000`
      );
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return;
      }
      setError(null);
      setEntries(data.entries ?? []);
    } catch {
      setError("Could not reach API");
    }
  }, [logFile]);

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, pollInterval);
    return () => clearInterval(interval);
  }, [fetchLogs, pollInterval]);

  if (error || entries.length === 0) {
    return null; // Don't render if no data
  }

  // Compute metrics
  const first = entries[0];
  const last = entries[entries.length - 1];

  const elapsedSeconds =
    first.timestamp && last.timestamp ? last.timestamp - first.timestamp : 0;
  const elapsedHours = elapsedSeconds / 3600;

  const electricityKwh = (tdpWatts / 1000) * elapsedHours;
  const costEur = electricityKwh * kwhCost;

  const totalTokens = last.total_tokens ?? 0;
  const currentTps = last.tokens_per_second ?? 0;
  const totalSteps = last.step ?? 0;

  const stats = [
    {
      label: "Elapsed",
      value: formatDuration(elapsedSeconds),
      color: "text-zinc-200",
    },
    {
      label: "Tokens",
      value: formatTokens(totalTokens),
      color: "text-zinc-200",
    },
    {
      label: "Speed",
      value: `${currentTps.toFixed(0)} tok/s`,
      color: "text-zinc-200",
    },
    {
      label: "Steps",
      value: totalSteps.toLocaleString(),
      color: "text-zinc-200",
    },
    {
      label: "Energy",
      value: `${electricityKwh.toFixed(2)} kWh`,
      color: "text-yellow-400",
    },
    {
      label: "Cost",
      value: `€${costEur.toFixed(2)}`,
      color: costEur > 1 ? "text-yellow-400" : "text-green-400",
    },
  ];

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-5 py-4">
      <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-widest mb-3">
        Cost &amp; Resources
      </h3>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
        {stats.map((stat) => (
          <div key={stat.label}>
            <p className="text-2xs text-zinc-500">{stat.label}</p>
            <p className={`text-sm font-mono font-medium ${stat.color}`}>
              {stat.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
