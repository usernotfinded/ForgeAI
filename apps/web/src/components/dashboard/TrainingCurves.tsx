/**
 * TrainingCurves
 * ==============
 * Real-time training loss and validation loss curves using Recharts.
 *
 * Polls /api/training-status every 5 seconds and displays:
 *   - Training loss (per step)
 *   - Validation loss (periodic)
 *   - Learning rate (right y-axis)
 *   - Gradient norm
 *   - Tokens per second
 */

"use client";

import { useEffect, useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

interface LogEntry {
  step: number;
  loss?: number;
  val_loss?: number;
  val_perplexity?: number;
  grad_norm?: number;
  learning_rate?: number;
  tokens_per_second?: number;
  total_tokens?: number;
  timestamp?: number;
}

interface TrainingCurvesProps {
  logFile?: string;
  pollInterval?: number;
}

export function TrainingCurves({
  logFile = "./checkpoints/run/train_log.jsonl",
  pollInterval = 5000,
}: TrainingCurvesProps) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/training-status?log_file=${encodeURIComponent(logFile)}&last_n=500`
      );
      const data = await res.json();

      if (data.error) {
        setError(data.error);
        return;
      }

      setError(null);
      const newEntries: LogEntry[] = data.entries ?? [];
      setEntries(newEntries);

      // Determine if training is still live (last entry within 30s)
      if (newEntries.length > 0) {
        const lastTs = newEntries[newEntries.length - 1].timestamp;
        if (lastTs) {
          setIsLive(Date.now() / 1000 - lastTs < 30);
        }
      }
    } catch {
      setError("Could not reach API");
    }
  }, [logFile]);

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, pollInterval);
    return () => clearInterval(interval);
  }, [fetchLogs, pollInterval]);

  // Separate train and val entries
  const trainEntries = entries.filter((e) => e.loss !== undefined);
  const valEntries = entries.filter((e) => e.val_loss !== undefined);

  // Merge for the chart: step → { loss, val_loss }
  const mergedData: Record<number, LogEntry> = {};
  for (const e of trainEntries) {
    mergedData[e.step] = { ...mergedData[e.step], ...e };
  }
  for (const e of valEntries) {
    mergedData[e.step] = { ...mergedData[e.step], ...e };
  }
  const chartData = Object.values(mergedData).sort((a, b) => a.step - b.step);

  if (error && entries.length === 0) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-8 text-center">
        <p className="text-zinc-400 text-sm">{error}</p>
        <p className="text-zinc-600 text-xs mt-1">
          Start a training run to see live curves here.
        </p>
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-8 text-center">
        <p className="text-zinc-400 text-sm">No training data yet.</p>
        <p className="text-zinc-600 text-xs mt-1">
          Run{" "}
          <code className="font-mono bg-zinc-800 px-1 py-0.5 rounded text-zinc-300">
            forge train
          </code>{" "}
          to start generating curves.
        </p>
      </div>
    );
  }

  const lastEntry = trainEntries[trainEntries.length - 1];
  const lastVal = valEntries[valEntries.length - 1];

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isLive ? "bg-green-500 animate-pulse" : "bg-zinc-600"
            }`}
          />
          <span className="text-zinc-400">
            {isLive ? "Training live" : "Training stopped"}
          </span>
        </div>
        {lastEntry && (
          <>
            <span className="text-zinc-600">|</span>
            <span className="text-zinc-400">
              Step <span className="text-zinc-200 font-mono">{lastEntry.step}</span>
            </span>
            <span className="text-zinc-400">
              Loss{" "}
              <span className="text-zinc-200 font-mono">
                {lastEntry.loss?.toFixed(4)}
              </span>
            </span>
            {lastEntry.tokens_per_second && (
              <span className="text-zinc-400">
                <span className="text-zinc-200 font-mono">
                  {lastEntry.tokens_per_second.toFixed(0)}
                </span>{" "}
                tok/s
              </span>
            )}
          </>
        )}
      </div>

      {/* Loss curves */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-widest mb-3">
          Loss Curves
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="step"
              stroke="#52525b"
              tick={{ fill: "#71717a", fontSize: 11 }}
              label={{ value: "Step", position: "insideBottom", offset: -5, fill: "#71717a" }}
            />
            <YAxis
              stroke="#52525b"
              tick={{ fill: "#71717a", fontSize: 11 }}
              label={{
                value: "Loss",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fill: "#71717a",
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid #3f3f46",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "#a1a1aa" }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              name="Train loss"
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={{ r: 3, fill: "#f59e0b" }}
              name="Val loss"
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Learning rate + gradient norm */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-widest mb-3">
            Learning Rate
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={trainEntries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="step" stroke="#52525b" tick={{ fill: "#71717a", fontSize: 10 }} />
              <YAxis
                stroke="#52525b"
                tick={{ fill: "#71717a", fontSize: 10 }}
                tickFormatter={(v: number) => v.toExponential(1)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #3f3f46",
                  borderRadius: 8,
                  fontSize: 11,
                }}
              />
              <Area
                type="monotone"
                dataKey="learning_rate"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.1}
                strokeWidth={1.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-widest mb-3">
            Gradient Norm
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={trainEntries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="step" stroke="#52525b" tick={{ fill: "#71717a", fontSize: 10 }} />
              <YAxis stroke="#52525b" tick={{ fill: "#71717a", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #3f3f46",
                  borderRadius: 8,
                  fontSize: 11,
                }}
              />
              <Area
                type="monotone"
                dataKey="grad_norm"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.1}
                strokeWidth={1.5}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
