/**
 * ForgeAI — Evaluation Page
 *
 * Run perplexity and benchmark evaluations against a checkpoint.
 * Calls forge-engine at /eval/* endpoints.
 */

"use client";

import { useState, FormEvent } from "react";
import Link from "next/link";

interface EvalResult {
  checkpoint_dir: string;
  loss?: number;
  perplexity?: number;
  tokens_evaluated?: number;
  tinystories?: {
    uniqueness: number;
    avg_length: number;
    diversity: number;
  };
  error?: string;
}

export default function EvalPage() {
  const [checkpointDir, setCheckpointDir] = useState("./checkpoints/run");
  const [tokenizerDir, setTokenizerDir] = useState("./tokenizers/my-tokenizer");
  const [dataPath, setDataPath] = useState("./data/processed");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<EvalResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleEval(e: FormEvent) {
    e.preventDefault();
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_dir: checkpointDir,
          tokenizer_dir: tokenizerDir,
          data_path: dataPath,
          max_batches: 50,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? `Eval failed (${res.status})`);
        return;
      }

      setResult({ checkpoint_dir: checkpointDir, ...data });
    } catch {
      setError("Could not reach forge-engine. Is it running on port 8000?");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">ForgeAI</h1>
          <p className="text-xs text-zinc-500">Evaluate</p>
        </div>
        <nav className="flex gap-6 text-sm text-zinc-400">
          <Link href="/dashboard" className="hover:text-zinc-200 transition-colors">Runs</Link>
          <Link href="/dashboard/eval" className="text-zinc-100 font-medium">Evaluate</Link>
          <Link href="/dashboard/chat" className="hover:text-zinc-200 transition-colors">Chat</Link>
          <Link href="/dashboard/hardware" className="hover:text-zinc-200 transition-colors">Hardware</Link>
        </nav>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-8 space-y-6">
        <section className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-5">
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-4">
            Perplexity Evaluation
          </h2>

          <form onSubmit={handleEval} className="space-y-4">
            <div>
              <label className="text-xs text-zinc-500 block mb-1">Checkpoint directory</label>
              <input
                type="text"
                value={checkpointDir}
                onChange={(e) => setCheckpointDir(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 font-mono focus:outline-none focus:border-zinc-500"
                placeholder="./checkpoints/run"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-500 block mb-1">Tokenizer directory</label>
              <input
                type="text"
                value={tokenizerDir}
                onChange={(e) => setTokenizerDir(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 font-mono focus:outline-none focus:border-zinc-500"
                placeholder="./tokenizers/my-tokenizer"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-500 block mb-1">Prepared dataset directory</label>
              <input
                type="text"
                value={dataPath}
                onChange={(e) => setDataPath(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 font-mono focus:outline-none focus:border-zinc-500"
                placeholder="./data/processed"
              />
            </div>

            <button
              type="submit"
              disabled={isRunning}
              className="w-full bg-brand-600 hover:bg-brand-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white py-2.5 rounded-lg text-sm font-medium transition-colors"
            >
              {isRunning ? "Running evaluation…" : "Run Eval"}
            </button>
          </form>
        </section>

        {error && (
          <div className="bg-red-950/50 border border-red-900 rounded-lg px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {result && (
          <section className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-5">
            <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-4">
              Results
            </h2>
            <div className="grid grid-cols-2 gap-4 text-sm">
              {result.loss !== undefined && (
                <div>
                  <p className="text-xs text-zinc-500">Loss</p>
                  <p className="text-zinc-100 font-mono text-xl mt-1">
                    {result.loss.toFixed(4)}
                  </p>
                </div>
              )}
              {result.perplexity !== undefined && (
                <div>
                  <p className="text-xs text-zinc-500">Perplexity</p>
                  <p className="text-zinc-100 font-mono text-xl mt-1">
                    {result.perplexity.toFixed(2)}
                  </p>
                </div>
              )}
              {result.tokens_evaluated !== undefined && (
                <div>
                  <p className="text-xs text-zinc-500">Tokens evaluated</p>
                  <p className="text-zinc-200 font-mono mt-1">
                    {result.tokens_evaluated.toLocaleString()}
                  </p>
                </div>
              )}
            </div>

            {result.tinystories && (
              <div className="mt-4 pt-4 border-t border-zinc-800">
                <p className="text-xs text-zinc-500 mb-3">TinyStories proxy check (qualitative)</p>
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className="text-xs text-zinc-600">Uniqueness</p>
                    <p className="text-zinc-200 font-mono">
                      {(result.tinystories.uniqueness * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-600">Avg length</p>
                    <p className="text-zinc-200 font-mono">
                      {result.tinystories.avg_length.toFixed(0)} tok
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-600">Diversity</p>
                    <p className="text-zinc-200 font-mono">
                      {(result.tinystories.diversity * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            <p className="text-xs text-zinc-600 mt-4 font-mono truncate">
              {result.checkpoint_dir}
            </p>
          </section>
        )}

        <section className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-4">
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-3">
            CLI Equivalent
          </h2>
          <code className="block font-mono bg-zinc-800 px-3 py-2 rounded text-zinc-300 text-xs">
            forge eval {checkpointDir} --tokenizer {tokenizerDir}
          </code>
        </section>
      </main>
    </div>
  );
}
