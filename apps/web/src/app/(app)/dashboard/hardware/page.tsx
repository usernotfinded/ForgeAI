/**
 * ForgeAI — Hardware Page
 *
 * Detailed hardware information and system capabilities.
 */

import { HardwareStatus } from "@/components/dashboard/HardwareStatus";
import { Suspense } from "react";
import Link from "next/link";

export const metadata = {
  title: "Hardware — ForgeAI",
};

export default function HardwarePage() {
  return (
    <div className="min-h-screen bg-zinc-950">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">ForgeAI</h1>
          <p className="text-xs text-zinc-500">Hardware</p>
        </div>
        <nav className="flex gap-6 text-sm text-zinc-400">
          <Link href="/dashboard" className="hover:text-zinc-200 transition-colors">Runs</Link>
          <Link href="/dashboard/eval" className="hover:text-zinc-200 transition-colors">Evaluate</Link>
          <Link href="/dashboard/chat" className="hover:text-zinc-200 transition-colors">Chat</Link>
          <Link href="/dashboard/hardware" className="text-zinc-100 font-medium">Hardware</Link>
        </nav>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        <section>
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-4">
            System Status
          </h2>
          <Suspense fallback={<div className="h-20 bg-zinc-900 rounded-lg animate-pulse" />}>
            <HardwareStatus />
          </Suspense>
        </section>

        <section className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-5">
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-4">
            Recommended Presets
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            {[
              {
                name: "forge-nano",
                params: "~50M",
                desc: "Fast iteration, prototyping, testing on any hardware",
                vram: "< 1 GB",
              },
              {
                name: "forge-tiny",
                params: "~120M",
                desc: "Balanced quality and speed for local research",
                vram: "2–4 GB",
              },
              {
                name: "forge-small",
                params: "~310M",
                desc: "Higher quality outputs, needs a capable GPU",
                vram: "8+ GB",
              },
            ].map((preset) => (
              <div
                key={preset.name}
                className="bg-zinc-800 rounded-lg px-4 py-3 border border-zinc-700"
              >
                <p className="font-mono text-zinc-100 font-medium">{preset.name}</p>
                <p className="text-xs text-zinc-500 mt-1">{preset.params} params</p>
                <p className="text-xs text-zinc-400 mt-2">{preset.desc}</p>
                <p className="text-xs text-zinc-600 mt-1">VRAM: {preset.vram}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="bg-zinc-900 border border-zinc-800 rounded-lg px-6 py-5">
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-3">
            Quick Start
          </h2>
          <div className="space-y-2 text-sm text-zinc-400">
            <p>Start forge-engine:</p>
            <code className="block font-mono bg-zinc-800 px-3 py-2 rounded text-zinc-300 text-xs">
              uvicorn app.main:app --port 8000 --reload
            </code>
            <p className="mt-4">Train a model:</p>
            <code className="block font-mono bg-zinc-800 px-3 py-2 rounded text-zinc-300 text-xs">
              forge train --arch transformer --preset forge-nano --data ./data/ --tokenizer ./tokenizers/my-tokenizer/
            </code>
          </div>
        </section>
      </main>
    </div>
  );
}
