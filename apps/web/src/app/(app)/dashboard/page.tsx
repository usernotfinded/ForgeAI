/**
 * ForgeAI — Training Dashboard
 *
 * Main local web UI. Shows:
 *   - Active training runs with live loss/LR curves
 *   - Hardware status (backend, VRAM, recommended preset)
 *   - Cost and resource tracking
 *   - Quick actions: start run, evaluate checkpoint, open chat
 *
 * All data comes from the forge-engine FastAPI at localhost:8000.
 * No auth. This is a local single-user tool.
 */

import { Suspense } from "react";
import Link from "next/link";
import { HardwareStatus } from "@/components/dashboard/HardwareStatus";
import { RunList } from "@/components/dashboard/RunList";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { TrainingCurves } from "@/components/dashboard/TrainingCurves";
import { CostTracker } from "@/components/dashboard/CostTracker";

export const metadata = {
  title: "Dashboard — ForgeAI",
};

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
            ForgeAI
          </h1>
          <p className="text-xs text-zinc-500">Local Training Dashboard (experimental MVP)</p>
        </div>
        <nav className="flex gap-6 text-sm text-zinc-400">
          <Link href="/dashboard" className="text-zinc-100 font-medium">Runs</Link>
          <Link href="/dashboard/eval" className="hover:text-zinc-200 transition-colors">Evaluate</Link>
          <Link href="/dashboard/chat" className="hover:text-zinc-200 transition-colors">Chat</Link>
          <Link href="/dashboard/hardware" className="hover:text-zinc-200 transition-colors">Hardware</Link>
        </nav>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Hardware status bar */}
        <Suspense fallback={<div className="h-16 bg-zinc-900 rounded-lg animate-pulse" />}>
          <HardwareStatus />
        </Suspense>

        {/* Quick actions */}
        <QuickActions />

        {/* Cost tracker */}
        <CostTracker />

        {/* Training curves */}
        <section>
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-4">
            Training Curves
          </h2>
          <TrainingCurves />
        </section>

        {/* Training runs / checkpoints */}
        <section>
          <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-4">
            Checkpoints
          </h2>
          <Suspense fallback={<RunListSkeleton />}>
            <RunList />
          </Suspense>
        </section>
      </main>
    </div>
  );
}

function RunListSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="h-20 bg-zinc-900 rounded-lg animate-pulse" />
      ))}
    </div>
  );
}
