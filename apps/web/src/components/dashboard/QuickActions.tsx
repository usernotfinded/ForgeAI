/**
 * QuickActions
 * ============
 * Top-level action buttons for common tasks:
 *   - New training run (opens a form or links to docs)
 *   - Evaluate checkpoint
 *   - Pull a redistributed model
 *   - Open chat
 */

"use client";

import Link from "next/link";

export function QuickActions() {
  const actions = [
    {
      label: "New run",
      description: "Open hardware page with CLI quick-start commands",
      href: "/dashboard/hardware",
      icon: "▶",
    },
    {
      label: "Evaluate",
      description: "Run benchmarks on a checkpoint",
      href: "/dashboard/eval",
      icon: "📊",
    },
    {
      label: "Pull model",
      description: "Use CLI pull + converted checkpoint in local chat",
      href: "/dashboard/chat",
      icon: "⬇",
    },
    {
      label: "Chat",
      description: "Chat with a local model via mlx-lm",
      href: "/dashboard/chat",
      icon: "💬",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {actions.map((action) => (
        <Link
          key={action.label}
          href={action.href}
          className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-4 hover:border-zinc-600 hover:bg-zinc-800/80 transition-all group"
        >
          <div className="text-xl mb-2">{action.icon}</div>
          <p className="text-sm font-medium text-zinc-200 group-hover:text-white transition-colors">
            {action.label}
          </p>
          <p className="text-xs text-zinc-500 mt-0.5">{action.description}</p>
        </Link>
      ))}
    </div>
  );
}
