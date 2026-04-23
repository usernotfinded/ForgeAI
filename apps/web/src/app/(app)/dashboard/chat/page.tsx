/**
 * ForgeAI — Chat Page
 *
 * ChatGPT-style interface for chatting with a local ForgeAI model.
 * The user selects a checkpoint directory and tokenizer, then exchanges
 * messages. The backend (forge-engine) handles inference.
 */

import { ChatInterface } from "@/components/dashboard/ChatInterface";

export const metadata = {
  title: "Chat — ForgeAI",
};

export default function ChatPage() {
  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
            ForgeAI
          </h1>
          <p className="text-xs text-zinc-500">Chat with a local model</p>
        </div>
        <nav className="flex gap-6 text-sm text-zinc-400">
          <a href="/dashboard" className="hover:text-zinc-200 transition-colors">Runs</a>
          <a href="/dashboard/eval" className="hover:text-zinc-200 transition-colors">Evaluate</a>
          <a href="/dashboard/chat" className="text-zinc-100 font-medium">Chat</a>
          <a href="/dashboard/hardware" className="hover:text-zinc-200 transition-colors">Hardware</a>
        </nav>
      </header>

      {/* Chat interface fills remaining space */}
      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto">
        <ChatInterface />
      </main>
    </div>
  );
}
