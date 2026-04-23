/**
 * ChatInterface
 * =============
 * ChatGPT-style conversation UI for chatting with a local ForgeAI model.
 *
 * Features:
 *   - Checkpoint and tokenizer path selection
 *   - Message history with user/model roles
 *   - Auto-scroll to latest message
 *   - Loading state during inference
 *   - Temperature / max tokens controls
 */

"use client";

import { useState, useRef, useEffect, FormEvent } from "react";

interface Message {
  role: "user" | "model";
  content: string;
}

export function ChatInterface() {
  // Model config
  const [checkpointDir, setCheckpointDir] = useState("./checkpoints/run");
  const [tokenizerDir, setTokenizerDir] = useState("./tokenizers/my-tokenizer");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(200);
  const [showSettings, setShowSettings] = useState(true);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    setError(null);

    const userMessage: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);

    setIsLoading(true);

    try {
      const conversation = messages.map((m) => m.content);

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_dir: checkpointDir,
          tokenizer_dir: tokenizerDir,
          message: text,
          conversation,
          temperature,
          max_tokens: maxTokens,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error ?? `Request failed (${res.status})`);
        setIsLoading(false);
        return;
      }

      const modelMessage: Message = {
        role: "model",
        content: data.response ?? "(empty response)",
      };
      setMessages((prev) => [...prev, modelMessage]);
    } catch {
      setError("Could not reach the server. Is forge-engine running?");
    } finally {
      setIsLoading(false);
    }
  }

  function handleClear() {
    setMessages([]);
    setError(null);
  }

  return (
    <div className="flex flex-col flex-1 h-full">
      {/* Settings panel */}
      {showSettings && (
        <div className="border-b border-zinc-800 px-6 py-4 bg-zinc-900/50 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-widest">
              Model Settings
            </h3>
            <button
              onClick={() => setShowSettings(false)}
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Hide
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-2xs text-zinc-500 block mb-1">
                Checkpoint directory
              </label>
              <input
                type="text"
                value={checkpointDir}
                onChange={(e) => setCheckpointDir(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-200 font-mono focus:outline-none focus:border-zinc-500"
                placeholder="./checkpoints/run"
              />
            </div>
            <div>
              <label className="text-2xs text-zinc-500 block mb-1">
                Tokenizer directory
              </label>
              <input
                type="text"
                value={tokenizerDir}
                onChange={(e) => setTokenizerDir(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-200 font-mono focus:outline-none focus:border-zinc-500"
                placeholder="./tokenizers/my-tokenizer"
              />
            </div>
            <div>
              <label className="text-2xs text-zinc-500 block mb-1">
                Temperature ({temperature})
              </label>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-brand-500"
              />
            </div>
            <div>
              <label className="text-2xs text-zinc-500 block mb-1">
                Max tokens ({maxTokens})
              </label>
              <input
                type="range"
                min="50"
                max="1000"
                step="50"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full accent-brand-500"
              />
            </div>
          </div>
        </div>
      )}

      {!showSettings && (
        <div className="border-b border-zinc-800 px-6 py-2 flex items-center justify-between bg-zinc-900/30">
          <span className="text-xs text-zinc-500 font-mono truncate">
            {checkpointDir}
          </span>
          <button
            onClick={() => setShowSettings(true)}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Settings
          </button>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && !error && (
          <div className="text-center py-16">
            <p className="text-zinc-500 text-sm">
              Start a conversation with your local model.
            </p>
            <p className="text-zinc-600 text-xs mt-1">
              Make sure forge-engine is running:{" "}
              <code className="font-mono bg-zinc-800 px-1.5 py-0.5 rounded text-zinc-400">
                uvicorn app.main:app --port 8000
              </code>
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-3 text-sm ${
                msg.role === "user"
                  ? "bg-brand-600 text-white"
                  : "bg-zinc-800 text-zinc-200 border border-zinc-700"
              }`}
            >
              <p className="text-2xs font-medium mb-1 opacity-60">
                {msg.role === "user" ? "You" : "Model"}
              </p>
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-sm">
              <p className="text-2xs font-medium mb-1 text-zinc-500">Model</p>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-950/50 border border-red-900 rounded-lg px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="border-t border-zinc-800 px-6 py-4 bg-zinc-900/30">
        <form onSubmit={handleSend} className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            disabled={isLoading}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-brand-600 hover:bg-brand-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            {isLoading ? "..." : "Send"}
          </button>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={handleClear}
              className="text-zinc-500 hover:text-zinc-300 px-3 py-2.5 text-sm transition-colors"
            >
              Clear
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
