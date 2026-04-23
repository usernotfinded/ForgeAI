/**
 * ForgeAI — Chat API Route
 *
 * Proxies chat requests to forge-engine /chat endpoint.
 * The Next.js server acts as a relay so the browser only talks to :3000.
 */

import { NextRequest, NextResponse } from "next/server";

const ENGINE_URL = process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const res = await fetch(`${ENGINE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => null);
      return NextResponse.json(
        { error: errorData?.detail ?? `Engine returned ${res.status}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "forge-engine not reachable at " + ENGINE_URL },
      { status: 503 }
    );
  }
}
