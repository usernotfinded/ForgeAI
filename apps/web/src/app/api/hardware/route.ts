/**
 * ForgeAI — Hardware API Route
 *
 * Proxies hardware info from forge-engine.
 */

import { NextResponse } from "next/server";

const ENGINE_URL = process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${ENGINE_URL}/hardware`, { cache: "no-store" });

    if (!res.ok) {
      return NextResponse.json(
        { error: `Engine returned ${res.status}` },
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
