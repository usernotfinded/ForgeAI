/**
 * ForgeAI — Training Status API Route
 *
 * Proxies training log data from forge-engine for the dashboard.
 * Reads the JSONL log file via the forge-engine /training/logs endpoint.
 */

import { NextRequest, NextResponse } from "next/server";

const ENGINE_URL = process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const logFile = searchParams.get("log_file") ?? "./checkpoints/run/train_log.jsonl";
  const lastN = searchParams.get("last_n") ?? "500";

  try {
    const res = await fetch(
      `${ENGINE_URL}/training/logs?log_file=${encodeURIComponent(logFile)}&last_n=${lastN}`,
      { cache: "no-store" }
    );

    if (!res.ok) {
      return NextResponse.json(
        { entries: [], error: `Engine returned ${res.status}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { entries: [], error: "forge-engine not reachable at " + ENGINE_URL },
      { status: 503 }
    );
  }
}
