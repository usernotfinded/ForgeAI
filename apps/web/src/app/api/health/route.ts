/**
 * ForgeAI — Frontend Health Check API Route
 * Used by Docker and Kubernetes for liveness/readiness probes.
 */

import { NextResponse } from 'next/server';

export async function GET() {
    return NextResponse.json({
        status: 'healthy',
        service: 'web',
        timestamp: new Date().toISOString(),
    });
}
