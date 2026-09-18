import { NextResponse } from "next/server";

import {
  budgetRemaining,
  getCandles,
  getProfile,
  getQuote,
} from "@/lib/stock-provider";

/**
 * GET /api/stocks/[symbol] — the full research payload for one equity.
 *
 * This is what the ⌘K Research Terminal calls when a ticker is selected. It
 * returns quote + company profile + historical candles in a single round
 * trip so the terminal never renders a half-populated panel.
 *
 *   /api/stocks/NVDA              → quote + profile + 100 daily candles
 *   /api/stocks/NVDA?fields=quote → quote only (cheap, for fast refresh)
 *
 * Every sub-fetch degrades independently: a missing profile or an exhausted
 * candle budget still returns the live quote rather than failing the panel.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Supported chart windows, expressed in trading days of daily candles. */
const RANGE_DAYS: Record<string, number> = {
  "1D": 2,
  "1W": 7,
  "1M": 22,
  "3M": 66,
  "1Y": 252,
};

export async function GET(
  request: Request,
  /* Next.js 15 hands route params in as a promise — it must be awaited. */
  { params }: { params: Promise<{ symbol: string }> },
) {
  const { symbol: raw } = await params;
  const symbol = decodeURIComponent(raw ?? "").trim().toUpperCase();

  if (!/^[A-Z0-9.\-]{1,10}$/.test(symbol)) {
    return NextResponse.json(
      { error: "invalid symbol" },
      { status: 400, headers: { "cache-control": "no-store" } },
    );
  }

  const { searchParams } = new URL(request.url);
  const fields = (searchParams.get("fields") ?? "all").toLowerCase();
  const range = (searchParams.get("range") ?? "1Y").toUpperCase();
  const wantProfile = fields === "all" || fields.includes("profile");
  const wantCandles = fields === "all" || fields.includes("candles");

  const quoteHit = await getQuote(symbol);

  if (!quoteHit) {
    return NextResponse.json(
      {
        error: "no data for symbol",
        symbol,
        upstreamBudgetRemaining: budgetRemaining(),
      },
      { status: 404, headers: { "cache-control": "no-store" } },
    );
  }

  /* Profile and candles are fetched in parallel — both are cache-guarded, so
     the common path costs zero upstream calls and the burst is bounded at
     two even on a cold symbol. */
  const [profileHit, candleHit] = await Promise.all([
    wantProfile ? getProfile(symbol) : Promise.resolve(null),
    wantCandles ? getCandles(symbol) : Promise.resolve(null),
  ]);

  const windowDays = RANGE_DAYS[range] ?? RANGE_DAYS["1Y"];
  const candles = candleHit ? candleHit.value.slice(-windowDays) : [];

  /* Derive the range's own high/low/change from the sliced window so the
     terminal's 52W bar and range stats stay internally consistent with the
     chart actually on screen. */
  const closes = candles.map((c) => c.close);
  const rangeStats =
    closes.length > 1
      ? {
          first: closes[0],
          last: closes[closes.length - 1],
          high: Math.max(...candles.map((c) => c.high)),
          low: Math.min(...candles.map((c) => c.low)),
          changePct:
            closes[0] > 0
              ? ((closes[closes.length - 1] - closes[0]) / closes[0]) * 100
              : 0,
        }
      : null;

  return NextResponse.json(
    {
      symbol,
      quote: quoteHit.value,
      profile: profileHit?.value ?? null,
      candles,
      range,
      rangeStats,
      source: quoteHit.source === "live" ? "alphavantage" : "cache",
      updatedAt: new Date().toISOString(),
      upstreamBudgetRemaining: budgetRemaining(),
    },
    { headers: { "cache-control": "no-store" } },
  );
}