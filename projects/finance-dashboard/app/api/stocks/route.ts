import { NextResponse } from "next/server";

import {
  DEFAULT_UNIVERSE,
  budgetRemaining,
  getQuotes,
  searchSymbols,
} from "@/lib/stock-provider";

/**
 * GET /api/stocks — batch quotes for the marquee, the portfolio cards and
 * anything else that needs many symbols at once.
 *
 *   /api/stocks                        → the default tracked universe
 *   /api/stocks?symbols=NVDA,TSLA,SHOP → an explicit set (max 12)
 *   /api/stocks?search=nvid            → ticker lookup for the ⌘K terminal
 *
 * The response shape matches the dashboard's existing `StocksFeedResponse`
 * type exactly, so the client swap is a drop-in.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Hard ceiling — an unbounded symbol list is a free-tier quota bomb. */
const MAX_SYMBOLS = 12;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  /* ---- Ticker search branch (powers the terminal's filter input) ------- */
  const search = searchParams.get("search");
  if (search !== null) {
    const hit = await searchSymbols(search);
    return NextResponse.json(
      {
        matches: hit?.value ?? [],
        source: hit?.source ?? "cache",
        updatedAt: new Date().toISOString(),
        upstreamBudgetRemaining: budgetRemaining(),
      },
      { headers: { "cache-control": "no-store" } },
    );
  }

  /* ---- Batch quote branch --------------------------------------------- */
  const requested = searchParams.get("symbols");
  const symbols = (
    requested
      ? requested
          .split(",")
          .map((s) => s.trim().toUpperCase())
          .filter(Boolean)
      : [...DEFAULT_UNIVERSE]
  )
    /* De-duplicate before slicing so a repeated symbol can't crowd out a
       distinct one and silently drop it from the response. */
    .filter((s, i, arr) => arr.indexOf(s) === i)
    .slice(0, MAX_SYMBOLS);

  if (symbols.length === 0) {
    return NextResponse.json(
      {
        quotes: [],
        source: "cache",
        updatedAt: new Date().toISOString(),
        upstreamBudgetRemaining: budgetRemaining(),
      },
      { headers: { "cache-control": "no-store" } },
    );
  }

  try {
    const { quotes, source } = await getQuotes(symbols);

    return NextResponse.json(
      {
        quotes: quotes.map((q) => ({
          symbol: q.symbol,
          name: q.name,
          price: q.price,
          dayPct: q.dayPct,
          change: q.change,
          open: q.open,
          high: q.high,
          low: q.low,
          previousClose: q.previousClose,
          volume: q.volume,
        })),
        source: source === "live" ? "alphavantage" : "cache",
        updatedAt: new Date().toISOString(),
        upstreamBudgetRemaining: budgetRemaining(),
      },
      { headers: { "cache-control": "no-store" } },
    );
  } catch (err) {
    console.error("[api/stocks] batch quote failed", err);
    return NextResponse.json(
      {
        quotes: [],
        source: "cache",
        updatedAt: new Date().toISOString(),
        upstreamBudgetRemaining: budgetRemaining(),
      },
      { status: 200, headers: { "cache-control": "no-store" } },
    );
  }
}