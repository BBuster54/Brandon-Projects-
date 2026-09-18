import { NextResponse } from "next/server";

/**
 * GET /api/crypto — live BTC / ETH / SOL prices and 24h moves.
 *
 * COINGECKO_API_KEY is read server-side and sent as the demo-tier header
 * `x-cg-demo-api-key`. The browser only ever talks to this local route, so
 * the key is never exposed and CoinGecko never sees a browser origin.
 *
 * Response shape is deliberately identical to what the dashboard's existing
 * `CryptoFeedResponse` type already expects:
 *   { assets: [{ id, symbol, name, usd, changePct24h }], source, updatedAt }
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/* Demo keys use the public base; a Pro key requires pro-api.coingecko.com
   AND the x-cg-pro-api-key header instead. Set COINGECKO_TIER=pro to swap. */
const IS_PRO = process.env.COINGECKO_TIER === "pro";
const API_BASE = IS_PRO
  ? "https://pro-api.coingecko.com/api/v3"
  : "https://api.coingecko.com/api/v3";
const KEY_HEADER = IS_PRO ? "x-cg-pro-api-key" : "x-cg-demo-api-key";

/** CoinGecko ids → the ticker symbols the cockpit's crypto book uses. */
const TRACKED = [
  { id: "bitcoin", symbol: "BTC", name: "Bitcoin" },
  { id: "ethereum", symbol: "ETH", name: "Ethereum" },
  { id: "solana", symbol: "SOL", name: "Solana" },
] as const;

type CryptoFeedAsset = {
  id: string;
  symbol: string;
  name: string;
  usd: number;
  changePct24h: number | null;
};

/**
 * The client polls every 3 seconds, but the demo tier allows roughly 30
 * calls/minute across ALL routes. This module-scope cache collapses that
 * polling into at most one upstream call per TTL window, no matter how many
 * tabs or intervals are running. 10s keeps the ticker feeling live while
 * using ~6 calls/minute.
 */
const CACHE_TTL_MS = Number(process.env.CRYPTO_CACHE_TTL_MS ?? 10_000);

let cached: { assets: CryptoFeedAsset[]; at: number } | null = null;

function finite(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export async function GET() {
  const now = Date.now();

  /* Fresh cache — serve without touching the upstream quota. */
  if (cached && now - cached.at < CACHE_TTL_MS) {
    return NextResponse.json(
      {
        assets: cached.assets,
        source: "cache",
        updatedAt: new Date(cached.at).toISOString(),
      },
      { headers: { "cache-control": "no-store" } },
    );
  }

  const apiKey = process.env.COINGECKO_API_KEY;

  const url = new URL(`${API_BASE}/simple/price`);
  url.searchParams.set("ids", TRACKED.map((t) => t.id).join(","));
  url.searchParams.set("vs_currencies", "usd");
  url.searchParams.set("include_24hr_change", "true");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);

  try {
    const res = await fetch(url.toString(), {
      signal: controller.signal,
      cache: "no-store",
      headers: {
        accept: "application/json",
        /* Demo keys are optional on the public base but lift the rate
           limit substantially — send it whenever it is configured. */
        ...(apiKey ? { [KEY_HEADER]: apiKey } : {}),
      },
    });

    if (!res.ok) throw new Error(`coingecko ${res.status}`);

    const payload = (await res.json()) as Record<
      string,
      { usd?: number; usd_24h_change?: number }
    >;

    const assets: CryptoFeedAsset[] = TRACKED.map((t) => {
      const row = payload[t.id] ?? {};
      return {
        id: t.id,
        symbol: t.symbol,
        name: t.name,
        usd: finite(row.usd) ?? 0,
        changePct24h: finite(row.usd_24h_change),
      };
    }).filter((a) => a.usd > 0);

    if (assets.length === 0) throw new Error("coingecko returned no prices");

    cached = { assets, at: now };

    return NextResponse.json(
      { assets, source: "coingecko", updatedAt: new Date(now).toISOString() },
      { headers: { "cache-control": "no-store" } },
    );
  } catch (err) {
    console.error("[api/crypto] upstream failed", err);

    /* Stale cache beats a blank ticker — the next tick retries anyway. */
    if (cached) {
      return NextResponse.json(
        {
          assets: cached.assets,
          source: "cache",
          updatedAt: new Date(cached.at).toISOString(),
        },
        { headers: { "cache-control": "no-store" } },
      );
    }

    return NextResponse.json(
      { assets: [], source: "cache", updatedAt: new Date(now).toISOString() },
      { status: 200, headers: { "cache-control": "no-store" } },
    );
  } finally {
    clearTimeout(timer);
  }
}