import "server-only";

/**
 * Server-only equity data adapter.
 *
 * WALL_ST_API_KEY is consumed exclusively here — this module is imported by
 * route handlers, never by a client component, and `server-only` makes that
 * a build-time error rather than a silent key leak.
 *
 * ── PROVIDER NOTE ────────────────────────────────────────────────────────
 * There is no public vendor named "Wall St API", so this adapter defaults to
 * Alpha Vantage (matching the provider named in the dashboard's own Phase 4
 * comments). Every vendor-specific detail is isolated in the ADAPTER block
 * below, so pointing this at Finnhub / Twelve Data / Polygon means editing
 * one object — the routes, the cache, the quota guard and the entire client
 * never change, because they only ever see the normalized shapes.
 *
 * Override without touching code:
 *   STOCK_API_BASE=https://finnhub.io/api/v1
 *   STOCK_API_DAILY_BUDGET=800
 */

/* -------------------------------------------------------------------------- */
/*                             Normalized shapes                              */
/* -------------------------------------------------------------------------- */

/** One live quote — the only equity shape the client ever receives. */
export type NormalizedQuote = {
  symbol: string;
  name: string;
  price: number;
  /** Absolute change on the session, in dollars. */
  change: number;
  /** Percentage change on the session. */
  dayPct: number;
  open: number;
  high: number;
  low: number;
  previousClose: number;
  volume: number;
};

/** Company reference data for the research terminal header. */
export type NormalizedProfile = {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  industry: string;
  description: string;
  marketCap: number;
  peRatio: number;
  week52High: number;
  week52Low: number;
};

/** One point on the historical series powering the terminal's SVG path. */
export type NormalizedCandle = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

/** One row in the ticker-search dropdown. */
export type NormalizedMatch = {
  symbol: string;
  name: string;
  region: string;
  currency: string;
};

export type ProviderSource = "live" | "cache";

/* -------------------------------------------------------------------------- */
/*                          Configuration + guardrails                        */
/* -------------------------------------------------------------------------- */

const API_BASE = process.env.STOCK_API_BASE ?? "https://www.alphavantage.co/query";

/**
 * Free equity tiers are brutally rate-limited (Alpha Vantage: 25 requests /
 * day). Two defences keep a 3-second UI polling loop from incinerating the
 * quota within a minute of the first page load:
 *
 *   1. A TTL cache in front of every upstream call, so N browser tabs and N
 *      polling intervals collapse into at most one upstream request per
 *      symbol per window.
 *   2. A hard daily budget counter. When it's spent, routes keep serving
 *      the last good cached payload instead of erroring — a stale price is
 *      strictly better than a blank dashboard.
 *
 * Both live in module scope, which on Vercel means per-warm-instance rather
 * than globally shared. That's adequate for a single-user cockpit; a
 * multi-tenant deployment should move these to Upstash/Redis.
 */
const DAILY_BUDGET = Number(process.env.STOCK_API_DAILY_BUDGET ?? 25);

/** Quote TTL. 60s is well inside free-tier limits for a handful of symbols. */
const QUOTE_TTL_MS = 60_000;
/** Profiles and candles move far more slowly than quotes. */
const PROFILE_TTL_MS = 12 * 60 * 60 * 1000;
const CANDLE_TTL_MS = 30 * 60 * 1000;
const SEARCH_TTL_MS = 24 * 60 * 60 * 1000;

type CacheEntry<T> = { value: T; expiresAt: number };

const cache = new Map<string, CacheEntry<unknown>>();

/** Requests spent today, reset on the first call of a new UTC day. */
let spentToday = 0;
let budgetDay = new Date().toISOString().slice(0, 10);

function rollBudgetDay(): void {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== budgetDay) {
    budgetDay = today;
    spentToday = 0;
  }
}

export function budgetRemaining(): number {
  rollBudgetDay();
  return Math.max(DAILY_BUDGET - spentToday, 0);
}

function readCache<T>(key: string): { value: T; fresh: boolean } | null {
  const hit = cache.get(key) as CacheEntry<T> | undefined;
  if (!hit) return null;
  return { value: hit.value, fresh: Date.now() < hit.expiresAt };
}

function writeCache<T>(key: string, value: T, ttlMs: number): void {
  cache.set(key, { value, expiresAt: Date.now() + ttlMs });
}

/**
 * Single choke point for every upstream request: serves fresh cache without
 * spending budget, spends budget only on a genuine miss, and falls back to
 * stale cache when the budget is gone or the upstream call fails.
 */
async function throughCache<T>(
  key: string,
  ttlMs: number,
  fetcher: () => Promise<T>,
): Promise<{ value: T; source: ProviderSource } | null> {
  const cached = readCache<T>(key);
  if (cached?.fresh) return { value: cached.value, source: "cache" };

  rollBudgetDay();
  if (spentToday >= DAILY_BUDGET) {
    /* Budget exhausted — stale beats nothing. */
    return cached ? { value: cached.value, source: "cache" } : null;
  }

  try {
    spentToday += 1;
    const value = await fetcher();
    writeCache(key, value, ttlMs);
    return { value, source: "live" };
  } catch {
    return cached ? { value: cached.value, source: "cache" } : null;
  }
}

/** Timeout-guarded JSON GET — a hung upstream must never hang the route. */
async function getJson(url: string, timeoutMs = 8000): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      cache: "no-store",
      headers: { accept: "application/json" },
    });
    if (!res.ok) throw new Error(`upstream ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function num(v: unknown): number {
  const n = typeof v === "string" ? Number(v.replace(/[%,$]/g, "")) : Number(v);
  return Number.isFinite(n) ? n : 0;
}

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" && v.length > 0 ? v : fallback;
}

/* -------------------------------------------------------------------------- */
/*                    ADAPTER — the only vendor-aware code                    */
/* -------------------------------------------------------------------------- */

function requireKey(): string {
  const key = process.env.WALL_ST_API_KEY;
  if (!key) throw new Error("WALL_ST_API_KEY is not configured");
  return key;
}

function endpoint(params: Record<string, string>): string {
  const url = new URL(API_BASE);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  url.searchParams.set("apikey", requireKey());
  return url.toString();
}

/**
 * Alpha Vantage answers rate-limit violations with HTTP 200 and a `Note` /
 * `Information` body. Treating that as a success would poison the cache with
 * an empty quote, so it is promoted to a thrown error here — which sends
 * `throughCache` down the stale-cache path instead.
 */
function assertNotThrottled(payload: unknown): void {
  if (payload && typeof payload === "object") {
    const p = payload as Record<string, unknown>;
    if (p.Note || p.Information || p["Error Message"]) {
      throw new Error(
        str(p.Note ?? p.Information ?? p["Error Message"], "upstream refused"),
      );
    }
  }
}

async function fetchQuoteUpstream(symbol: string): Promise<NormalizedQuote> {
  const payload = await getJson(
    endpoint({ function: "GLOBAL_QUOTE", symbol }),
  );
  assertNotThrottled(payload);
  const q =
    (payload as Record<string, Record<string, unknown>>)["Global Quote"] ?? {};

  const price = num(q["05. price"]);
  if (!(price > 0)) throw new Error(`no quote for ${symbol}`);

  return {
    symbol: str(q["01. symbol"], symbol).toUpperCase(),
    name: str(q["01. symbol"], symbol).toUpperCase(),
    price,
    change: num(q["09. change"]),
    dayPct: num(q["10. change percent"]),
    open: num(q["02. open"]),
    high: num(q["03. high"]),
    low: num(q["04. low"]),
    previousClose: num(q["08. previous close"]),
    volume: num(q["06. volume"]),
  };
}

async function fetchProfileUpstream(symbol: string): Promise<NormalizedProfile> {
  const payload = await getJson(endpoint({ function: "OVERVIEW", symbol }));
  assertNotThrottled(payload);
  const p = (payload ?? {}) as Record<string, unknown>;
  if (!str(p.Symbol)) throw new Error(`no profile for ${symbol}`);

  return {
    symbol: str(p.Symbol, symbol).toUpperCase(),
    name: str(p.Name, symbol),
    exchange: str(p.Exchange, "—"),
    sector: str(p.Sector, "—"),
    industry: str(p.Industry, "—"),
    description: str(p.Description),
    marketCap: num(p.MarketCapitalization),
    peRatio: num(p.PERatio),
    week52High: num(p["52WeekHigh"]),
    week52Low: num(p["52WeekLow"]),
  };
}

async function fetchCandlesUpstream(symbol: string): Promise<NormalizedCandle[]> {
  const payload = await getJson(
    endpoint({
      function: "TIME_SERIES_DAILY",
      symbol,
      outputsize: "compact",
    }),
  );
  assertNotThrottled(payload);
  const series =
    (payload as Record<string, Record<string, Record<string, unknown>>>)[
      "Time Series (Daily)"
    ] ?? {};

  const candles = Object.entries(series)
    .map(([date, row]) => ({
      date,
      open: num(row["1. open"]),
      high: num(row["2. high"]),
      low: num(row["3. low"]),
      close: num(row["4. close"]),
      volume: num(row["5. volume"]),
    }))
    .filter((c) => c.close > 0)
    /* Upstream returns newest-first; charts want oldest-first. */
    .sort((a, b) => a.date.localeCompare(b.date));

  if (candles.length === 0) throw new Error(`no history for ${symbol}`);
  return candles;
}

async function searchUpstream(query: string): Promise<NormalizedMatch[]> {
  const payload = await getJson(
    endpoint({ function: "SYMBOL_SEARCH", keywords: query }),
  );
  assertNotThrottled(payload);
  const matches =
    (payload as Record<string, Record<string, unknown>[]>).bestMatches ?? [];

  return matches
    .map((m) => ({
      symbol: str(m["1. symbol"]).toUpperCase(),
      name: str(m["2. name"]),
      region: str(m["4. region"], "—"),
      currency: str(m["8. currency"], "USD"),
    }))
    .filter((m) => m.symbol.length > 0)
    .slice(0, 10);
}

/* -------------------------------------------------------------------------- */
/*                              Public interface                              */
/* -------------------------------------------------------------------------- */

export async function getQuote(
  symbol: string,
): Promise<{ value: NormalizedQuote; source: ProviderSource } | null> {
  const s = symbol.trim().toUpperCase();
  if (!s) return null;
  return throughCache(`quote:${s}`, QUOTE_TTL_MS, () => fetchQuoteUpstream(s));
}

/**
 * Batch quotes. Requests are issued sequentially rather than with
 * Promise.all: free tiers throttle on burst concurrency, and a cached symbol
 * costs nothing anyway, so the serial path is both safer and usually
 * instantaneous.
 */
export async function getQuotes(symbols: string[]): Promise<{
  quotes: NormalizedQuote[];
  source: ProviderSource;
}> {
  const out: NormalizedQuote[] = [];
  let anyLive = false;

  for (const symbol of symbols) {
    const hit = await getQuote(symbol);
    if (!hit) continue;
    if (hit.source === "live") anyLive = true;
    out.push(hit.value);
  }

  return { quotes: out, source: anyLive ? "live" : "cache" };
}

export async function getProfile(
  symbol: string,
): Promise<{ value: NormalizedProfile; source: ProviderSource } | null> {
  const s = symbol.trim().toUpperCase();
  if (!s) return null;
  return throughCache(`profile:${s}`, PROFILE_TTL_MS, () =>
    fetchProfileUpstream(s),
  );
}

export async function getCandles(
  symbol: string,
): Promise<{ value: NormalizedCandle[]; source: ProviderSource } | null> {
  const s = symbol.trim().toUpperCase();
  if (!s) return null;
  return throughCache(`candles:${s}`, CANDLE_TTL_MS, () =>
    fetchCandlesUpstream(s),
  );
}

export async function searchSymbols(
  query: string,
): Promise<{ value: NormalizedMatch[]; source: ProviderSource } | null> {
  const q = query.trim();
  if (q.length < 1) return { value: [], source: "cache" };
  return throughCache(`search:${q.toLowerCase()}`, SEARCH_TTL_MS, () =>
    searchUpstream(q),
  );
}

/** Default tracked universe — mirrored by the client's market store. */
export const DEFAULT_UNIVERSE = ["NVDA", "TSLA", "AAPL", "AMZN"] as const;