"use client";

/**
 * Stock Research Terminal — a Fey-inspired, fully interactive research surface.
 * All state is client-side (useState): selected ticker, price interval,
 * ledger tab, summary expansion, hover position, and a simulated live price.
 * Every number, row, chart, and sentence is derived from that state.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import {
  useMarketData,
  useTickerSearch,
  type TerminalSnapshot,
} from "@/lib/market-store";

/* -------------------------------------------------------------------------- */
/*                                   Types                                    */
/* -------------------------------------------------------------------------- */

type IntervalKey = "1D" | "1W" | "1M" | "1Y";
type LedgerTab = "insiders" | "filings";

type PricePoint = { p: number; label: string };

type InsiderTrade = {
  id: string;
  name: string;
  role: string;
  action: "Buy" | "Sell" | "Option Exercise";
  shares: number;
  price: number;
  date: string;
};

type SecFiling = {
  id: string;
  form: string;
  title: string;
  period: string;
  date: string;
};

type Summary = {
  headline: string;
  bullets: string[];
  more: string[];
};

type Holding = {
  shares: number;
  avgCost: number;
  account: string;
};

type Stock = {
  ticker: string;
  company: string;
  exchange: string;
  sector: string;
  price: number;
  volume: number;
  marketCap: number;
  peRatio: number;
  intervalChange: Record<IntervalKey, number>;
  series: Record<IntervalKey, PricePoint[]>;
  insiders: InsiderTrade[];
  filings: SecFiling[];
  earnings: Summary;
  news: Summary;
  holding: Holding;
};

/* -------------------------------------------------------------------------- */
/*                          Live market data adapter                          */
/* -------------------------------------------------------------------------- */

/**
 * PHASE 7 — the deterministic data engine is GONE.
 *
 * What used to live here: a seeded PRNG, a synthetic series generator, and a
 * 240-line `STOCKS` array carrying invented prices, insider trades, SEC
 * filings and earnings copy for five hardcoded tickers. Every number this
 * terminal displayed was manufactured at module load.
 *
 * What replaces it: `useLiveStock`, which assembles the exact same `Stock`
 * render contract from real sources —
 *
 *   • quote, profile and daily candles   → /api/stocks/[symbol]
 *   • continuously-updating live price   → the shared market store
 *   • earnings + news digest             → /api/chat (Gemini, grounded in
 *                                          the fetched fundamentals)
 *
 * Insider trades and SEC filings have NO source in the current key set:
 * neither Gemini nor the equity provider's free tier exposes Form 4 or
 * EDGAR data. Those panels therefore render an explicit unavailable state
 * rather than fabricated rows. Fake filings in a research terminal are not
 * a placeholder, they are misinformation — and SEC EDGAR's submissions API
 * is free and keyless when you want to wire them for real.
 */

/** Timeframe switcher order. */
const INTERVALS: IntervalKey[] = ["1D", "1W", "1M", "1Y"];

/** Formats an ISO date as the compact label the chart axis expects. */
function candleLabel(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  return `${MONTH_LABELS[m - 1]} ${d}`;
}

const MONTH_LABELS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** Slice depth per interval, in daily candles. */
const INTERVAL_DEPTH: Record<IntervalKey, number> = {
  "1D": 2,
  "1W": 7,
  "1M": 22,
  "1Y": 252,
};

type LiveState = {
  stock: Stock | null;
  loading: boolean;
  error: string | null;
  /** True while the AI digest is still being generated. */
  summarising: boolean;
};

/**
 * Loads one symbol's full research payload and shapes it into `Stock`.
 * Re-runs whenever the symbol changes; the AI digest is requested only
 * after the fundamentals land, so the panels fill progressively instead of
 * blocking on the slowest dependency.
 */
function useLiveStock(symbol: string): LiveState {
  const [snapshot, setSnapshot] = useState<TerminalSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [digest, setDigest] = useState<{
    earnings: Summary;
    news: Summary;
  } | null>(null);
  const [summarising, setSummarising] = useState(false);

  /* ---- Fundamentals ---- */
  useEffect(() => {
    const s = symbol.trim().toUpperCase();
    if (!s) return;

    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setDigest(null);

    void (async () => {
      try {
        const res = await fetch(`/api/stocks/${encodeURIComponent(s)}?range=1Y`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!res.ok) {
          setError(
            res.status === 404
              ? `No market data available for ${s}.`
              : "The market feed is unreachable right now.",
          );
          setSnapshot(null);
          return;
        }
        setSnapshot((await res.json()) as TerminalSnapshot);
      } catch {
        if (!controller.signal.aborted) {
          setError("The market feed is unreachable right now.");
          setSnapshot(null);
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();

    return () => controller.abort();
  }, [symbol]);

  /* ---- AI digest, grounded in the fetched fundamentals ---- */
  useEffect(() => {
    if (!snapshot) return;
    const controller = new AbortController();
    setSummarising(true);

    void (async () => {
      try {
        const profile = snapshot.profile;
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            question: `Write a research digest for ${snapshot.symbol}. Give exactly 3 bullets on the business and its most recent reported fundamentals, then the literal line "---", then exactly 3 bullets on what currently matters for the stock. Ground every claim in the data supplied; if something is not in the data, say it is not available rather than guessing.`,
            context: {
              activeView: "stocks",
              equities: [
                {
                  symbol: snapshot.symbol,
                  shares: 0,
                  price: snapshot.quote.price,
                  dayPct: snapshot.quote.dayPct,
                },
              ],
              researchSubject: {
                symbol: snapshot.symbol,
                name: profile?.name ?? snapshot.symbol,
                exchange: profile?.exchange ?? "",
                sector: profile?.sector ?? "",
                industry: profile?.industry ?? "",
                description: profile?.description ?? "",
                marketCap: profile?.marketCap ?? 0,
                peRatio: profile?.peRatio ?? 0,
                week52High: profile?.week52High ?? 0,
                week52Low: profile?.week52Low ?? 0,
                price: snapshot.quote.price,
                dayPct: snapshot.quote.dayPct,
                open: snapshot.quote.open,
                high: snapshot.quote.high,
                low: snapshot.quote.low,
                volume: snapshot.quote.volume,
                rangeChangePct: snapshot.rangeStats?.changePct ?? 0,
              },
            },
          }),
        });

        const data = (await res.json().catch(() => null)) as {
          reply?: string;
        } | null;
        const reply = data?.reply ?? "";
        if (!reply) {
          setDigest(null);
          return;
        }

        /* Split on the requested separator; fall back to halving the
           bullet list if the model ignored it. */
        const [head, tail] = reply.includes("---")
          ? reply.split("---")
          : [reply, ""];
        const toBullets = (block: string) =>
          block
            .split("\n")
            .map((l) => l.replace(/^[-*•]\s*/, "").trim())
            .filter((l) => l.length > 0);

        const headBullets = toBullets(head);
        const tailBullets = toBullets(tail);
        const fallbackSplit = Math.ceil(headBullets.length / 2);

        setDigest({
          earnings: {
            headline: "Fundamentals",
            bullets: (tailBullets.length > 0
              ? headBullets
              : headBullets.slice(0, fallbackSplit)
            ).slice(0, 3),
            more: [],
          },
          news: {
            headline: "What matters now",
            bullets: (tailBullets.length > 0
              ? tailBullets
              : headBullets.slice(fallbackSplit)
            ).slice(0, 3),
            more: [],
          },
        });
      } catch {
        if (!controller.signal.aborted) setDigest(null);
      } finally {
        if (!controller.signal.aborted) setSummarising(false);
      }
    })();

    return () => controller.abort();
  }, [snapshot]);

  /* ---- Shape the render contract ---- */
  const stock = useMemo<Stock | null>(() => {
    if (!snapshot) return null;
    const { quote, profile, candles } = snapshot;

    const seriesFor = (key: IntervalKey): PricePoint[] => {
      const slice = candles.slice(-INTERVAL_DEPTH[key]);
      if (slice.length >= 2) {
        return slice.map((c) => ({ p: c.close, label: candleLabel(c.date) }));
      }
      /* Too few candles for this window (a brand-new listing, or a budget
         -exhausted candle fetch). Two points from the live quote keep the
         chart mathematically valid rather than dividing by zero. */
      return [
        { p: quote.previousClose || quote.open || quote.price, label: "Open" },
        { p: quote.price, label: "Now" },
      ];
    };

    const series = {
      "1D": seriesFor("1D"),
      "1W": seriesFor("1W"),
      "1M": seriesFor("1M"),
      "1Y": seriesFor("1Y"),
    } as Record<IntervalKey, PricePoint[]>;

    const changeFor = (key: IntervalKey): number => {
      const pts = series[key];
      const first = pts[0]?.p ?? 0;
      const last = pts[pts.length - 1]?.p ?? 0;
      return first > 0 ? ((last - first) / first) * 100 : 0;
    };

    return {
      ticker: quote.symbol,
      company: profile?.name ?? quote.name ?? quote.symbol,
      exchange: profile?.exchange ?? "—",
      sector: profile?.sector ?? "—",
      price: quote.price,
      volume: quote.volume,
      marketCap: profile?.marketCap ?? 0,
      peRatio: profile?.peRatio ?? 0,
      intervalChange: {
        /* The session move is authoritative from the quote itself; longer
           windows are derived from the candle series. */
        "1D": quote.dayPct,
        "1W": changeFor("1W"),
        "1M": changeFor("1M"),
        "1Y": changeFor("1Y"),
      },
      series,
      /* No free source — see the note at the top of this block. */
      insiders: [],
      filings: [],
      earnings:
        digest?.earnings ?? { headline: "Fundamentals", bullets: [], more: [] },
      news:
        digest?.news ?? { headline: "What matters now", bullets: [], more: [] },
      /* Positions live in the dashboard's book, not here. The terminal
         reports zero rather than inventing a holding. */
      holding: { shares: 0, avgCost: 0, account: "" },
    };
  }, [snapshot, digest]);

  return { stock, loading, error, summarising };
}


const MONO = "font-[family-name:var(--font-geist-mono)]";

function usd(n: number, digits = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function compact(n: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(n);
}

function pct(n: number, digits = 2): string {
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}%`;
}

function seriesRange(pts: PricePoint[]): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const pt of pts) {
    if (pt.p < lo) lo = pt.p;
    if (pt.p > hi) hi = pt.p;
  }
  return { lo, hi };
}

/* -------------------------------------------------------------------------- */
/*                              Atomic components                             */
/* -------------------------------------------------------------------------- */

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-text-muted">
        {label}
      </p>
      <p className={`mt-1 text-sm tabular-nums text-foreground ${MONO}`}>
        {value}
      </p>
    </div>
  );
}

/** Renders text with `**highlighted**` metrics in crisp foreground weight. */
function RichText({ text }: { text: string }) {
  const parts = text.split(/\*\*(.+?)\*\*/g);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <span
            key={i}
            className="font-medium tabular-nums text-foreground"
          >
            {part}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */
/*                                Screener                                    */
/* -------------------------------------------------------------------------- */

/**
 * PHASE 7 — live full-market screener.
 *
 * The old implementation filtered a five-element hardcoded array with
 * `Array.prototype.filter`, so typing anything outside those five tickers
 * returned "No matches" no matter how valid the symbol was. This one is
 * wired to `useTickerSearch`, which debounces the query and hits the
 * provider's symbol-search endpoint — the entire listed market is now
 * reachable from this input.
 *
 * Presentation follows the "Find what you seek" command popup: a floating
 * results overlay anchored to the field, keyboard-driven (↑ ↓ to move,
 * return to select, escape to dismiss), with an exchange chip on each row.
 * Selecting a row loads it into the shared price store, which is what makes
 * the chart, the AI panel and the bottom marquee update from one identical
 * number in the same frame.
 */
function Screener({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (ticker: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLElement>(null);

  const { matches, searching } = useTickerSearch(query);
  const { all: liveQuotes } = useMarketData();

  /* Symbols already streaming in the shared store — shown before the user
     types anything, so the popup is never empty on first open. */
  const tracked = useMemo(
    () =>
      Object.values(liveQuotes)
        .filter((q) => /^[A-Z.\-]{1,6}$/.test(q.symbol))
        .sort((a, b) => a.symbol.localeCompare(b.symbol)),
    [liveQuotes],
  );

  const showingSearch = query.trim().length > 0;

  const rows = useMemo(
    () =>
      showingSearch
        ? matches.map((m) => ({
            symbol: m.symbol,
            name: m.name,
            venue: m.region,
          }))
        : tracked.map((q) => ({
            symbol: q.symbol,
            name: q.name,
            venue: "Tracked",
          })),
    [showingSearch, matches, tracked],
  );

  /* Reset the highlight whenever the result set changes underneath it —
     otherwise a stale index can point past the end of a shorter list. */
  useEffect(() => setCursor(0), [rows.length, showingSearch]);

  /* "/" focuses the screener from anywhere on the page. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  /* Dismiss on outside pointer-down — the popup floats over content now,
     so a blur-only dismissal would leave it stranded on a stray click. */
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent | TouchEvent) => {
      const el = containerRef.current;
      if (el && !el.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("touchstart", onDown);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("touchstart", onDown);
    };
  }, [open]);

  const select = useCallback(
    (ticker: string) => {
      onSelect(ticker);
      setQuery("");
      setOpen(false);
      inputRef.current?.blur();
    },
    [onSelect],
  );

  return (
    <section
      ref={containerRef}
      aria-label="Ticker screener"
      className="relative rounded-xl border border-border-muted bg-card p-4"
    >
      <div className="flex items-center gap-2.5">
        <svg
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
          className="size-4 shrink-0 text-text-muted"
        >
          <circle cx="7" cy="7" r="4.5" stroke="currentColor" />
          <path
            d="M10.5 10.5 14 14"
            stroke="currentColor"
            strokeLinecap="round"
          />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setOpen(true);
              setCursor((c) => Math.min(c + 1, Math.max(rows.length - 1, 0)));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setCursor((c) => Math.max(c - 1, 0));
            } else if (e.key === "Enter") {
              e.preventDefault();
              const row = rows[cursor];
              if (row) select(row.symbol);
              /* No result yet but a plausible symbol typed — try it
                 directly. The research route validates it server-side and
                 answers 404 cleanly if it isn't real. */
              else if (/^[A-Za-z.\-]{1,10}$/.test(query.trim())) {
                select(query.trim().toUpperCase());
              }
            } else if (e.key === "Escape") {
              setQuery("");
              setOpen(false);
              inputRef.current?.blur();
            }
          }}
          placeholder="Search any stock…"
          aria-label="Search tickers"
          aria-expanded={open}
          aria-controls="screener-results"
          role="combobox"
          aria-autocomplete="list"
          className="w-full bg-transparent text-sm text-foreground placeholder:text-text-muted focus:outline-none"
        />
        <kbd
          className={`hidden rounded border border-border-muted px-1.5 py-0.5 text-[10px] text-text-muted sm:block ${MONO}`}
        >
          /
        </kbd>
      </div>

      {open && (
        <div
          id="screener-results"
          className="terminal-overlay-card absolute inset-x-0 top-full z-40 mt-2 overflow-hidden rounded-xl border border-border-muted bg-card shadow-2xl"
        >
          <div className="flex items-center justify-between border-b border-border-muted px-3 py-2.5">
            <span className="rounded-full border border-border-muted px-2 py-0.5 text-[10px] uppercase tracking-wider text-text-muted">
              {showingSearch ? "Full market" : "Tracked"}
            </span>
            <span className="flex items-center gap-1.5 text-[10px] text-text-muted">
              {searching ? (
                <>
                  <span className="anim-pulse-soft size-1 rounded-full bg-accent-green" />
                  Searching…
                </>
              ) : (
                <>
                  Search any stock and hit
                  <kbd
                    className={`rounded border border-border-muted px-1.5 py-0.5 text-[10px] text-text-muted ${MONO}`}
                  >
                    return
                  </kbd>
                </>
              )}
            </span>
          </div>

          <ul className="max-h-[320px] overflow-y-auto fey-scroll">
            {rows.map((row, i) => {
              const quote = liveQuotes[row.symbol];
              const isCursor = i === cursor;
              const isSelected = row.symbol === selected;
              return (
                <li key={`${row.symbol}-${i}`}>
                  <button
                    type="button"
                    onMouseEnter={() => setCursor(i)}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      select(row.symbol);
                    }}
                    aria-pressed={isSelected}
                    className={`flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left transition-colors focus-visible:outline-none ${
                      isCursor ? "bg-foreground/[0.06]" : "hover:bg-foreground/[0.03]"
                    }`}
                  >
                    <span className="flex min-w-0 items-center gap-2.5">
                      <span
                        aria-hidden
                        className="flex size-6 shrink-0 items-center justify-center rounded-full border border-border-muted bg-background text-[10px] font-bold text-foreground/80"
                      >
                        {row.symbol.charAt(0)}
                      </span>
                      <span
                        className={`text-sm font-semibold ${MONO} ${
                          isSelected ? "text-foreground" : "text-foreground/90"
                        }`}
                      >
                        {row.symbol}
                      </span>
                      <span className="truncate text-xs text-text-muted">
                        {row.name}
                      </span>
                    </span>
                    <span className="flex shrink-0 items-center gap-3">
                      {/* Only rows already in the shared store carry a price.
                          A search hit that has never been quoted shows its
                          venue instead of a fabricated number. */}
                      {quote ? (
                        <>
                          <span
                            className={`text-xs tabular-nums text-foreground/90 ${MONO}`}
                          >
                            {usd(quote.price)}
                          </span>
                          <span
                            className={`w-14 text-right text-xs tabular-nums ${MONO} ${
                              quote.dayPct >= 0
                                ? "text-accent-green"
                                : "text-foreground/70"
                            }`}
                          >
                            {pct(quote.dayPct)}
                          </span>
                        </>
                      ) : (
                        <span className="rounded border border-border-muted px-1.5 py-0.5 text-[10px] text-text-muted">
                          {row.venue}
                        </span>
                      )}
                    </span>
                  </button>
                </li>
              );
            })}

            {rows.length === 0 && (
              <li className="px-3 py-5 text-center text-xs text-text-muted">
                {searching
                  ? "Searching the full market…"
                  : showingSearch
                    ? "No listed security matches that query."
                    : "Start typing a ticker or company name."}
              </li>
            )}
          </ul>
        </div>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/*                                Price chart                                 */
/* -------------------------------------------------------------------------- */

const CHART_W = 600;
const CHART_H = 220;
const CHART_PAD = 12;

function PriceChart({
  pts,
  live,
  label,
}: {
  pts: PricePoint[];
  live: number;
  label: string;
}) {
  const [hover, setHover] = useState<number | null>(null);

  /* The tail of the series follows the live ticking price. */
  const series = useMemo(() => {
    const s = pts.slice();
    s[s.length - 1] = { ...s[s.length - 1], p: live };
    return s;
  }, [pts, live]);

  const { lo, hi } = useMemo(() => seriesRange(series), [series]);
  const span = hi - lo || 1;

  const xAt = (i: number) =>
    CHART_PAD + (i / (series.length - 1)) * (CHART_W - CHART_PAD * 2);
  const yAt = (p: number) =>
    CHART_PAD + (1 - (p - lo) / span) * (CHART_H - CHART_PAD * 2);
  const xPct = (i: number) => (xAt(i) / CHART_W) * 100;
  const yPct = (p: number) => (yAt(p) / CHART_H) * 100;

  const linePath = series
    .map(
      (pt, i) =>
        `${i === 0 ? "M" : "L"}${xAt(i).toFixed(2)},${yAt(pt.p).toFixed(2)}`,
    )
    .join(" ");
  const areaPath = `${linePath} L${xAt(series.length - 1).toFixed(2)},${
    CHART_H - CHART_PAD
  } L${xAt(0).toFixed(2)},${CHART_H - CHART_PAD} Z`;

  const lastP = series[series.length - 1].p;
  const openP = series[0].p;

  const hoverIdx = hover != null ? Math.min(hover, series.length - 1) : null;
  const hoverPt = hoverIdx != null ? series[hoverIdx] : null;
  const tooltipShift =
    hoverIdx == null
      ? ""
      : hoverIdx < series.length * 0.15
        ? "translate-x-2"
        : hoverIdx > series.length * 0.85
          ? "-translate-x-full"
          : "-translate-x-1/2";

  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    const idx = Math.round(ratio * (series.length - 1));
    setHover(Math.max(0, Math.min(series.length - 1, idx)));
  };

  return (
    <div className="grid-lines relative mt-5 h-56 w-full select-none rounded-lg">
      <svg
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${label} price chart for the selected timeframe`}
        className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="chart-fade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f4f4f7" stopOpacity="0.14" />
            <stop offset="100%" stopColor="#f4f4f7" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#chart-fade)" className="anim-fade-up" />
        <path
          d={linePath}
          fill="none"
          stroke="#f4f4f7"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
          className="anim-draw"
        />
        <line
          x1={CHART_PAD}
          x2={CHART_W - CHART_PAD}
          y1={yAt(lastP)}
          y2={yAt(lastP)}
          stroke="#f4f4f7"
          strokeOpacity="0.25"
          strokeDasharray="3 5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      {/* Current price tag pinned to the dashed line */}
      <div
        className={`pointer-events-none absolute right-0 z-10 -translate-y-1/2 rounded bg-foreground px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-background ${MONO}`}
        style={{ top: `${yPct(lastP)}%` }}
      >
        {usd(lastP)}
      </div>

      {/* Window high / low markers */}
      <span
        className={`pointer-events-none absolute left-0 top-1 text-[10px] tabular-nums text-text-muted ${MONO}`}
      >
        {usd(hi)}
      </span>
      <span
        className={`pointer-events-none absolute bottom-1 left-0 text-[10px] tabular-nums text-text-muted ${MONO}`}
      >
        {usd(lo)}
      </span>

      {/* Hover crosshair, marker dot, and tooltip */}
      {hoverPt && hoverIdx != null && (
        <>
          <div
            className="pointer-events-none absolute w-px bg-foreground/25"
            style={{
              left: `${xPct(hoverIdx)}%`,
              top: `${(CHART_PAD / CHART_H) * 100}%`,
              bottom: `${(CHART_PAD / CHART_H) * 100}%`,
            }}
          />
          <div
            className="pointer-events-none absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground ring-2 ring-background"
            style={{
              left: `${xPct(hoverIdx)}%`,
              top: `${yPct(hoverPt.p)}%`,
            }}
          />
          <div
            className={`pointer-events-none absolute z-10 -translate-y-full rounded-md border border-border-muted bg-card px-2.5 py-1.5 shadow-xl ${tooltipShift}`}
            style={{
              left: `${xPct(hoverIdx)}%`,
              top: `calc(${yPct(hoverPt.p)}% - 10px)`,
            }}
          >
            <p
              className={`text-xs font-medium tabular-nums text-foreground ${MONO}`}
            >
              ${usd(hoverPt.p)}
            </p>
            <p
              className={`text-[10px] tabular-nums text-text-muted ${MONO}`}
            >
              {hoverPt.label} ·{" "}
              {pct(((hoverPt.p - openP) / openP) * 100)}
            </p>
          </div>
        </>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*                               Price engine                                 */
/* -------------------------------------------------------------------------- */

function PriceEngine({
  stock,
  timeframe,
  onTimeframe,
  live,
}: {
  stock: Stock;
  timeframe: IntervalKey;
  onTimeframe: (tf: IntervalKey) => void;
  live: number;
}) {
  const series = stock.series[timeframe];
  const openP = series[0].p;
  const changeAbs = live - openP;
  const changePct = (changeAbs / openP) * 100;
  const positive = changePct >= 0;

  const { lo: winLo, hi: winHi } = useMemo(() => seriesRange(series), [series]);
  const range52 = useMemo(() => seriesRange(stock.series["1Y"]), [stock]);
  const pos52 = Math.max(
    0,
    Math.min(
      100,
      ((live - range52.lo) / (range52.hi - range52.lo || 1)) * 100,
    ),
  );

  return (
    <section
      aria-label="Price engine"
      className="rounded-xl border border-border-muted bg-card p-4 sm:p-5"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2.5">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              {stock.ticker}
            </h2>
            <span className="rounded border border-border-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-text-muted">
              {stock.exchange}
            </span>
            <span className="hidden text-xs text-text-muted sm:inline">
              {stock.sector}
            </span>
          </div>
          <p className="mt-1 text-xs text-text-muted">
            {stock.company} · Mkt Cap ${compact(stock.marketCap)} · P/E{" "}
            {stock.peRatio.toFixed(1)}
          </p>
        </div>

        <div
          className="flex items-center gap-0.5 rounded-lg border border-border-muted p-0.5"
          aria-label="Price timeframe"
        >
          {INTERVALS.map((tf) => (
            <button
              key={tf}
              type="button"
              onClick={() => onTimeframe(tf)}
              aria-pressed={tf === timeframe}
              className={`rounded-md px-2.5 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                tf === timeframe
                  ? "bg-foreground/10 text-foreground"
                  : "text-text-muted hover:text-foreground"
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-end gap-x-4 gap-y-2">
        <p className="text-4xl font-semibold tabular-nums tracking-tight text-foreground">
          ${usd(live)}
        </p>
        <p
          className={`pb-1 text-sm font-medium tabular-nums ${
            positive ? "text-accent-green" : "text-foreground"
          }`}
        >
          {positive ? "+" : "−"}${usd(Math.abs(changeAbs))} ({pct(changePct)})
        </p>
        <span className="flex items-center gap-1.5 pb-1.5 text-[10px] uppercase tracking-wider text-text-muted">
          <span className="anim-pulse-soft size-1 rounded-full bg-accent-green" />
          Real-time
        </span>
      </div>

      <PriceChart
        key={`${stock.ticker}-${timeframe}`}
        pts={series}
        live={live}
        label={stock.ticker}
      />

      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-border-muted pt-4 sm:grid-cols-4">
        <Stat label="Open" value={`$${usd(openP)}`} />
        <Stat label="High" value={`$${usd(winHi)}`} />
        <Stat label="Low" value={`$${usd(winLo)}`} />
        {/* Real session volume from the quote. The old code multiplied it
            by a per-interval fudge factor to fake a longer-window figure;
            the provider reports session volume only, so that is what is
            shown, labelled honestly. */}
        <Stat label="Volume" value={compact(stock.volume)} />
      </div>

      <div className="mt-4 flex items-center gap-3 border-t border-border-muted pt-4">
        <span
          className={`text-[10px] uppercase tracking-wider text-text-muted ${MONO}`}
        >
          52W
        </span>
        <span
          className={`text-xs tabular-nums text-text-muted ${MONO}`}
        >
          ${usd(range52.lo, 0)}
        </span>
        <div className="relative h-1 flex-1 rounded-full bg-border-muted">
          <div
            className="absolute size-2.5 -translate-x-1/2 -translate-y-[3px] rounded-full bg-foreground transition-[left] duration-500"
            style={{ left: `${pos52}%` }}
          />
        </div>
        <span
          className={`text-xs tabular-nums text-text-muted ${MONO}`}
        >
          ${usd(range52.hi, 0)}
        </span>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/*                            Fundamental ledger                              */
/* -------------------------------------------------------------------------- */

const LEDGER_TABS: { key: LedgerTab; label: string }[] = [
  { key: "insiders", label: "Insider Trades" },
  { key: "filings", label: "SEC Filings" },
];

function actionBadgeCls(action: InsiderTrade["action"]): string {
  if (action === "Buy")
    return "border-accent-green/30 bg-accent-green/10 text-accent-green";
  if (action === "Sell")
    return "border-border-muted bg-foreground/5 text-foreground";
  return "border-border-muted text-text-muted";
}

function Ledger({
  stock,
  tab,
  onTab,
}: {
  stock: Stock;
  tab: LedgerTab;
  onTab: (t: LedgerTab) => void;
}) {
  const meta =
    tab === "insiders" ? "FORM 4 · EDGAR FEED" : "EDGAR · FULL-TEXT SEARCH";

  return (
    <section
      aria-label="Fundamental and SEC ledger"
      className="rounded-xl border border-border-muted bg-card"
    >
      <div className="flex items-center justify-between border-b border-border-muted px-4">
        <div className="flex items-center gap-5" aria-label="Ledger tabs">
          {LEDGER_TABS.map((t) => {
            const active = t.key === tab;
            const count =
              t.key === "insiders"
                ? stock.insiders.length
                : stock.filings.length;
            return (
              <button
                key={t.key}
                type="button"
                onClick={() => onTab(t.key)}
                aria-pressed={active}
                className={`relative px-1 pb-3 pt-4 text-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                  active ? "text-foreground" : "text-text-muted hover:text-foreground"
                }`}
              >
                {t.label}
                <span
                  className={`ml-1.5 text-[10px] tabular-nums ${MONO} ${
                    active ? "text-foreground/70" : "text-text-muted"
                  }`}
                >
                  {count}
                </span>
                {active && (
                  <span className="absolute inset-x-0 -bottom-px h-px bg-foreground" />
                )}
              </button>
            );
          })}
        </div>
        <span
          className={`hidden pb-3 text-[10px] uppercase tracking-wider text-text-muted sm:block ${MONO}`}
        >
          {meta}
        </span>
      </div>

      {/* PHASE 7 — insider trades and SEC filings have no source in the
          current key set (neither Gemini nor the equity provider's free
          tier exposes Form 4 or EDGAR data), so both tabs render an
          explicit unavailable state. Fabricated filings in a research
          terminal are not a placeholder, they are misinformation. SEC
          EDGAR's submissions API is free and keyless when you want these
          panels for real. */}
      {(tab === "insiders" ? stock.insiders : stock.filings).length === 0 ? (
        <div className="px-4 py-8 text-center">
          <p className="text-sm text-foreground">
            {tab === "insiders" ? "Insider activity" : "SEC filings"} not
            available
          </p>
          <p className="mx-auto mt-1.5 max-w-sm text-xs leading-relaxed text-text-muted">
            No connected data source provides{" "}
            {tab === "insiders" ? "Form 4 transactions" : "EDGAR filings"} on
            the current plan. Wire SEC EDGAR (free, no key required) to
            populate this panel.
          </p>
        </div>
      ) : tab === "insiders" ? (
        <ul>
          {stock.insiders.map((t) => (
            <li
              key={t.id}
              className="flex items-center justify-between gap-4 px-4 py-3.5 transition-colors hover:bg-foreground/[0.03]"
            >
              <div className="min-w-0">
                <p className="truncate text-sm text-foreground">{t.name}</p>
                <p className="mt-0.5 truncate text-xs text-text-muted">
                  {t.role}
                </p>
              </div>
              <div className="hidden shrink-0 items-center gap-3 md:flex">
                <span
                  className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wider ${actionBadgeCls(t.action)}`}
                >
                  {t.action}
                </span>
                <span
                  className={`text-xs tabular-nums text-text-muted ${MONO}`}
                >
                  {t.shares.toLocaleString()} sh @ ${usd(t.price)}
                </span>
              </div>
              <div className="shrink-0 text-right">
                <p
                  className={`text-sm font-medium tabular-nums text-foreground ${MONO}`}
                >
                  ${compact(t.shares * t.price)}
                </p>
                <p className="mt-0.5 text-xs text-text-muted">{t.date}</p>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <ul>
          {stock.filings.map((f) => (
            <li
              key={f.id}
              className="flex items-center justify-between gap-4 px-4 py-3.5 transition-colors hover:bg-foreground/[0.03]"
            >
              <div className="flex min-w-0 items-center gap-3">
                <span
                  className={`shrink-0 rounded border border-border-muted px-1.5 py-0.5 text-[10px] font-medium text-foreground/80 ${MONO}`}
                >
                  {f.form}
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm text-foreground">{f.title}</p>
                  <p className="mt-0.5 truncate text-xs text-text-muted">
                    {f.period}
                  </p>
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2 text-right">
                <span className="text-xs text-text-muted">{f.date}</span>
                <span
                  aria-hidden="true"
                  className="text-xs text-text-muted transition-colors group-hover:text-foreground"
                >
                  ↗
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/*                              AI summaries                                  */
/* -------------------------------------------------------------------------- */

function SummaryBlock({
  title,
  tag,
  summary,
  accent,
}: {
  title: string;
  tag: string;
  summary: Summary;
  accent?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const bullets = expanded
    ? [...summary.bullets, ...summary.more]
    : summary.bullets;
  const canExpand = summary.more.length > 0;

  return (
    <div className="rounded-xl border border-border-muted bg-card p-4">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-medium text-foreground">{title}</h3>
        <span className={`text-[10px] uppercase tracking-wider text-text-muted ${MONO}`}>
          {tag}
        </span>
      </div>
      <ul className="mt-3 space-y-2.5">
        {bullets.map((b, i) => (
          <li key={i} className="flex gap-2.5">
            <span
              aria-hidden="true"
              className={`mt-[7px] size-1 shrink-0 rounded-full ${
                accent ? "bg-accent-green" : "bg-text-muted"
              }`}
            />
            <p className="text-[13px] leading-relaxed text-text-muted">
              <RichText text={b} />
              {canExpand && !expanded && i === bullets.length - 1 && (
                <>
                  {" "}
                  <button
                    type="button"
                    onClick={() => setExpanded(true)}
                    className="text-accent-green transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                  >
                    Read more →
                  </button>
                </>
              )}
            </p>
          </li>
        ))}
      </ul>
      {expanded && (
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="mt-3 text-xs text-text-muted transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
        >
          Show less
        </button>
      )}
    </div>
  );
}

/**
 * PHASE 7 — the earnings and news blocks used to read from a hand-written
 * `Summary` object baked into the STOCKS array, so every ticker showed the
 * same invented copy forever. They now render the Gemini digest generated
 * in `useLiveStock`, grounded strictly in the fetched quote and profile.
 * While that round trip is in flight the blocks show a shimmer; if it fails
 * they say so rather than falling back to fiction.
 */
function AiSummaries({
  stock,
  summarising,
}: {
  stock: Stock;
  summarising: boolean;
}) {
  return (
    <section aria-label="AI research summaries" className="flex flex-col gap-3">
      <div className="flex items-center justify-between px-1">
        <h2
          className={`text-[10px] uppercase tracking-[0.14em] text-text-muted ${MONO}`}
        >
          AI Research
        </h2>
        <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-text-muted">
          <span
            className={`size-1 rounded-full ${
              summarising ? "anim-pulse-soft bg-accent-green" : "bg-text-muted"
            }`}
          />
          {summarising ? "Generating" : "Gemini"}
        </span>
      </div>

      {summarising ? (
        <>
          <SummarySkeleton title="Earnings Summary" />
          <SummarySkeleton title="News Digest" />
        </>
      ) : stock.earnings.bullets.length > 0 || stock.news.bullets.length > 0 ? (
        <>
          {stock.earnings.bullets.length > 0 && (
            <SummaryBlock
              key={`${stock.ticker}-earnings`}
              title="Earnings Summary"
              tag={stock.earnings.headline}
              summary={stock.earnings}
              accent
            />
          )}
          {stock.news.bullets.length > 0 && (
            <SummaryBlock
              key={`${stock.ticker}-news`}
              title="News Digest"
              tag={stock.news.headline}
              summary={stock.news}
            />
          )}
        </>
      ) : (
        /* The digest failed or returned nothing usable. Saying so beats
           rendering an empty card that looks like a loading state which
           never resolves. */
        <div className="rounded-xl border border-border-muted bg-card p-4">
          <p className="text-sm text-foreground">Digest unavailable</p>
          <p className="mt-1.5 text-xs leading-relaxed text-text-muted">
            The advisor engine did not return a summary for {stock.ticker}.
            The quote and price history above are unaffected.
          </p>
        </div>
      )}
    </section>
  );
}

/** Shimmer placeholder matching a summary card's geometry. */
function SummarySkeleton({ title }: { title: string }) {
  return (
    <div className="rounded-xl border border-border-muted bg-card p-4">
      <h3 className="text-sm font-medium text-foreground">{title}</h3>
      <div className="mt-3 flex flex-col gap-2.5">
        {["92%", "78%", "85%"].map((w, i) => (
          <span
            key={i}
            className="fey-skeleton-bar block h-2 rounded-full"
            style={{ width: w, animationDelay: `${i * 140}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*                             Portfolio impact                               */
/* -------------------------------------------------------------------------- */

function PortfolioImpact({
  stock,
  timeframe,
}: {
  stock: Stock;
  timeframe: IntervalKey;
}) {
  /* PHASE 7 — the hardcoded $128,400 portfolio denominator is gone.
     Positions live in the dashboard's book, which this terminal cannot
     reach, so `holding` always reports zero here and the panel renders its
     honest "no position" state. Wiring real weights means lifting the
     position book into the shared market store alongside prices — a
     deliberate next step, not something to paper over with a constant. */
  const { shares, avgCost, account } = stock.holding;
  const held = shares > 0;

  const positionValue = shares * stock.price;
  const weightPct = 0;
  const deltaPct = 0;
  const deltaPositive = deltaPct >= 0;

  return (
    <section
      aria-label="Portfolio impact"
      className="rounded-xl border border-border-muted bg-card p-4"
    >
      <div className="flex items-center justify-between gap-3">
        <h2
          className={`text-[10px] uppercase tracking-[0.14em] text-text-muted ${MONO}`}
        >
          Portfolio Impact
        </h2>
        <span className="rounded border border-border-muted px-1.5 py-0.5 text-[10px] text-text-muted">
          {account}
        </span>
      </div>

      {held ? (
        <>
          <p className="mt-3 text-[13px] leading-relaxed text-text-muted">
            You hold{" "}
            <span className="font-medium tabular-nums text-foreground">
              {shares.toLocaleString()} shares
            </span>{" "}
            of {stock.ticker} in your{" "}
            <span className="font-medium text-foreground">{account}</span>{" "}
            account. Over the selected{" "}
            <span className="font-medium text-foreground">{timeframe}</span>{" "}
            window, this position directly alters your Invest Portfolio weight
            by{" "}
            <span
              className={`font-medium tabular-nums ${
                deltaPositive ? "text-accent-green" : "text-foreground"
              }`}
            >
              {pct(deltaPct, 1)}
            </span>
            .
          </p>

          <div className="mt-4 grid grid-cols-3 gap-3 border-t border-border-muted pt-4">
            <Stat label="Position" value={`$${compact(positionValue)}`} />
            <Stat label="Avg Cost" value={`$${usd(avgCost)}`} />
            <Stat label="Weight" value={`${weightPct.toFixed(1)}%`} />
          </div>

          <div
            className="mt-3 h-1 overflow-hidden rounded-full bg-border-muted"
            role="img"
            aria-label={`Portfolio weight ${weightPct.toFixed(1)} percent`}
          >
            <div
              className="h-full rounded-full bg-accent-green/70 transition-[width] duration-500"
              style={{ width: `${Math.min(100, weightPct)}%` }}
            />
          </div>
        </>
      ) : (
        <p className="mt-3 text-[13px] leading-relaxed text-text-muted">
          No current position in {stock.ticker}. Add one from your {account}{" "}
          account and its Invest Portfolio weight impact will appear here in
          real time.
        </p>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/*                                Terminal shell                              */
/* -------------------------------------------------------------------------- */

export default function StockResearchTerminal({
  /**
   * "page" — standalone full-screen surface (default; used by /search).
   * "modal" — fitted to the root layout's floating terminal card.
   */
  variant = "page",
  /** Rendered as a ✕ button in the header when provided (layout overlay). */
  onClose,
}: {
  variant?: "page" | "modal";
  onClose?: () => void;
}) {
  const [ticker, setTicker] = useState("NVDA");
  const [timeframe, setTimeframe] = useState<IntervalKey>("1D");
  const [ledgerTab, setLedgerTab] = useState<LedgerTab>("insiders");
  const [now, setNow] = useState<Date | null>(null);

  /* ============ PHASE 7 — live data, one shared price dictionary =========
     `useLiveStock` assembles the research payload from /api/stocks/[symbol]
     and /api/chat. The SESSION PRICE, though, comes from the shared market
     store, not from that REST snapshot — the snapshot is a point-in-time
     read, while the store is continuously repolled and is the same object
     the dashboard cards and the bottom marquee render from.

     That distinction is the whole anti-drift mechanism: if this component
     displayed its own separately-fetched price, it would disagree with the
     marquee by a few cents the moment either refreshed on its own clock.
     Reading the store at render time means every surface commits the
     identical number in the identical frame. The simulated mean-reverting
     tape that used to live here — random-walking a fake price every 1.6
     seconds — is gone entirely. */
  const { stock, loading, error, summarising } = useLiveStock(ticker);
  const { all: liveQuotes, trackSymbols } = useMarketData();

  /* Fold the viewed symbol into the shared polling universe so it keeps
     ticking for as long as it is on screen. */
  useEffect(() => {
    const s = ticker.trim().toUpperCase();
    if (s) trackSymbols([s]);
  }, [ticker, trackSymbols]);

  /* Terminal clock. First read is deferred one tick to avoid hydration
     drift (a server-rendered `new Date()` disagrees with the client). */
  useEffect(() => {
    const tick = () => setNow(new Date());
    const first = window.setTimeout(tick, 0);
    const id = window.setInterval(tick, 1000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(id);
    };
  }, []);

  const liveQuote = stock ? liveQuotes[stock.ticker] : undefined;
  /* Store price wins; the snapshot price is the fallback for the first
     frame after a symbol change, before the store has quoted it. */
  const live = liveQuote?.price ?? stock?.price ?? 0;

  /* PHASE 7 — the top-level `series` / `changePct` pair that used to live
     here fed ONLY into <PortfolioImpact>'s now-removed weight math (the
     hardcoded $128,400 portfolio denominator). PriceEngine computes its own
     `series`/`changePct` internally from `stock` and `live`, so nothing
     downstream needs a second copy at this scope. */
  const isModal = variant === "modal";

  /* Left panel — 60%: screener, price engine, fundamental ledger.

     The screener renders unconditionally: it is how a user recovers from a
     failed load, so gating it behind `stock` would strand them on an error
     with no way to search for something else. Everything downstream of it
     waits for real data rather than rendering a skeleton of zeros. */
  const leftPanel = (
    <div className="flex flex-col gap-4 lg:col-span-3">
      <div className="anim-fade-up">
        <Screener selected={ticker} onSelect={setTicker} />
      </div>

      {stock ? (
        <>
          <div key={`engine-${stock.ticker}`} className="anim-fade-up">
            <PriceEngine
              stock={stock}
              timeframe={timeframe}
              onTimeframe={setTimeframe}
              live={live}
            />
          </div>
          <div key={`ledger-${stock.ticker}`} className="anim-fade-up">
            <Ledger stock={stock} tab={ledgerTab} onTab={setLedgerTab} />
          </div>
        </>
      ) : (
        <div className="anim-fade-up rounded-xl border border-border-muted bg-card p-8 text-center">
          {loading ? (
            <>
              <span className="anim-pulse-soft mx-auto block size-1.5 rounded-full bg-accent-green" />
              <p className="mt-3 text-sm text-foreground">
                Loading {ticker.toUpperCase()}
              </p>
              <p className="mt-1 text-xs text-text-muted">
                Pulling quote, fundamentals and price history…
              </p>
            </>
          ) : (
            <>
              <p className="text-sm text-foreground">
                {error ?? `No data for ${ticker.toUpperCase()}.`}
              </p>
              <p className="mx-auto mt-2 max-w-sm text-xs leading-relaxed text-text-muted">
                Free equity tiers are hard-capped on daily requests. If this
                persists, the quota is likely spent rather than the symbol
                being wrong — cached symbols keep working either way.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );

  /* Right panel — 40%: AI summaries and portfolio impact. */
  const rightPanel = (
    <aside className="flex flex-col gap-4 lg:col-span-2">
      {stock ? (
        <>
          <div key={`ai-${stock.ticker}`} className="anim-fade-up">
            <AiSummaries stock={stock} summarising={summarising} />
          </div>
          <div key={`impact-${stock.ticker}`} className="anim-fade-up">
            <PortfolioImpact stock={stock} timeframe={timeframe} />
          </div>
        </>
      ) : null}
    </aside>
  );

  return (
    <div
      className={
        isModal
          ? "flex h-full min-h-0 flex-col bg-card font-sans text-foreground"
          : "min-h-screen bg-background font-sans text-foreground"
      }
    >
      <style>{`
        @keyframes terminal-draw {
          from { stroke-dashoffset: 2000; }
          to { stroke-dashoffset: 0; }
        }
        @keyframes terminal-fade-up {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes terminal-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
        .anim-draw {
          stroke-dasharray: 2000;
          animation: terminal-draw 1.1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
        .anim-fade-up { animation: terminal-fade-up 0.35s ease both; }
        .anim-pulse-soft { animation: terminal-pulse 2.2s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) {
          .anim-draw, .anim-fade-up, .anim-pulse-soft { animation: none; }
          .anim-draw { stroke-dasharray: none; }
        }
      `}</style>

      <div
        className={
          isModal
            ? "flex min-h-0 flex-1 flex-col gap-4 p-4 sm:p-6"
            : "mx-auto flex min-h-screen max-w-[1400px] flex-col gap-4 p-4 sm:p-6"
        }
      >
        <header className="flex shrink-0 items-center justify-between gap-4">
          <div className="flex min-w-0 items-baseline gap-3">
            <h1 className="text-lg font-semibold tracking-tight">Research</h1>
            <p className="truncate text-xs text-text-muted">
              Terminal /{" "}
              <span className="text-foreground">
                {stock?.ticker ?? ticker.toUpperCase()}
              </span>
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-text-muted">
            <span className="flex items-center gap-1.5">
              <span className="anim-pulse-soft size-1.5 rounded-full bg-accent-green" />
              LIVE
            </span>
            <span className={`hidden tabular-nums sm:block ${MONO}`}>
              {now
                ? now.toLocaleTimeString("en-US", { hour12: false })
                : "--:--:--"}
            </span>
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                aria-label="Close terminal"
                className="rounded-md border border-border-muted p-1.5 text-text-muted transition-colors hover:bg-foreground/5 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
              >
                <svg
                  viewBox="0 0 12 12"
                  fill="none"
                  aria-hidden="true"
                  className="size-3"
                >
                  <path
                    d="M2.5 2.5l7 7m0-7l-7 7"
                    stroke="currentColor"
                    strokeWidth="1.25"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            )}
          </div>
        </header>

        {isModal ? (
          /* Modal: the header stays pinned while the research surface
             scrolls inside the floating card. */
          <main className="min-h-0 flex-1 overflow-y-auto">
            <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-5">
              {leftPanel}
              {rightPanel}
            </div>
          </main>
        ) : (
          <main className="grid flex-1 grid-cols-1 items-start gap-4 lg:grid-cols-5">
            {leftPanel}
            {rightPanel}
          </main>
        )}
      </div>
    </div>
  );
}