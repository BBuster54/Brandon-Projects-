"use client";

/**
 * Centralized market price store — the single source of truth for every
 * live quote rendered anywhere in the cockpit.
 *
 * THE PROBLEM THIS SOLVES
 * Before this module, the dashboard ran its own 3s crypto loop and 30s
 * equity loop, the marquee ticker read prices out of dashboard state, and
 * the research terminal fetched independently. Three consumers, three clocks
 * — so NVDA could legitimately read $178.43 in the terminal and $178.50 in
 * the marquee at the same instant. That is the cent-drift.
 *
 * THE FIX
 * One provider owns one polling loop per asset class and writes into one
 * dictionary keyed by symbol. Every consumer reads that dictionary at RENDER
 * time — not by copying prices into local state through an effect, which
 * would reintroduce a one-render lag and therefore the drift. When the store
 * updates, React re-renders all consumers from the identical snapshot, so
 * within any given committed frame every surface shows the same number.
 *
 * MOUNTING
 * The provider must sit in `app/layout.tsx`, ABOVE both the dashboard and
 * the ⌘K terminal overlay. The terminal is rendered by the layout, not by
 * the dashboard, so a provider mounted inside the dashboard could not reach
 * it — and the drift would survive in exactly the place it is most visible.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

/* -------------------------------------------------------------------------- */
/*                                   Types                                    */
/* -------------------------------------------------------------------------- */

/** One normalized quote as held in the store. */
export type MarketQuote = {
  symbol: string;
  name: string;
  price: number;
  dayPct: number;
  /** Epoch ms of the tick that produced this price. */
  at: number;
};

/** Symbol → quote. Dictionary rather than array: consumers look up by key
    and never depend on ordering, so a provider reshuffle can't reorder UI. */
export type QuoteMap = Record<string, MarketQuote>;

type MarketDataValue = {
  /** Live equity quotes, keyed by uppercase ticker. */
  equities: QuoteMap;
  /** Live crypto quotes, keyed by uppercase token symbol. */
  crypto: QuoteMap;
  /** Every quote in one map — what the marquee consumes. */
  all: QuoteMap;
  /** True until the first successful payload of each class has landed. */
  isHydrating: boolean;
  /** Epoch ms of the most recent successful tick, or null before the first. */
  lastUpdated: number | null;
  /** Remaining upstream daily budget reported by `/api/stocks`. */
  equityBudgetRemaining: number | null;
  /**
   * Adds symbols to the polled universe — the research terminal calls this
   * when a user searches a ticker, so a newly-viewed symbol starts streaming
   * into the same shared dictionary as everything else.
   */
  trackSymbols: (symbols: string[]) => void;
  /** Forces an immediate refresh outside the normal cadence. */
  refresh: () => void;
};

const MarketDataContext = createContext<MarketDataValue | null>(null);

/* -------------------------------------------------------------------------- */
/*                                  Config                                    */
/* -------------------------------------------------------------------------- */

/** Baseline tracked universe — mirrors DEFAULT_UNIVERSE on the server. */
const BASE_SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMZN"];

/** Crypto is cheap and volatile — poll fast. */
const CRYPTO_INTERVAL_MS = 3_000;
/** Equities are quota-expensive — poll slow; the server caches on top. */
const EQUITY_INTERVAL_MS = 30_000;

/* -------------------------------------------------------------------------- */
/*                                  Provider                                  */
/* -------------------------------------------------------------------------- */

export function MarketDataProvider({
  children,
  /** Set false to suspend polling entirely (e.g. behind the auth gateway). */
  enabled = true,
}: {
  children: ReactNode;
  enabled?: boolean;
}) {
  const [equities, setEquities] = useState<QuoteMap>({});
  const [crypto, setCrypto] = useState<QuoteMap>({});
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [equityBudgetRemaining, setEquityBudgetRemaining] = useState<
    number | null
  >(null);
  const [cryptoReady, setCryptoReady] = useState(false);
  const [equitiesReady, setEquitiesReady] = useState(false);

  /* The tracked universe lives in state (so changes trigger a refetch) and
     in a ref (so the polling closure always reads the current set without
     tearing down and rebuilding its interval on every change). */
  const [symbols, setSymbols] = useState<string[]>(BASE_SYMBOLS);
  const symbolsRef = useRef<string[]>(BASE_SYMBOLS);
  useEffect(() => {
    symbolsRef.current = symbols;
  }, [symbols]);

  const trackSymbols = useCallback((incoming: string[]) => {
    const clean = incoming
      .map((s) => s.trim().toUpperCase())
      .filter((s) => /^[A-Z0-9.\-]{1,10}$/.test(s));
    if (clean.length === 0) return;

    setSymbols((prev) => {
      const next = [...prev];
      let changed = false;
      for (const s of clean) {
        if (!next.includes(s)) {
          next.push(s);
          changed = true;
        }
      }
      /* Returning the identical reference when nothing is new keeps this
         callable from a render-adjacent effect without looping. */
      if (!changed) return prev;
      /* Server caps at 12 symbols; keep the base universe and the most
         recently viewed tickers. */
      return next.length > 12
        ? [...BASE_SYMBOLS, ...next.filter((s) => !BASE_SYMBOLS.includes(s)).slice(-8)]
        : next;
    });
  }, []);

  /* ------------------------------ Crypto loop ------------------------------ */
  const fetchCrypto = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await fetch("/api/crypto", { cache: "no-store", signal });
      if (!res.ok) return;
      const feed = (await res.json()) as {
        assets?: {
          symbol: string;
          name: string;
          usd: number;
          changePct24h: number | null;
        }[];
      };
      if (!Array.isArray(feed.assets) || feed.assets.length === 0) return;

      const at = Date.now();
      setCrypto((prev) => {
        const next: QuoteMap = { ...prev };
        for (const a of feed.assets ?? []) {
          if (!(a.usd > 0)) continue;
          const key = a.symbol.toUpperCase();
          next[key] = {
            symbol: key,
            name: a.name,
            price: a.usd,
            dayPct:
              typeof a.changePct24h === "number" &&
              Number.isFinite(a.changePct24h)
                ? a.changePct24h
                : (prev[key]?.dayPct ?? 0),
            at,
          };
        }
        return next;
      });
      setLastUpdated(at);
      setCryptoReady(true);
    } catch {
      /* Network hiccup — last known prices stand; the next tick retries. */
    }
  }, []);

  /* ----------------------------- Equity loop ------------------------------ */
  const fetchEquities = useCallback(async (signal?: AbortSignal) => {
    try {
      const list = symbolsRef.current;
      if (list.length === 0) return;
      const res = await fetch(
        `/api/stocks?symbols=${encodeURIComponent(list.join(","))}`,
        { cache: "no-store", signal },
      );
      if (!res.ok) return;
      const feed = (await res.json()) as {
        quotes?: { symbol: string; name: string; price: number; dayPct: number }[];
        upstreamBudgetRemaining?: number;
      };

      if (typeof feed.upstreamBudgetRemaining === "number") {
        setEquityBudgetRemaining(feed.upstreamBudgetRemaining);
      }
      if (!Array.isArray(feed.quotes) || feed.quotes.length === 0) {
        setEquitiesReady(true);
        return;
      }

      const at = Date.now();
      setEquities((prev) => {
        const next: QuoteMap = { ...prev };
        for (const q of feed.quotes ?? []) {
          if (!(q.price > 0)) continue;
          const key = q.symbol.toUpperCase();
          next[key] = {
            symbol: key,
            name: q.name || key,
            price: q.price,
            dayPct: Number.isFinite(q.dayPct)
              ? q.dayPct
              : (prev[key]?.dayPct ?? 0),
            at,
          };
        }
        return next;
      });
      setLastUpdated(at);
      setEquitiesReady(true);
    } catch {
      /* Same contract as crypto: degrade to last known, never blank out. */
    }
  }, []);

  /* Two independent cadences. Each aborts its in-flight request on cleanup
     so a fast unmount (or a symbol change) can't land a stale write. */
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    void fetchCrypto(controller.signal);
    const id = window.setInterval(
      () => void fetchCrypto(controller.signal),
      CRYPTO_INTERVAL_MS,
    );
    return () => {
      controller.abort();
      window.clearInterval(id);
    };
  }, [enabled, fetchCrypto]);

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    void fetchEquities(controller.signal);
    const id = window.setInterval(
      () => void fetchEquities(controller.signal),
      EQUITY_INTERVAL_MS,
    );
    return () => {
      controller.abort();
      window.clearInterval(id);
    };
    /* `symbols` is a dependency on purpose: adding a ticker from the
       terminal should pull its first quote immediately, not up to 30s
       later. `fetchEquities` reads the ref, so it stays referentially
       stable and the interval is only rebuilt on a real universe change. */
  }, [enabled, fetchEquities, symbols]);

  const refresh = useCallback(() => {
    void fetchCrypto();
    void fetchEquities();
  }, [fetchCrypto, fetchEquities]);

  const all = useMemo(
    () => ({ ...crypto, ...equities }),
    [crypto, equities],
  );

  const value = useMemo<MarketDataValue>(
    () => ({
      equities,
      crypto,
      all,
      isHydrating: !(cryptoReady && equitiesReady),
      lastUpdated,
      equityBudgetRemaining,
      trackSymbols,
      refresh,
    }),
    [
      equities,
      crypto,
      all,
      cryptoReady,
      equitiesReady,
      lastUpdated,
      equityBudgetRemaining,
      trackSymbols,
      refresh,
    ],
  );

  return (
    <MarketDataContext.Provider value={value}>
      {children}
    </MarketDataContext.Provider>
  );
}

/* -------------------------------------------------------------------------- */
/*                                   Hooks                                    */
/* -------------------------------------------------------------------------- */

/** Reads the shared store. Throws if the provider is missing — a silent
    fallback here would reintroduce drift invisibly, which is worse. */
export function useMarketData(): MarketDataValue {
  const ctx = useContext(MarketDataContext);
  if (!ctx) {
    throw new Error(
      "useMarketData must be used inside <MarketDataProvider> (mount it in app/layout.tsx)",
    );
  }
  return ctx;
}

/** Convenience reader for a single symbol from the shared dictionary. */
export function useQuote(symbol: string): MarketQuote | null {
  const { all } = useMarketData();
  return all[symbol.trim().toUpperCase()] ?? null;
}

/* -------------------------------------------------------------------------- */
/*                        Research terminal search hook                       */
/* -------------------------------------------------------------------------- */

export type TerminalMatch = {
  symbol: string;
  name: string;
  region: string;
  currency: string;
};

export type TerminalSnapshot = {
  symbol: string;
  quote: {
    symbol: string;
    name: string;
    price: number;
    change: number;
    dayPct: number;
    open: number;
    high: number;
    low: number;
    previousClose: number;
    volume: number;
  };
  profile: {
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
  } | null;
  candles: {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
  range: string;
  rangeStats: {
    first: number;
    last: number;
    high: number;
    low: number;
    changePct: number;
  } | null;
};

/**
 * Live ticker search + research loader for the ⌘K terminal.
 *
 * Replaces every hardcoded ticker filter: `matches` is whatever the upstream
 * symbol search returns for the typed query (debounced 300ms so a fast
 * typist costs one request, not eight), and `load(symbol)` pulls the full
 * research payload for the selected ticker.
 *
 * Any loaded symbol is pushed into the shared store via `trackSymbols`, so
 * the price the terminal shows is the SAME dictionary entry the marquee and
 * the dashboard cards read — which is what keeps the three in lockstep.
 */
export function useTickerSearch(query: string, range = "1Y") {
  const { trackSymbols, all } = useMarketData();

  const [matches, setMatches] = useState<TerminalMatch[]>([]);
  const [searching, setSearching] = useState(false);
  const [snapshot, setSnapshot] = useState<TerminalSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /* ---- Debounced symbol search ---------------------------------------- */
  useEffect(() => {
    const q = query.trim();
    if (q.length < 1) {
      setMatches([]);
      setSearching(false);
      return;
    }

    setSearching(true);
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const res = await fetch(
            `/api/stocks?search=${encodeURIComponent(q)}`,
            { cache: "no-store", signal: controller.signal },
          );
          if (!res.ok) throw new Error(`search ${res.status}`);
          const data = (await res.json()) as { matches?: TerminalMatch[] };
          setMatches(Array.isArray(data.matches) ? data.matches : []);
        } catch {
          /* Aborts are routine while typing — never surface them. */
          if (!controller.signal.aborted) setMatches([]);
        } finally {
          if (!controller.signal.aborted) setSearching(false);
        }
      })();
    }, 300);

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query]);

  /* ---- Full research payload for a selected ticker --------------------- */
  const load = useCallback(
    async (symbol: string) => {
      const s = symbol.trim().toUpperCase();
      if (!s) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `/api/stocks/${encodeURIComponent(s)}?range=${encodeURIComponent(range)}`,
          { cache: "no-store" },
        );
        if (!res.ok) {
          setError(
            res.status === 404
              ? `No market data for ${s}.`
              : "The market feed is unreachable right now.",
          );
          setSnapshot(null);
          return;
        }
        const data = (await res.json()) as TerminalSnapshot;
        setSnapshot(data);
        /* Fold the viewed symbol into the shared polling universe. */
        trackSymbols([s]);
      } catch {
        setError("The market feed is unreachable right now.");
        setSnapshot(null);
      } finally {
        setLoading(false);
      }
    },
    [range, trackSymbols],
  );

  /**
   * The store's live price for the loaded symbol, if one has ticked since
   * the snapshot was taken. The terminal should render THIS for the headline
   * price, falling back to the snapshot — that is the last mile of drift
   * elimination, since the snapshot is a point-in-time REST read while the
   * store is the continuously-updated shared truth.
   */
  const livePrice = snapshot ? (all[snapshot.symbol] ?? null) : null;

  return { matches, searching, snapshot, livePrice, loading, error, load };
}