"use client";

/**
 * Fey-style Personal Finance Dashboard — Command Center shell.
 *
 * A single `activeView` state drives the entire surface:
 *   "home"     — grid of live snapshot bubbles + core financial metric
 *                cards (the portal)
 *   "finance"  — Net Worth command center + linked ledger
 *   "stocks"   — dedicated single-equity research terminal
 *   "spending" — category budget tracker
 *   "forecast" — Wealth Forecasting Engine simulation canvas
 *
 * Every view wrapper carries the `.animate-fey-fade` entrance (globals.css)
 * so switching tabs yields a fluid fading lift transition. The
 * `isPrivateMode` eye toggle in the global header applies a uniform
 * `blur-md` veil over every currency figure, percentage indicator, crypto
 * balance and line-graph point across all views.
 *
 * The "+ Connect Portfolio" / "+ Add Account" buttons launch the Plaid-style
 * multi-step syncing overlay (ConnectAccountModal). When an institution
 * completes the flow, its mock balance is ingested into the Net Worth / Cash
 * metric totals and its transactions are prepended to the chronological ledger.
 *
 * Layout routing is gated by a premium App Authentication Screen Gateway:
 * an `authState` of "login" | "register" intercepts the entire workspace
 * shell behind an ultra-minimal dark-mode credentials window; only after
 * "Create Account" / "Sign In" resolves does `authState` flip to
 * "authenticated". First-time users (`isNewUser`) then land on a clean
 * onboarding screen — "Welcome to your financial cockpit..." — which embeds
 * the 3-step Plaid-style multi-institution sync grid before the cockpit
 * mounts. "Skip for now" safely routes them to a blank, zeroed dashboard
 * state for manual data entry later.
 *
 * Phase 2 additions:
 *   - Personal Cash Flow Input Calculator — clicking the Monthly Cash Flow
 *     card opens an elegant overlay (`CashFlowModal`) where raw monthly
 *     metrics (take-home income, rent/mortgage, utilities & insurance,
 *     discretionary spend) are entered or slid; a reactive panel derives
 *     Total Monthly Outlays, Net Savings Capacity and a 50/30/20-baselined
 *     Cash Flow Health Score. "Save Cash Flow Profile" commits the values to
 *     global state and the Monthly Cash Flow card updates instantly.
 *   - Running Market Marquee Ticker — a fixed bottom bar of live quotes that
 *     mounts ONLY while `activeView === "stocks"` or the ⌘K research overlay
 *     (broadcast by the root layout) is open; other views unmount it cleanly.
 *
 * Phase 3 additions:
 *   - Blur-and-Focus View Loading Transition — every view navigation flips a
 *     temporary `isPageLoading` hook true for exactly 450ms; the active
 *     center panel veils with `blur-md opacity-60 scale-[0.99]` while a
 *     minimal geometric ring pulses dead-center of the viewport, then the
 *     veil dissolves the panel back to full-focus crisp rendering.
 *   - Floating Right-Hand AI Advisor Sidebar — the cockpit is restructured
 *     into an app shell anchoring a dedicated `w-80 h-full` right-hand column
 *     (hidden below lg). A minimalist header toggle next to the Privacy Eye
 *     slides it fully open/collapsed (`transition-transform duration-300`).
 *     Its interior is a scrollable messaging bubble history that parses the
 *     manual cash flow profile, connected account records and live crypto
 *     tickers to stream tailored markdown advice; new messages auto-scroll
 *     to the absolute bottom, and Privacy Mode uniformly masks every currency
 *     figure through the `mask()` utility. The old finance-view Assistant
 *     brief card is superseded by this column.
 *
 * Phase 4 additions:
 *   - Zero-Baseline Asset Initialization — every dollar aggregate (Net
 *     Worth, Cash, Liabilities, crypto holdings, equity positions, ledger,
 *     spending) mounts strictly at $0. The placeholder transaction logs are
 *     gone: the ledger mounts empty and the finance view renders the
 *     elegant "Your cockpit is clear." empty-state row. Linked state
 *     persists in localStorage (`finance-dashboard:linked-state:v1`):
 *     registration wipes the store back to the pristine zero baseline,
 *     sign-in rehydrates whatever was previously linked, and an empty store
 *     simply stays $0.
 *   - Live Market Hydration — the former random-drift simulation is now a
 *     real feed: a 3-second loop pulls BTC/ETH/SOL prices + 24h deltas from
 *     the local `/api/crypto` route (CoinGecko proxied server-side — the key
 *     never reaches the browser) and a 30-second loop pulls NVDA/TSLA/AAPL/
 *     AMZN quotes from `/api/stocks` (Alpha Vantage proxied + quota-guarded).
 *     Incoming real-world prices hydrate the ticker cards, the marquee
 *     track and — through the market-value delta effect — the master
 *     dashboard balances.
 *   - Live AI Advisor — composer questions POST to `/api/chat`, which drives
 *     `gemini-3.6-flash` server-side (GEMINI_API_KEY never leaves the
 *     server); replies render as contextual markdown in the sidebar with a
 *     graceful offline snapshot fallback when the engine is unreachable.
 *
 * Phase 5 additions — zero-state resolution + navigation systems:
 *   1. Identity Menu — the header greeting is now a trigger opening an
 *      absolute `w-48` overlay card (Settings · Account Info · Linked
 *      Institutions · Log Out). "Log Out" drops `authState` back to "login",
 *      re-arming Gate 1 and tearing down per-session UI state.
 *   2. Zero-State Hero CTA — at a precise $0 aggregate the home net-worth
 *      canvas no longer draws the seeded historical SVG walk (performance
 *      that never happened); the node is replaced by a "Connect First
 *      Portfolio" hero wired straight into the Plaid sync modal. The finance
 *      view's trailing-year chart follows the same rule.
 *   3. View-Scoped Advisor — the sidebar brief is a pure derivation of live
 *      state AND `activeView`, replacing the stream-once global zero block:
 *      home → onboarding/net worth, finance → cash flow + margins, stocks →
 *      watchlist/allocation, spending → budget pressure, forecast →
 *      compounding math. The conversation thread renders beneath it and
 *      survives view switches intact.
 *   4. Forecast Soft-Lock — a $0 principal makes `FV = PV × (1 + r/12)^(12t)`
 *      return zero at every horizon, so the canvas is gated behind a
 *      translucent "Unlock Forecasting Engine" banner with the engine
 *      rendered inert underneath.
 *   5. Watchlist Framing — with zero shares held, the equity panel declares
 *      itself a Market Watchlist and withholds every position-derived metric
 *      (portfolio value, weighted day move, share counts, $0 market values),
 *      showing only live quotes; funding any position promotes it to Active
 *      Investment Holdings with a secondary watch-only strip.
 */

import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { useMarketData } from "@/lib/market-store";
import { clearSession, readSession, writeSession } from "@/lib/auth-session";
import BillCalendar, { resolveBillDay, type Bill } from "./bill-calendar";
import AuthGateway, { type AuthMode } from "./auth-gateway";
import CashFlowModal, { type CashFlowProfile } from "./cash-flow-modal";
import ConnectAccountModal from "./connect-account-modal";
import MarketMarqueeTicker, { type TickerQuote } from "./market-marquee-ticker";
import OnboardingSync from "./onboarding-sync";
import StockResearchTerminal from "./search/stock-research-terminal";
import { TERMINAL_VISIBILITY_EVENT } from "./terminal-events";
import type { Institution } from "./institutions";

/* -------------------------------------------------------------------------- */
/*                                   Types                                    */
/* -------------------------------------------------------------------------- */

type DashboardView =
  | "home"
  | "finance"
  | "stocks"
  | "spending"
  | "forecast"
  | "calendar";

type PortfolioMetric = {
  label: string;
  value: number;
  delta: string;
  positive: boolean;
  active?: boolean;
};

type LedgerEntry = {
  id: string;
  merchant: string;
  /** Negative = debit, positive = credit. */
  amount: number;
  category: string;
  time: string;
  institution: string;
  color: string;
};

type ConnectedAccount = {
  id: string;
  name: string;
  color: string;
  balance: number;
};

type ExpenseCategory = {
  name: string;
  /** Dollars spent this month against the category budget. */
  spent: number;
  progress: number; // 0–100, ratio of monthly budget used
  negative?: boolean;
};

/** A wallet/exchange token position shown in the Crypto Balances hub. */
type CryptoHolding = {
  token: string;
  name: string;
  color: string;
  amount: number;
  price: number;
  deltaPct: number;
};

/** A live position mirrored from the linked brokerage research terminal. */
type TrackedPosition = {
  ticker: string;
  name: string;
  color: string;
  shares: number;
  avgCost: number;
  price: number;
  dayPct: number;
};

/** Phase 3 — one message in the floating AI Advisor sidebar thread. */
type AdvisorMessage = {
  id: string;
  role: "advisor" | "user";
  /** Minimal inline markdown: **bold** and `code` tokens, "\n" line breaks. */
  body: string;
};

/**
 * Live-state snapshot the advisor brief is parsed from. Phase 5 widens it
 * with the position/allocation and category fields the view-scoped branches
 * need, so a single snapshot can serve every surface of the cockpit.
 */
type AdvisorContext = {
  netWorthValue: number;
  netWorthDelta: string;
  cash: number;
  income: number;
  outlays: number;
  profileSaved: boolean;
  connectedCount: number;
  crypto: CryptoHolding[];
  overBudget: string[];
  essentialsMonthly: number;
  recentLedger: LedgerEntry[];
  /** Phase 5 — equity book state, for the stocks-view allocation branch. */
  portfolioValue: number;
  ownedPositions: number;
  /** Tracked tickers currently held at 0 shares (watch-only). */
  watchlistTickers: string[];
  topMover: { ticker: string; dayPct: number } | null;
  /** Phase 5 — spending-view branch inputs. */
  totalSpent: number;
  topCategory: ExpenseCategory | null;
  /** Phase 7 — calendar-view branch inputs. */
  billCount: number;
  billMonthlyTotal: number;
  nextBill: { label: string; amount: number; daysOut: number } | null;
};

/* Phase 6 — the `/api/crypto` and `/api/stocks` feed payload types formerly
   declared here have moved to lib/market-store.tsx, which is now the only
   module in the app that talks to those routes directly. */

/* -------------------------------------------------------------------------- */
/*                  Zero-baseline seed data (Phase 4)                         */
/* -------------------------------------------------------------------------- */

/**
 * Zero-baseline asset initialization — every aggregate mounts strictly at
 * $0. Balances only move when an institution is linked or a persisted
 * linked-state (localStorage) is rehydrated on sign-in. The placeholder
 * transaction logs are removed entirely: the ledger mounts empty and the
 * finance view renders the clear-cockpit empty-state row.
 */
const ZERO_METRICS: PortfolioMetric[] = [
  { label: "Net Worth", value: 0, delta: "0.0%", positive: true, active: true },
  { label: "Cash", value: 0, delta: "0.0%", positive: true },
  { label: "Liabilities", value: 0, delta: "0.0%", positive: false },
];

/**
 * Category scaffolding with default monthly budgets. Phase 4: `spent` is no
 * longer seeded — it is derived live from real ledger debits, so a clear
 * cockpit spends $0 across every category and linking accounts populates
 * the tracker honestly.
 */
const SPENDING_CATEGORIES: { name: string; monthlyBudget: number }[] = [
  { name: "Housing", monthlyBudget: 1850 },
  { name: "Groceries", monthlyBudget: 950 },
  { name: "Transport", monthlyBudget: 460 },
  { name: "Dining", monthlyBudget: 450 },
  { name: "Subscriptions", monthlyBudget: 290 },
  { name: "Utilities", monthlyBudget: 780 },
  { name: "Health", monthlyBudget: 450 },
  { name: "Shopping", monthlyBudget: 450 },
  { name: "Travel", monthlyBudget: 1000 },
  { name: "Misc", monthlyBudget: 600 },
];

/**
 * Tracked equity universe (NVDA, TSLA, AAPL, AMZN) — zero baseline: share
 * counts mount at 0 and prices at $0 until `/api/stocks` hydrates live
 * quotes and linked share counts arrive from persisted state. All rendering
 * flows through component state, never through this seed.
 */
const TRACKED_UNIVERSE: TrackedPosition[] = [
  { ticker: "NVDA", name: "NVIDIA Corporation", color: "bg-green-600", shares: 0, avgCost: 0, price: 0, dayPct: 0 },
  { ticker: "TSLA", name: "Tesla, Inc.", color: "bg-red-600", shares: 0, avgCost: 0, price: 0, dayPct: 0 },
  { ticker: "AAPL", name: "Apple Inc.", color: "bg-zinc-500", shares: 0, avgCost: 0, price: 0, dayPct: 0 },
  { ticker: "AMZN", name: "Amazon.com, Inc.", color: "bg-orange-500", shares: 0, avgCost: 0, price: 0, dayPct: 0 },
];

/**
 * Crypto wallet/exchange balances for the Crypto Balances hub — zero
 * baseline: token amounts mount at 0 (market value $0) until linked data is
 * rehydrated; live prices + 24h deltas hydrate from `/api/crypto`.
 */
const ZERO_CRYPTO_HOLDINGS: CryptoHolding[] = [
  { token: "BTC", name: "Bitcoin", color: "bg-amber-500", amount: 0, price: 0, deltaPct: 0 },
  { token: "ETH", name: "Ethereum", color: "bg-indigo-400", amount: 0, price: 0, deltaPct: 0 },
  { token: "SOL", name: "Solana", color: "bg-fuchsia-500", amount: 0, price: 0, deltaPct: 0 },
];

/**
 * Phase 4 — the Personal Cash Flow Input Calculator starts from an all-zero
 * profile; committing real inputs via "Save Cash Flow Profile" populates it.
 */
const ZERO_CASH_FLOW_PROFILE: CashFlowProfile = {
  income: 0,
  rent: 0,
  utilities: 0,
  discretionary: 0,
};

/** Current borrowing rating + ceiling for the credit health monitor ring. */
const CREDIT_SCORE = 782;
const CREDIT_SCORE_MAX = 850;

/** Categories counted toward essential living expenses (emergency runway). */
const ESSENTIAL_CATEGORIES = [
  "Housing",
  "Groceries",
  "Transport",
  "Utilities",
  "Health",
];

/* ============ PHASE 7 — FLOATING COMMAND DOCK DEFINITIONS =============
   The four primary view states named in the design spec, plus a secondary
   pair. Spending and Forecast are fully built surfaces with live data
   behind them; dropping them from navigation entirely would orphan working
   features, so they sit in a second dock group after a hairline divider —
   present, but visually subordinate to the four primaries. */

type DockItem = { key: DashboardView; label: string; icon: DockIconName };

type DockIconName =
  | "home"
  | "finance"
  | "stocks"
  | "calendar"
  | "spending"
  | "forecast";

const DOCK_PRIMARY: DockItem[] = [
  { key: "home", label: "Home", icon: "home" },
  { key: "finance", label: "Finance", icon: "finance" },
  { key: "stocks", label: "Stocks", icon: "stocks" },
  { key: "calendar", label: "Calendar", icon: "calendar" },
];

const DOCK_SECONDARY: DockItem[] = [
  { key: "spending", label: "Spending", icon: "spending" },
  { key: "forecast", label: "Forecast", icon: "forecast" },
];

/** Dock glyphs. One stroked 24×24 grid, so every icon shares an optical
    weight — mixing filled and stroked marks is what makes a dock look
    assembled from clip art. */
function DockIcon({ name }: { name: DockIconName }) {
  const paths: Record<DockIconName, React.ReactNode> = {
    home: <path d="M3 10.5 12 3l9 7.5M5.5 9.5V20h13V9.5" />,
    finance: (
      <>
        <path d="M3 17.5 9 11l4 4 8-8.5" />
        <path d="M21 6.5h-4.5M21 6.5V11" />
      </>
    ),
    stocks: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M7 15.5l3-3.5 2.5 2.5L17 9" />
      </>
    ),
    calendar: (
      <>
        <rect x="3" y="5" width="18" height="16" rx="2" />
        <path d="M3 10h18M8 3v4M16 3v4" />
      </>
    ),
    spending: (
      <>
        <circle cx="12" cy="12" r="8.5" />
        <path d="M12 3.5v8.5h8.5" />
      </>
    ),
    forecast: (
      <>
        <path d="M3 20h18" />
        <path d="M5 16.5c3.5 0 4.5-9 7.5-9s4 5.5 6.5 5.5" />
      </>
    ),
  };

  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="h-[18px] w-[18px]"
    >
      {paths[name]}
    </svg>
  );
}

const ranges = ["1M", "3M", "YTD", "1Y", "2Y"];

/** Phase 5 — shared row styling for the header identity menu items. */
const menuItemCls =
  "flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left text-xs tracking-tight text-text-muted transition-colors hover:bg-foreground/5 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40";

/** Trailing-year Net Worth walk (white line in both home bubble + finance view). */
const NET_WORTH_PATH =
  "M0 150 L40 140 L80 145 L120 120 L160 125 L200 100 L240 105 L280 80 " +
  "L320 60 L360 70 L400 45 L440 55 L480 30 L520 25 L600 10";

/** Secondary cash-balance line for the multi-line home sparkline. */
const CASH_SPARK_PATH =
  "M0 128 L60 124 L120 127 L180 118 L240 121 L300 112 L360 116 " +
  "L420 110 L480 113 L540 106 L600 108";

/* -------------------------------------------------------------------------- */
/*                                  Helpers                                   */
/* -------------------------------------------------------------------------- */

/** Whole-dollar USD, e.g. 248300 → "248,300". */
function usd0(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

/** Two-decimal USD for ledger amounts, e.g. 86.41 → "86.41". */
function usd2(n: number): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/** Signed amount, e.g. -86.41 → "−$86.41" / 4210 → "+$4,210.00". */
function signedUsd2(n: number): string {
  return `${n >= 0 ? "+" : "−"}$${usd2(Math.abs(n))}`;
}

/** Thin horizontal tracking bar — accent-green fill on muted track. */
function TrackingBar({ progress, negative }: { progress: number; negative?: boolean }) {
  return (
    <div className="h-px w-32 bg-border-muted relative overflow-visible">
      <div
        className={`absolute left-0 top-0 h-px ${
          negative ? "bg-red-500" : "bg-accent-green"
        } transition-all duration-500`}
        style={{ width: `${Math.min(progress, 100)}%` }}
      />
    </div>
  );
}

/** Shared "+ action" affordance matching the Fey metric-ticker style. */
function AddAction({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1.5 text-text-muted hover:text-foreground cursor-pointer"
    >
      <span className="inline-block w-3.5 h-3.5 rounded-full border border-border-muted text-[9px] leading-[13px] text-center">
        +
      </span>
      {label}
    </button>
  );
}

/** Eye / eye-off glyph for the global privacy toggle. */
function EyeIcon({ closed }: { closed: boolean }) {
  return closed ? (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="h-4 w-4"
    >
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  ) : (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="h-4 w-4"
    >
      <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

/**
 * Clickable snapshot bubble for the home grid — hover-responsive module card
 * that deep-links into its dedicated view.
 */
function BubbleCard({
  onClick,
  hint,
  ariaLabel,
  className = "",
  children,
}: {
  onClick: () => void;
  hint: string;
  ariaLabel: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section
      role="button"
      tabIndex={0}
      aria-label={ariaLabel}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      className={`group bg-card border border-border-muted rounded-xl p-6 text-left transition-all cursor-pointer hover:border-foreground/30 focus-visible:outline-none focus-visible:border-foreground/50 ${className}`}
    >
      <div className="flex h-full flex-col">{children}</div>
      <div className="mt-4 flex items-center justify-between">
        <span className="text-xs tracking-tight text-text-muted transition-colors group-hover:text-foreground">
          {hint} <span aria-hidden>→</span>
        </span>
      </div>
    </section>
  );
}

/** Data-driven cash-flow sparkline rendered from real ledger amounts. */
function CashFlowSparkline({ amounts }: { amounts: number[] }) {
  const pts = amounts.length >= 2 ? amounts : [0, 1];
  const max = Math.max(...pts);
  const min = Math.min(...pts);
  const span = max - min || 1;
  const step = 600 / (pts.length - 1);
  const d = pts
    .map(
      (v, i) =>
        `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(
          50 -
          ((v - min) / span) * 44
        ).toFixed(1)}`,
    )
    .join(" ");
  return (
    <svg
      viewBox="0 0 600 60"
      preserveAspectRatio="none"
      className="h-12 w-full"
      aria-hidden
    >
      <path d={d} fill="none" stroke="#8b7ec8" strokeWidth="1.5" opacity="0.8" />
    </svg>
  );
}

/** Circular progress monitor ring for the Credit Score health card. */
function CreditScoreRing({ score }: { score: number }) {
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.min(score / CREDIT_SCORE_MAX, 1);
  return (
    <div className="relative h-16 w-16 flex-shrink-0">
      <svg viewBox="0 0 64 64" className="h-16 w-16 -rotate-90" aria-hidden>
        <circle
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          stroke="var(--border-muted)"
          strokeWidth="4"
        />
        <circle
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          stroke="var(--accent-green)"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - pct)}
          className="transition-all duration-500"
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold tabular-nums tracking-tight text-foreground">
        {score}
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*                    Phase 3 — AI advisor sidebar pieces                     */
/* -------------------------------------------------------------------------- */

/** Header toggle glyph — a browser frame whose right column is the advisor. */
function AdvisorPanelIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="h-4 w-4"
    >
      <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
      <line x1="15" y1="4.5" x2="15" y2="19.5" />
      <circle
        cx="18.6"
        cy="12"
        r="1.3"
        fill="currentColor"
        stroke="none"
        fillOpacity={open ? 1 : 0.35}
      />
    </svg>
  );
}

/** Four-point spark avatar for advisor-authored bubbles. */
function SparkIcon({ className = "h-3 w-3" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden className={className}>
      <path d="M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2 2-6z" />
    </svg>
  );
}

/** Collapse chevron for the sidebar header. */
function ChevronRightIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="h-3.5 w-3.5"
    >
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

/** Composer send arrow. */
function SendIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="h-3.5 w-3.5"
    >
      <path d="M5 12h13" />
      <path d="M13 6l6 6-6 6" />
    </svg>
  );
}

/** Currency ($1,234.56) and percent (12.3%) figure matcher — routes chat
 * text through the privacy mask() veil without touching ordinary numbers. */
const ADVISOR_FIGURE_PATTERN =
  /([-+−]?\$[\d][\d,]*(?:\.\d+)?)|([-+−]?\d[\d,]*(?:\.\d+)?%)/g;

/** Splits `text` so every currency/percent figure renders inside mask(). */
function figureSegments(
  text: string,
  mask: (figure: ReactNode) => ReactNode,
): ReactNode {
  if (!text) return null;
  return text
    .split(ADVISOR_FIGURE_PATTERN)
    .map((part, i) => {
      if (!part) return null;
      /* With two capture groups, every odd index is a matched figure. */
      return i % 2 === 1 ? (
        <span key={i}>{mask(part)}</span>
      ) : (
        <Fragment key={i}>{part}</Fragment>
      );
    });
}

/** Minimal markdown renderer for advisor lines: **bold**, `code`, and line
 * breaks — with every currency/percent figure masked through mask(). */
function renderAdvisorRich(
  body: string,
  mask: (figure: ReactNode) => ReactNode,
): ReactNode {
  return body.split("\n").map((line, li) => (
    <Fragment key={li}>
      {li > 0 && <br />}
      {line
        .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
        .map((token, ti) => {
          if (!token) return null;
          if (
            token.startsWith("**") &&
            token.endsWith("**") &&
            token.length > 4
          ) {
            return (
              <strong
                key={`${li}-${ti}`}
                className="font-semibold text-foreground"
              >
                {figureSegments(token.slice(2, -2), mask)}
              </strong>
            );
          }
          if (
            token.startsWith("`") &&
            token.endsWith("`") &&
            token.length > 2
          ) {
            return (
              <code
                key={`${li}-${ti}`}
                className="rounded bg-border-muted/60 px-1 py-0.5 font-sans text-[10px] text-foreground"
              >
                {figureSegments(token.slice(1, -1), mask)}
              </code>
            );
          }
          return (
            <Fragment key={`${li}-${ti}`}>
              {figureSegments(token, mask)}
            </Fragment>
          );
        })}
    </Fragment>
  ));
}

/**
 * Phase 6 — advisor loading skeleton. Occupies the same bubble geometry the
 * real reply will land in (avatar + rounded panel), so the thread doesn't
 * jump when the response arrives. Three pulsing bars of uneven width read as
 * "prose is coming", where a spinner would read as "something is stuck".
 */
function AdvisorSkeleton() {
  return (
    <div
      className="advisor-bubble flex items-start gap-2.5"
      role="status"
      aria-label="Advisor is composing a reply"
    >
      <span className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border border-border-muted bg-card text-accent-green">
        <span className="anim-pulse-soft">
          <SparkIcon />
        </span>
      </span>
      <div className="min-w-0 flex-1 rounded-xl rounded-tl-sm border border-border-muted bg-card/80 px-3 py-2.5">
        <div className="flex flex-col gap-2">
          <span className="fey-skeleton-bar block h-2 w-[88%] rounded-full" />
          <span className="fey-skeleton-bar block h-2 w-[64%] rounded-full [animation-delay:140ms]" />
          <span className="fey-skeleton-bar block h-2 w-[76%] rounded-full [animation-delay:280ms]" />
        </div>
      </div>
    </div>
  );
}

/** One advisor/user chat bubble — mounts with the .advisor-bubble rise-in
 * (globals.css) so streamed lines glide into the history column. */
function AdvisorBubble({
  role,
  body,
  mask,
  delayMs = 0,
}: {
  role: AdvisorMessage["role"];
  body: string;
  mask: (figure: ReactNode) => ReactNode;
  /** Phase 5 — stagger for context-brief lines so they cascade in like a
      stream without any interval/setState machinery driving them. */
  delayMs?: number;
}) {
  const rendered = useMemo(() => renderAdvisorRich(body, mask), [body, mask]);
  const stagger = delayMs > 0 ? { animationDelay: `${delayMs}ms` } : undefined;
  if (role === "user") {
    return (
      <div className="advisor-bubble flex justify-end" style={stagger}>
        <p className="max-w-[85%] whitespace-pre-wrap rounded-xl rounded-br-sm border border-border-muted bg-border-muted/40 px-3 py-2 text-xs leading-relaxed tracking-tight text-foreground">
          {rendered}
        </p>
      </div>
    );
  }
  return (
    <div className="advisor-bubble flex items-start gap-2.5" style={stagger}>
      <span className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border border-border-muted bg-card text-accent-green">
        <SparkIcon />
      </span>
      <p className="min-w-0 max-w-[85%] whitespace-pre-wrap rounded-xl rounded-tl-sm border border-border-muted bg-card/80 px-3 py-2 text-xs leading-relaxed tracking-tight text-foreground/90">
        {rendered}
      </p>
    </div>
  );
}

/**
 * Phase 5 — per-view advisor framing. The sidebar header chip reads from
 * this map so the panel always declares which lens it is reasoning through.
 */
const ADVISOR_VIEW_FRAME: Record<DashboardView, { label: string; scope: string }> =
  {
    home: { label: "Home", scope: "Onboarding · Net worth" },
    finance: { label: "Finance", scope: "Cash flow · Margins" },
    stocks: { label: "Stocks", scope: "Watchlist · Allocation" },
    spending: { label: "Spending", scope: "Budgets · Categories" },
    forecast: { label: "Forecast", scope: "Compounding math" },
    calendar: { label: "Calendar", scope: "Bills · Due dates" },
  };

/**
 * Phase 5 — view-scoped context brief.
 *
 * The old builder emitted one global block of zero metrics that repeated
 * identically on every page. This one takes the live `activeView` and
 * returns advice written strictly for that surface:
 *
 *   home      → onboarding posture / net-worth composition
 *   finance   → cash flow configuration + operating margins
 *   stocks    → watchlist vs. held allocation
 *   spending  → category budget pressure
 *   forecast  → the compounding math driving the projection
 *
 * Every branch also has a distinct pre-data voice, so a $0 cockpit reads as
 * a setup checklist rather than a wall of zeros.
 */
function buildAdvisorBriefLines(
  ctx: AdvisorContext,
  view: DashboardView,
): string[] {
  const capacity = ctx.income - ctx.outlays;
  const rate = ctx.income > 0 ? (capacity / ctx.income) * 100 : 0;
  const cryptoValue = ctx.crypto.reduce((s, c) => s + c.amount * c.price, 0);
  const cryptoDay =
    ctx.crypto.reduce((s, c) => s + c.amount * c.price * c.deltaPct, 0) /
    (cryptoValue || 1);
  const tokens = ctx.crypto.map((c) => c.token).join(" · ");
  const runway = ctx.cash / (ctx.essentialsMonthly || 1);
  const debit = [...ctx.recentLedger]
    .filter((t) => t.amount < 0)
    .sort((a, b) => a.amount - b.amount)[0];
  const funded = ctx.netWorthValue !== 0;
  const hasPositions = ctx.ownedPositions > 0;

  switch (view) {
    /* ---------------- HOME · onboarding posture / net worth -------------- */
    case "home": {
      if (!funded) {
        return [
          `**Advisor online.** You're on a clean $0 baseline — nothing is broken, there's just no balance sheet to read yet.`,
          `Fastest path to a live cockpit: **Connect First Portfolio** on the net-worth card. A read-only token pulls balances plus historical ledger lines in one pass.`,
          ctx.connectedCount > 0
            ? `**${ctx.connectedCount}** institution${ctx.connectedCount === 1 ? "" : "s"} linked so far — balances are still settling into the aggregate.`
            : `Prefer manual? The **Monthly Cash Flow** card takes income and outlays directly — no bank link required.`,
          `Once a balance lands, this panel switches to composition analysis: liquid vs. invested vs. leveraged.`,
        ];
      }
      const invested = ctx.portfolioValue + cryptoValue;
      const liquidShare = (ctx.cash / (ctx.netWorthValue || 1)) * 100;
      return [
        `**Net worth: $${usd0(ctx.netWorthValue)}** (${ctx.netWorthDelta}) across **${ctx.connectedCount}** linked institution${ctx.connectedCount === 1 ? "" : "s"}.`,
        `Composition: **$${usd0(ctx.cash)}** liquid (${liquidShare.toFixed(1)}% of the base) against **$${usd0(invested)}** in market-exposed assets.`,
        `Emergency runway sits at **${runway.toFixed(1)} months** of essentials at the current burn.`,
        debit
          ? `Largest recent debit: **${debit.merchant}** (${signedUsd2(debit.amount)}) in ${debit.category}.`
          : `No debits on the ledger yet — category tracking stays idle until transactions land.`,
      ];
    }

    /* ------------- FINANCE · cash flow configuration & margins ----------- */
    case "finance": {
      if (ctx.income === 0 && ctx.outlays === 0) {
        return [
          `**Cash flow is unconfigured.** Income and outlays are both $0, so margin, savings rate and runway are all undefined rather than bad.`,
          `Open **Monthly Cash Flow** and enter take-home income, rent/mortgage, utilities and discretionary spend — that's the minimum viable profile.`,
          `I'll baseline it against **50/30/20**: needs ≤ 50% of take-home, wants ≤ 30%, savings ≥ 20%.`,
          `Linking a depository account instead back-fills the same numbers from real ledger debits.`,
        ];
      }
      return [
        `**Operating margin: ${capacity >= 0 ? "+" : "−"}$${usd0(Math.abs(capacity))}/mo** — **$${usd0(ctx.income)}** in against **$${usd0(ctx.outlays)}** out.`,
        `Savings rate is **${rate.toFixed(1)}%**${rate >= 20 ? " — above the 20% 50/30/20 floor." : rate > 0 ? " — under the 20% floor; trim discretionary first." : " — negative capacity, outlays exceed income."}`,
        `Liquid reserves cover **${runway.toFixed(1)} months** of essentials (**$${usd0(ctx.essentialsMonthly)}**/mo).`,
        ctx.profileSaved
          ? `Reading your **saved manual profile** — re-open the calculator any time to re-baseline.`
          : `These figures derive from **ledger activity**; a saved manual profile would override them.`,
      ];
    }

    /* --------------- STOCKS · watchlist vs. held allocation -------------- */
    case "stocks": {
      if (!hasPositions) {
        return [
          `**Watchlist mode.** You hold **0 shares** across ${ctx.watchlistTickers.length} tracked tickers — these are quotes, not positions.`,
          ctx.watchlistTickers.length > 0
            ? `Tracking **${ctx.watchlistTickers.join(" · ")}** on live quotes. Allocation analysis unlocks the moment a brokerage link reports real share counts.`
            : `Add tickers through the ⌘K research terminal to start a watchlist.`,
          ctx.topMover
            ? `Biggest move on the list today: **${ctx.topMover.ticker}** at ${ctx.topMover.dayPct >= 0 ? "+" : ""}${ctx.topMover.dayPct.toFixed(2)}%.`
            : `Quotes are still hydrating from the live feed.`,
          cryptoValue > 0
            ? `Crypto book — ${tokens} — carries **$${usd0(cryptoValue)}** at ${cryptoDay >= 0 ? "+" : ""}${cryptoDay.toFixed(2)}% today.`
            : `Crypto book (${tokens}) is tracked at zero balance — prices are live, holdings are not.`,
        ];
      }
      const equityShare =
        (ctx.portfolioValue / (ctx.portfolioValue + cryptoValue || 1)) * 100;
      return [
        `**Portfolio: $${usd0(ctx.portfolioValue)}** across **${ctx.ownedPositions}** held position${ctx.ownedPositions === 1 ? "" : "s"}.`,
        `Allocation split: **${equityShare.toFixed(0)}% equities** / **${(100 - equityShare).toFixed(0)}% crypto** of market-exposed capital.`,
        ctx.topMover
          ? `Session leader: **${ctx.topMover.ticker}** at ${ctx.topMover.dayPct >= 0 ? "+" : ""}${ctx.topMover.dayPct.toFixed(2)}%.`
          : `Session moves are still hydrating.`,
        ctx.watchlistTickers.length > 0
          ? `Still watch-only (0 shares): **${ctx.watchlistTickers.join(" · ")}**.`
          : `Every tracked ticker is a funded position — no watch-only names left.`,
      ];
    }

    /* ------------------ SPENDING · category budget pressure -------------- */
    case "spending": {
      if (ctx.totalSpent === 0) {
        return [
          `**No spend recorded this month.** Categories are scaffolded with monthly budgets but no debits have posted against them.`,
          `Category bars fill from real ledger debits — link a depository account and historical transactions back-fill instantly.`,
          `Essentials tracked for runway math: **${ESSENTIAL_CATEGORIES.join(" · ")}**.`,
        ];
      }
      return [
        `**$${usd0(ctx.totalSpent)}** spent this month, **$${usd0(ctx.essentialsMonthly)}** of it on essentials.`,
        ctx.topCategory
          ? `Heaviest category: **${ctx.topCategory.name}** at **$${usd0(ctx.topCategory.spent)}** (${ctx.topCategory.progress}% of budget).`
          : `No category is carrying unusual weight.`,
        ctx.overBudget.length > 0
          ? `Budget watch: **${ctx.overBudget.join(" and ")}** ${ctx.overBudget.length === 1 ? "is" : "are"} over budget — the top trim candidate${ctx.overBudget.length === 1 ? "" : "s"} this month.`
          : `Every category is tracking inside its monthly budget.`,
        debit
          ? `Largest single debit: **${debit.merchant}** (${signedUsd2(debit.amount)}).`
          : `No single debit dominates the ledger.`,
      ];
    }

    /* ---------------- CALENDAR · recurring liability pressure ------------ */
    case "calendar": {
      if (ctx.billCount === 0) {
        return [
          `**No recurring liabilities pinned yet.** The calendar is the one surface that models *obligations* rather than history.`,
          `Click any day tile to pin what repeats — rent on the 1st, subscriptions mid-month, a loan payment. Each one renders as a thin line under its date.`,
          `Once they're in, I fold them into runway math: committed monthly outflow is the number that decides how much of your income is genuinely discretionary.`,
        ];
      }
      const coverage =
        ctx.income > 0 ? (ctx.billMonthlyTotal / ctx.income) * 100 : 0;
      return [
        `**${ctx.billCount}** recurring ${ctx.billCount === 1 ? "liability" : "liabilities"} committing **$${usd0(ctx.billMonthlyTotal)}/mo**.`,
        ctx.income > 0
          ? `That's **${coverage.toFixed(1)}%** of take-home locked before any discretionary spend${coverage > 50 ? " — above the 50% needs ceiling in the 50/30/20 frame." : "."}`
          : `Enter your income in the cash flow calculator and I can tell you what share of take-home this locks up.`,
        ctx.nextBill
          ? `Next due: **${ctx.nextBill.label}**${ctx.nextBill.amount > 0 ? ` (**$${usd0(ctx.nextBill.amount)}**)` : ""} ${ctx.nextBill.daysOut === 0 ? "**today**" : ctx.nextBill.daysOut === 1 ? "**tomorrow**" : `in **${ctx.nextBill.daysOut} days**`}.`
          : `Nothing falls due in the next two weeks.`,
        ctx.cash > 0 && ctx.billMonthlyTotal > 0
          ? `Liquid cash covers **${(ctx.cash / ctx.billMonthlyTotal).toFixed(1)} months** of committed obligations at the current rate.`
          : `Link a depository account and I can tell you how many months of obligations your cash covers.`,
      ];
    }

    /* --------------- FORECAST · the compounding math itself -------------- */
    case "forecast": {
      if (!funded) {
        return [
          `**The engine is locked because compounding needs a non-zero principal.** The model is \`FV = PV × (1 + r/12)^(12t)\` — with \`PV = 0\`, every horizon returns 0.`,
          `That's arithmetic, not a bug: multiplying zero by any growth factor stays zero, so a $0 start can never cross a target.`,
          `Sync a portfolio balance, or enter a manual figure, and the curve anchors to it immediately.`,
          `At **7%/yr** compounded monthly the effective monthly rate is **0.565%**, doubling capital roughly every **10.2 years**.`,
        ];
      }
      const monthlyRate = (Math.pow(1.07, 1 / 12) - 1) * 100;
      const doublingYears = Math.log(2) / Math.log(1.07);
      const thirtyYear = ctx.netWorthValue * Math.pow(1.07, 30);
      return [
        `Anchored at **$${usd0(ctx.netWorthValue)}** — \`FV = PV × (1 + r/12)^(12t)\`, compounded monthly.`,
        `At **7%/yr** the effective monthly rate is **${monthlyRate.toFixed(3)}%**; capital doubles roughly every **${doublingYears.toFixed(1)} years**.`,
        `Left untouched, today's base projects to **$${usd0(thirtyYear)}** at the 30-year horizon — a **${Math.pow(1.07, 30).toFixed(2)}×** multiple.`,
        capacity > 0
          ? `Note the curve models the **principal only**. Your **${capacity >= 0 ? "+" : "−"}$${usd0(Math.abs(capacity))}/mo** capacity is modeled separately — contributions bend it far steeper.`
          : `The curve models principal only; a positive monthly contribution would steepen it materially.`,
      ];
    }
  }
}

/* -------------------------------------------------------------------------- */
/*                         Wealth Forecasting Engine                          */
/* -------------------------------------------------------------------------- */

/** Chart canvas geometry in viewBox units (stretched fluidly by CSS). */
const FORECAST_W = 600;
const FORECAST_H = 320;
const FORECAST_PAD_X = 10;
const FORECAST_PAD_TOP = 30;
const FORECAST_PAD_BOTTOM = 26;

const RETURN_PRESETS = [4, 7, 10];
const HORIZON_PRESETS = [5, 10, 20, 30];
const TARGET_PRESETS: { label: string; value: number }[] = [
  { label: "$250K", value: 250_000 },
  { label: "$500K", value: 500_000 },
  { label: "$1M", value: 1_000_000 },
  { label: "$2M", value: 2_000_000 },
];

/**
 * Monthly-compounded growth series: `start` compounded at `annualPct`
 * (geometric monthly rate) over `years`, returned month-indexed so the SVG
 * path arrays can be rebuilt synchronously on any parameter change.
 */
function compoundSeries(
  start: number,
  annualPct: number,
  years: number,
): number[] {
  const months = Math.round(years * 12);
  const monthly = Math.pow(1 + annualPct / 100, 1 / 12) - 1;
  const series = new Array<number>(months + 1);
  series[0] = start;
  for (let m = 1; m <= months; m += 1) {
    series[m] = series[m - 1] * (1 + monthly);
  }
  return series;
}

/** Compact USD, e.g. 248300 → "$248K", 2_370_000 → "$2.4M". */
function usdCompact(n: number): string {
  if (n >= 1_000_000) {
    return `$${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  if (n >= 1_000) return `$${Math.round(n / 1_000)}K`;
  return `$${Math.round(n)}`;
}

/**
 * Exact fractional months until `start` compounded at `annualPct` crosses
 * `target` — closed-form logarithm (0 when already reached, Infinity when
 * mathematically unreachable).
 */
function monthsToTarget(
  start: number,
  annualPct: number,
  target: number,
): number {
  if (target <= start) return 0;
  const monthly = Math.pow(1 + annualPct / 100, 1 / 12) - 1;
  if (start <= 0 || monthly <= 0) return Number.POSITIVE_INFINITY;
  return Math.log(target / start) / Math.log(1 + monthly);
}

/** Calendar month/year at which the projection first crosses the target. */
function hitDateLabel(months: number): string {
  const d = new Date();
  d.setMonth(d.getMonth() + Math.ceil(months));
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

/** "7y 6m"-style duration label from a fractional month count. */
function durationLabel(months: number): string {
  const yrs = Math.floor(months / 12);
  const mos = Math.round(months % 12);
  if (yrs === 0) return `${mos}mo`;
  return mos === 0 ? `${yrs}y` : `${yrs}y ${mos}m`;
}

/**
 * Interactive compounding-projection canvas. Columns 1–2 hold a pure-SVG
 * forward-sloping compound curve anchored on the live Net Worth value plus a
 * dashed Financial Freedom Target overlay whose side label resolves the exact
 * year/month of the crossing; column 3 is the Simulation Control Center
 * (Expected Annual Return %, Time Horizon, Target Goal Amount). Every change
 * recomputes the path arrays synchronously — the curve redraws with zero lag.
 */
function WealthForecastEngine({
  netWorth,
  mask,
  chartVeil,
}: {
  netWorth: number;
  mask: (figure: ReactNode) => ReactNode;
  chartVeil: string;
}) {
  /* Simulation parameters — return preset to a conservative index fund. */
  const [returnPct, setReturnPct] = useState(7);
  const [years, setYears] = useState(30);
  const [target, setTarget] = useState(1_000_000);

  /* Path arrays — rebuilt synchronously on any parameter / ticker change. */
  const series = useMemo(
    () => compoundSeries(netWorth, returnPct, years),
    [netWorth, returnPct, years],
  );
  const endValue = series[series.length - 1] ?? netWorth;
  const totalMonths = Math.max(years * 12, 1);

  const monthsHit = monthsToTarget(netWorth, returnPct, target);
  const withinHorizon = monthsHit <= totalMonths;
  const targetReached = monthsHit === 0;

  /* Geometry mapping — viewBox units; CSS stretches the canvas fluidly. */
  const innerW = FORECAST_W - FORECAST_PAD_X * 2;
  const innerH = FORECAST_H - FORECAST_PAD_TOP - FORECAST_PAD_BOTTOM;
  const maxV = Math.max(endValue, target) * 1.06;
  const xFor = (m: number) => FORECAST_PAD_X + (m / totalMonths) * innerW;
  const yFor = (v: number) => FORECAST_PAD_TOP + (1 - v / maxV) * innerH;

  const curvePath = series
    .map(
      (v, i) =>
        `${i === 0 ? "M" : "L"}${xFor(i).toFixed(1)},${yFor(v).toFixed(1)}`,
    )
    .join(" ");
  const baselineY = (FORECAST_H - FORECAST_PAD_BOTTOM).toFixed(1);
  const areaPath = `${curvePath} L${xFor(totalMonths).toFixed(1)},${baselineY} L${FORECAST_PAD_X},${baselineY} Z`;

  const goalY = yFor(target);
  const hitX = withinHorizon ? xFor(Math.min(monthsHit, totalMonths)) : null;
  /* Label chip centers on the crossing but clamps inside the canvas. */
  const labelLeftPct =
    hitX !== null
      ? Math.min(Math.max((hitX / FORECAST_W) * 100, 16), 84)
      : 84;
  const growthMultiple = netWorth > 0 ? endValue / netWorth : 0;

  const chipBase =
    "cursor-pointer rounded-md border px-2 py-1 text-xs tracking-tight transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40";
  const chipActive = "border-foreground/40 bg-border-muted/60 text-foreground";
  const chipIdle = "border-border-muted text-text-muted hover:text-foreground";

  return (
    <>
      {/* ------ Columns 1–2 (60%): compounding projection canvas ---------- */}
      <section className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-border-muted bg-card lg:col-span-3">
        <div className="p-6 pb-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Wealth Forecasting Engine
            </h2>
            <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-text-muted">
              <span className="anim-pulse-soft inline-block h-1 w-1 rounded-full bg-accent-green" />
              Live model
            </span>
          </div>
          <p className="mt-1 text-xs tracking-tight text-text-muted">
            Monthly compounding at {mask(`${returnPct.toFixed(1)}%`)} from a
            live start of {mask(`$${usd0(netWorth)}`)} — anchored to real-time
            net worth
          </p>
        </div>

        <div className={`relative mx-6 mb-2 min-h-[340px] flex-1 ${chartVeil}`}>
          <svg
            viewBox={`0 0 ${FORECAST_W} ${FORECAST_H}`}
            preserveAspectRatio="none"
            className="absolute inset-0 h-full w-full"
            aria-hidden
          >
            <defs>
              <linearGradient id="forecast-area" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="0%"
                  stopColor="var(--foreground)"
                  stopOpacity="0.09"
                />
                <stop
                  offset="100%"
                  stopColor="var(--foreground)"
                  stopOpacity="0"
                />
              </linearGradient>
            </defs>

            {/* Financial Freedom Target — crisp dashed goal line */}
            <line
              x1={FORECAST_PAD_X}
              x2={FORECAST_W - FORECAST_PAD_X}
              y1={goalY}
              y2={goalY}
              stroke="var(--text-muted)"
              strokeWidth="1"
              strokeDasharray="3 6"
              vectorEffect="non-scaling-stroke"
            />
            {/* Intersection drop-line down to the baseline */}
            {hitX !== null && (
              <line
                x1={hitX}
                x2={hitX}
                y1={goalY}
                y2={FORECAST_H - FORECAST_PAD_BOTTOM}
                stroke="var(--accent-green)"
                strokeOpacity="0.4"
                strokeWidth="1"
                strokeDasharray="2 5"
                vectorEffect="non-scaling-stroke"
              />
            )}

            <path d={areaPath} fill="url(#forecast-area)" />
            <path
              d={curvePath}
              fill="none"
              stroke="var(--foreground)"
              strokeWidth="1.5"
              vectorEffect="non-scaling-stroke"
            />
          </svg>

          {/* Crossing + start-node markers as HTML dots (aspect-true). */}
          {hitX !== null && (
            <span
              className="absolute z-10 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent-green ring-4 ring-accent-green/15"
              style={{
                left: `${(hitX / FORECAST_W) * 100}%`,
                top: `${(goalY / FORECAST_H) * 100}%`,
              }}
            />
          )}
          <span
            className="absolute z-10 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground"
            style={{
              left: `${(FORECAST_PAD_X / FORECAST_W) * 100}%`,
              top: `${(yFor(netWorth) / FORECAST_H) * 100}%`,
            }}
          />

          {/* Side interactive label — resolves the exact year/month the
              compounding line crosses the goal line. */}
          <div
            className="absolute z-10 -translate-x-1/2 -translate-y-[calc(100%_+_10px)] cursor-default rounded-lg border border-border-muted bg-background/90 px-3 py-2 backdrop-blur transition-colors hover:border-foreground/40"
            style={{
              left: `${labelLeftPct}%`,
              top: `${(goalY / FORECAST_H) * 100}%`,
            }}
            title="Monthly compounding of live net worth toward the target"
          >
            <p className="whitespace-nowrap text-[10px] uppercase tracking-wider text-text-muted">
              Financial Freedom Target · {mask(usdCompact(target))}
            </p>
            <p
              className={`whitespace-nowrap text-xs font-medium tracking-tight ${
                targetReached
                  ? "text-foreground"
                  : withinHorizon
                    ? "text-accent-green"
                    : "text-amber-500"
              }`}
            >
              {targetReached
                ? "Goal achieved — raise the target"
                : withinHorizon
                  ? mask(
                      `Reached ${hitDateLabel(monthsHit)} · ${durationLabel(monthsHit)}`,
                    )
                  : `Beyond the ${years}-year horizon`}
            </p>
          </div>
        </div>

        {/* X-axis — horizon ticks */}
        <div className="flex justify-between px-6 pb-4 pt-1 text-[11px] tabular-nums tracking-tight text-text-muted">
          {[0, 1, 2, 3, 4].map((i) => (
            <span key={i}>
              {i === 0 ? "Today" : `+${Math.round((years * i) / 4)}y`}
            </span>
          ))}
        </div>
      </section>

      {/* ------ Column 3 (40%): Simulation Control Center ----------------- */}
      <aside className="flex min-w-0 flex-col gap-4 lg:col-span-2">
        <section className="rounded-xl border border-border-muted bg-card p-6">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-foreground">
              Simulation Control Center
            </span>
            <span className="text-[10px] uppercase tracking-wider text-text-muted">
              Compounding model
            </span>
          </div>

          {/* Expected Annual Return */}
          <div className="mt-6">
            <div className="flex items-baseline justify-between">
              <label
                htmlFor="forecast-return"
                className="text-[11px] uppercase tracking-wider text-text-muted"
              >
                Expected Annual Return
              </label>
              <span className="text-sm font-medium tabular-nums text-accent-green">
                {mask(`${returnPct.toFixed(1)}% / yr`)}
              </span>
            </div>
            <input
              id="forecast-return"
              type="range"
              min={2}
              max={15}
              step={0.5}
              value={returnPct}
              onChange={(e) => setReturnPct(Number(e.target.value))}
              className="fey-range mt-4 w-full"
              aria-valuetext={`${returnPct.toFixed(1)} percent per year`}
            />
            <div className="mt-3 flex flex-wrap gap-1.5">
              {RETURN_PRESETS.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setReturnPct(r)}
                  aria-pressed={returnPct === r}
                  className={`${chipBase} ${returnPct === r ? chipActive : chipIdle}`}
                >
                  {r}%
                </button>
              ))}
            </div>
            <p className="mt-2 text-[11px] tracking-tight text-text-muted">
              7% default — conservative index-fund baseline.
            </p>
          </div>

          {/* Time Horizon */}
          <div className="mt-6">
            <div className="flex items-baseline justify-between">
              <span className="text-[11px] uppercase tracking-wider text-text-muted">
                Time Horizon
              </span>
              <span className="text-sm font-medium tabular-nums text-foreground">
                {mask(`${years} years`)}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-4 gap-1.5">
              {HORIZON_PRESETS.map((y) => (
                <button
                  key={y}
                  type="button"
                  onClick={() => setYears(y)}
                  aria-pressed={years === y}
                  className={`${chipBase} text-center ${years === y ? chipActive : chipIdle}`}
                >
                  {y}y
                </button>
              ))}
            </div>
          </div>

          {/* Target Goal Amount */}
          <div className="mt-6">
            <div className="flex items-baseline justify-between">
              <label
                htmlFor="forecast-target"
                className="text-[11px] uppercase tracking-wider text-text-muted"
              >
                Target Goal Amount
              </label>
              <span className="text-sm font-medium tabular-nums text-foreground">
                {mask(`$${usd0(target)}`)}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-4 gap-1.5">
              {TARGET_PRESETS.map((t) => (
                <button
                  key={t.value}
                  type="button"
                  onClick={() => setTarget(t.value)}
                  aria-pressed={target === t.value}
                  className={`${chipBase} text-center ${target === t.value ? chipActive : chipIdle}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <input
              id="forecast-target"
              type="range"
              min={100_000}
              max={5_000_000}
              step={25_000}
              value={target}
              onChange={(e) => setTarget(Number(e.target.value))}
              className="fey-range mt-4 w-full"
              aria-valuetext={`${usdCompact(target)} goal`}
            />
            <p className="mt-2 text-[11px] tracking-tight text-text-muted">
              Drag to reposition the dashed goal line ($100K – $5M).
            </p>
          </div>
        </section>

        {/* Projection summary */}
        <section className="rounded-xl border border-border-muted bg-card p-6">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-foreground">
              Projection Summary
            </span>
            <span className="text-[10px] uppercase tracking-wider text-text-muted">
              {years}y · {returnPct.toFixed(1)}%
            </span>
          </div>
          <dl className="mt-4 flex flex-col gap-3 text-sm tracking-tight">
            <div className="flex items-center justify-between gap-4">
              <dt className="text-text-muted">Projected at {years}y</dt>
              <dd className="font-medium tabular-nums text-foreground">
                {mask(`$${usd0(endValue)}`)}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-text-muted">Growth multiple</dt>
              <dd className="font-medium tabular-nums text-accent-green">
                {mask(`${growthMultiple.toFixed(2)}×`)}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-4">
              <dt className="text-text-muted">Time to target</dt>
              <dd
                className={`font-medium tabular-nums ${
                  targetReached
                    ? "text-foreground"
                    : withinHorizon
                      ? "text-accent-green"
                      : "text-amber-500"
                }`}
              >
                {targetReached
                  ? "Achieved"
                  : withinHorizon
                    ? `${durationLabel(monthsHit)} · ${hitDateLabel(monthsHit)}`
                    : "Not within horizon"}
              </dd>
            </div>
          </dl>
          <p className="mt-4 border-t border-border-muted pt-3 text-[11px] leading-relaxed tracking-tight text-text-muted">
            Pure compounding of current net worth — monthly contributions and
            market drift are modeled separately.
          </p>
        </section>
      </aside>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/*                    Phase 4 — linked-state persistence                      */
/* -------------------------------------------------------------------------- */

/** localStorage bucket for linked dashboard data (zero-baseline otherwise). */
const LINKED_STATE_KEY = "finance-dashboard:linked-state:v1";

/** Phase 7 — recurring bill reminders live in their own bucket. They are
    user-authored commitments, not synced financial data, so they survive a
    registration wipe of the linked-state store and restore independently. */
const BILLS_KEY = "fey-cockpit:bills:v1";

/**
 * The persisted slice: personal holdings and records only — never live
 * market prices, which rehydrate from the local API proxies on every mount.
 */
type PersistedLinkedState = {
  version: 1;
  metrics: PortfolioMetric[];
  ledger: LedgerEntry[];
  connected: ConnectedAccount[];
  cryptoAmounts: Record<string, number>;
  positionHoldings: Record<string, { shares: number; avgCost: number }>;
  cashFlowProfile: CashFlowProfile;
  cashFlowProfileSaved: boolean;
};

/** Finite-number coercion with a fallback (stored JSON is untrusted). */
function finiteOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Shape-check one persisted metric (labels fixed by the cockpit contract). */
function sanitizeMetric(m: unknown): PortfolioMetric | null {
  if (typeof m !== "object" || m === null) return null;
  const e = m as PortfolioMetric;
  if (typeof e.label !== "string" || e.label.trim() === "") return null;
  const metric: PortfolioMetric = {
    label: e.label,
    value: finiteOr(e.value, 0),
    delta: typeof e.delta === "string" ? e.delta : "0.0%",
    positive: e.positive !== false,
  };
  if (e.active === true) metric.active = true;
  return metric;
}

/** Shape-check one persisted ledger entry. */
function sanitizeLedgerEntry(t: unknown): LedgerEntry | null {
  if (typeof t !== "object" || t === null) return null;
  const e = t as LedgerEntry;
  if (
    typeof e.id !== "string" ||
    typeof e.merchant !== "string" ||
    typeof e.amount !== "number" ||
    !Number.isFinite(e.amount) ||
    typeof e.category !== "string" ||
    typeof e.time !== "string" ||
    typeof e.institution !== "string" ||
    typeof e.color !== "string"
  ) {
    return null;
  }
  return e;
}

/** Shape-check one persisted connected-account chip. */
function sanitizeConnected(a: unknown): ConnectedAccount | null {
  if (typeof a !== "object" || a === null) return null;
  const e = a as ConnectedAccount;
  if (
    typeof e.id !== "string" ||
    typeof e.name !== "string" ||
    typeof e.color !== "string" ||
    typeof e.balance !== "number" ||
    !Number.isFinite(e.balance)
  ) {
    return null;
  }
  return e;
}

/** Coerce a persisted cash-flow profile into safe, non-negative numbers. */
function sanitizeCashFlowProfile(p: unknown): CashFlowProfile {
  if (typeof p !== "object" || p === null) {
    return { income: 0, rent: 0, utilities: 0, discretionary: 0 };
  }
  const e = p as CashFlowProfile;
  return {
    income: Math.max(finiteOr(e.income, 0), 0),
    rent: Math.max(finiteOr(e.rent, 0), 0),
    utilities: Math.max(finiteOr(e.utilities, 0), 0),
    discretionary: Math.max(finiteOr(e.discretionary, 0), 0),
  };
}

/* -------------------------------------------------------------------------- */
/*                                Dashboard                                   */
/* -------------------------------------------------------------------------- */

export default function FinanceDashboard() {
  /* ================= App Authentication Screen Gateway ===================
     "login" | "register" intercepts the entire workspace shell behind the
     credentials window; only "authenticated" mounts the cockpit. */
  const [authState, setAuthState] = useState<AuthMode | "authenticated">(
    "login",
  );
  /* First-time users are routed through the onboarding bank-sync gate. */
  const [isNewUser, setIsNewUser] = useState(false);
  /* Display name — captured at registration; seeded for the demo sign-in. */
  const [userName, setUserName] = useState("Brandon");

  /* Global view state — "home" | "finance" | "stocks" | "spending" | "forecast". */
  const [activeView, setActiveView] = useState<DashboardView>("home");

  /* Privacy filter — masks every dollar figure across all views. */
  const [isPrivateMode, setIsPrivateMode] = useState(false);

  /* Phase 5 — identity menu anchored under the header greeting. The
     dropdown is an absolute overlay card owned by the header cluster, so it
     floats over the cockpit without displacing a single pixel of layout. */
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement | null>(null);

  /* Phase 3 — blur-and-focus view transition. Flipped true on any view
     navigation, held for exactly 450ms, then released so the veiled center
     panel dissolves back to full-focus crisp rendering. */
  const [isPageLoading, setIsPageLoading] = useState(false);
  const pageLoadTimeoutRef = useRef<number | null>(null);

  /* Phase 3 — floating right-hand AI advisor sidebar. */
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const [advisorMessages, setAdvisorMessages] = useState<AdvisorMessage[]>([]);
  const [advisorDraft, setAdvisorDraft] = useState("");
  const [advisorThinking, setAdvisorThinking] = useState(false);
  const advisorScrollRef = useRef<HTMLDivElement | null>(null);
  const advisorSeqRef = useRef(0);

  /* ================== Zero-baseline master state (Phase 4) ================
     Every dollar aggregate mounts strictly at $0: no seeded balances, no
     placeholder transactions, no phantom holdings. Linked data arrives only
     through the connect flow or localStorage rehydration. */
  const [metrics, setMetrics] = useState<PortfolioMetric[]>(() =>
    ZERO_METRICS.map((m) => ({ ...m })),
  );
  const [activeMetric, setActiveMetric] = useState("Net Worth");
  const [range, setRange] = useState("1Y");
  const [ledger, setLedger] = useState<LedgerEntry[]>([]);
  const [connected, setConnected] = useState<ConnectedAccount[]>([]);

  /* ============== Phase 6 — books of record vs. live prices ==============
     These two pieces of state are now BOOKS: they hold only what the user
     owns (token amounts, share counts, average cost). They deliberately do
     NOT hold prices. Prices come from the shared market store, and the two
     are merged at render time into the `cryptoHoldings` / `trackedPositions`
     arrays every downstream consumer already reads. Ownership persists to
     localStorage; prices never do, because they are always re-derivable. */
  const [cryptoBook, setCryptoBook] = useState<CryptoHolding[]>(() =>
    ZERO_CRYPTO_HOLDINGS.map((c) => ({ ...c })),
  );

  /* Tracked equity positions — zero shares until linked data lands. */
  const [positionBook, setPositionBook] = useState<TrackedPosition[]>(
    () => TRACKED_UNIVERSE.map((p) => ({ ...p })),
  );

  /* ================= Phase 6 — the shared market price store =============
     One dictionary, owned by <MarketDataProvider> in app/layout.tsx and
     consumed identically by this dashboard, the bottom marquee ticker and
     the ⌘K research terminal. Because all three read the same object in the
     same committed render, NVDA cannot read $178.43 in one surface and
     $178.50 in another. */
  const {
    equities: equityQuotes,
    crypto: cryptoQuotes,
    isHydrating: marketHydrating,
    equityBudgetRemaining,
  } = useMarketData();

  /* Book × live price → the holdings arrays the whole view tree renders.
     A symbol with no quote yet resolves to price 0, which the zero-state
     logic already handles correctly. */
  const cryptoHoldings: CryptoHolding[] = useMemo(
    () =>
      cryptoBook.map((c) => {
        const q = cryptoQuotes[c.token];
        return q
          ? { ...c, price: q.price, deltaPct: q.dayPct }
          : { ...c, price: 0, deltaPct: 0 };
      }),
    [cryptoBook, cryptoQuotes],
  );

  const trackedPositions: TrackedPosition[] = useMemo(
    () =>
      positionBook.map((p) => {
        const q = equityQuotes[p.ticker];
        return q
          ? { ...p, price: q.price, dayPct: q.dayPct }
          : { ...p, price: 0, dayPct: 0 };
      }),
    [positionBook, equityQuotes],
  );

  /* Phase 2 — Personal Cash Flow Input Calculator: overlay visibility plus
     the committed manual profile ("Save" writes here; the Monthly Cash Flow
     card reads it back instantly). Phase 4 — zero-baseline profile. */
  const [isCashFlowModalOpen, setIsCashFlowModalOpen] = useState(false);
  const [cashFlowProfile, setCashFlowProfile] = useState<CashFlowProfile>(
    () => ({ ...ZERO_CASH_FLOW_PROFILE }),
  );
  const [cashFlowProfileSaved, setCashFlowProfileSaved] = useState(false);

  /* Phase 2 — ⌘K research-panel visibility, mirrored from the root layout
     via a window event (the overlay lives above this tree). */
  const [isTerminalOverlayOpen, setIsTerminalOverlayOpen] = useState(false);

  /* Which modal session is open; 0 = closed, N = session id (fresh mount). */
  const [modalSession, setModalSession] = useState(0);
  const openModal = useCallback(() => setModalSession((s) => s + 1), []);
  const closeModal = useCallback(() => setModalSession(0), []);

  /* Phase 3 — append one message to the advisor thread (advisor role by
     default). The auto-scroll effect pins the history to the newest line. */
  const pushAdvisorMessage = useCallback(
    (body: string, role: AdvisorMessage["role"] = "advisor") => {
      advisorSeqRef.current += 1;
      const id = `advisor-${advisorSeqRef.current}`;
      setAdvisorMessages((prev) => [...prev, { id, role, body }]);
    },
    [],
  );

  /* Greeting + date resolved after mount to avoid hydration drift. Phase 7
     also carries the raw Date: the bill calendar needs a real calendar
     anchor, and resolving it here (once, client-side) keeps every
     date-aware surface reading from the same instant. */
  const [clock, setClock] = useState<{
    greeting: string;
    date: string;
    now: Date;
  } | null>(null);
  useEffect(() => {
    /* Deferred one tick so the effect body never sets state synchronously
       (same hydration-safe pattern as the research terminal clock). */
    const tick = () => {
      const now = new Date();
      const h = now.getHours();
      setClock({
        greeting:
          h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening",
        date: now.toLocaleDateString("en-US", {
          weekday: "long",
          month: "long",
          day: "numeric",
        }),
        now,
      });
    };
    const first = window.setTimeout(tick, 0);
    /* Re-resolve hourly so a cockpit left open overnight rolls its greeting
       and its due-date math into the new day. */
    const id = window.setInterval(tick, 60 * 60 * 1000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(id);
    };
  }, []);

  /** Raw calendar anchor — null until the post-mount clock resolves. */
  const clockDate = clock?.now ?? null;

  /* The advisor streams only once the real cockpit is on stage — past the
     auth gateway and the first-run onboarding sync. */
  const cockpitActive = authState === "authenticated" && !isNewUser;

  /* Phase 3 — pin the advisor history to the absolute bottom whenever a
     message lands (streamed lines, event pushes, composer replies). */
  useEffect(() => {
    const el = advisorScrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    /* `advisorThinking` is a dependency so the skeleton is scrolled into
       view the instant it mounts, not one message later. */
  }, [advisorMessages, advisorThinking]);

  /* Phase 5 — a view change re-frames the brief at the TOP of the column, so
     the panel scrolls back up to it rather than stranding the user at the
     bottom of an older thread. */
  useEffect(() => {
    const el = advisorScrollRef.current;
    if (el) el.scrollTo({ top: 0, behavior: "smooth" });
  }, [activeView]);

  /* ============ Phase 7 — persistent session restore (on mount) =========
     The gateway is skipped entirely when a valid token is already in
     localStorage. Three details make this safe:

       1. It runs in an effect, never during render. `authState` therefore
          initialises identically on the server and the first client pass,
          so React's hydration diff stays clean — reading localStorage in a
          useState initialiser is the classic way to get a hydration
          mismatch here.
       2. `readSession` itself is `typeof window` guarded and treats a
          corrupt or expired entry as "no session", deleting it on the way
          out rather than throwing on every subsequent read.
       3. `isNewUser` is explicitly false — a restored session has already
          been through onboarding, so it must never replay the bank-sync
          gate. Only a fresh registration sets that flag.

     `sessionChecked` gates the first paint: until the check resolves we
     render nothing rather than flashing the login wall for one frame and
     then yanking it away. */
  const [sessionChecked, setSessionChecked] = useState(false);
  useEffect(() => {
    const restore = window.setTimeout(() => {
      const session = readSession();
      if (session) {
        if (session.name.trim()) setUserName(session.name.trim());
        setIsNewUser(false);
        setAuthState("authenticated");
      }
      setSessionChecked(true);
    }, 0);
    return () => window.clearTimeout(restore);
  }, []);

  /* ================== Phase 7 — recurring bill reminders ================
     Restored once on mount from their own bucket, then mirrored back on
     every change. Bills are day-of-month recurrences rather than absolute
     dates, so the same list stays correct in every future month. */
  const [bills, setBills] = useState<Bill[]>([]);
  const billsHydratedRef = useRef(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(BILLS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as unknown;
        if (Array.isArray(parsed)) {
          /* Validate every field — a hand-edited or stale store must not be
             able to inject an accent key that has no Tailwind class. */
          const valid = parsed.filter(
            (b): b is Bill =>
              !!b &&
              typeof b === "object" &&
              typeof (b as Bill).id === "string" &&
              typeof (b as Bill).label === "string" &&
              Number.isFinite((b as Bill).day) &&
              (b as Bill).day >= 1 &&
              (b as Bill).day <= 31 &&
              Number.isFinite((b as Bill).amount) &&
              ["green", "violet", "amber", "rose", "sky"].includes(
                (b as Bill).accent,
              ),
          );
          setTimeout(() => {
            setBills(valid);
          }, 0);

      }
      }
    } catch {
      /* Corrupt store — start empty rather than crash the cockpit. */
    } finally {
      billsHydratedRef.current = true;
    }
  }, []);

  useEffect(() => {
    /* Never persist before the restore pass resolves, or the initial empty
       array would overwrite a real saved list. */
    if (!billsHydratedRef.current) return;
    try {
      window.localStorage.setItem(BILLS_KEY, JSON.stringify(bills));
    } catch {
      /* Storage full or blocked — the in-memory list still works. */
    }
  }, [bills]);

  /* Phase 5 — dismiss the identity menu on outside pointer-down or Escape.
     Both listeners are only bound while the menu is actually open. */
  useEffect(() => {
    if (!isProfileMenuOpen) return;
    const onPointerDown = (e: MouseEvent | TouchEvent) => {
      const el = profileMenuRef.current;
      if (el && !el.contains(e.target as Node)) setIsProfileMenuOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsProfileMenuOpen(false);
    };
    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("touchstart", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("touchstart", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isProfileMenuOpen]);

  /* Phase 3 — release a pending blur-and-focus timer on unmount. */
  useEffect(
    () => () => {
      if (pageLoadTimeoutRef.current !== null) {
        window.clearTimeout(pageLoadTimeoutRef.current);
      }
    },
    [],
  );

  /* ============ Phase 4 — linked-state rehydration (sign-in) =============
     Once the cockpit activates, restore any previously linked state from
     localStorage. A missing/invalid store is NOT an error: the zero
     baseline stands until the user links an institution. Registration
     wipes the store instead (see handleAuthenticate), so first-time users
     always mount strictly at $0. */
  const rehydratedRef = useRef(false);
  useEffect(() => {
    if (!cockpitActive || rehydratedRef.current) return;
    /* Deferred one tick (same hydration-safe pattern as the greeting clock)
       so the restore never calls setState synchronously inside the effect
       body, and the persist effect below can gate on `rehydratedRef`. */
    const restore = window.setTimeout(() => {
      rehydratedRef.current = true;

      let raw: string | null = null;
      try {
        raw = window.localStorage.getItem(LINKED_STATE_KEY);
      } catch {
        raw = null; // storage unavailable — zero baseline stands
      }
      if (!raw) return;

      try {
      const parsed = JSON.parse(raw) as PersistedLinkedState;
      if (!parsed || typeof parsed !== "object" || parsed.version !== 1) return;

      if (Array.isArray(parsed.metrics)) {
        const restored = parsed.metrics
          .map(sanitizeMetric)
          .filter((m): m is PortfolioMetric => m !== null);
        const hasAllLabels = ZERO_METRICS.every((z) =>
          restored.some((r) => r.label === z.label),
        );
        if (hasAllLabels) setMetrics(restored);
      }
      if (Array.isArray(parsed.ledger)) {
        setLedger(
          parsed.ledger
            .map(sanitizeLedgerEntry)
            .filter((t): t is LedgerEntry => t !== null),
        );
      }
      if (Array.isArray(parsed.connected)) {
        setConnected(
          parsed.connected
            .map(sanitizeConnected)
            .filter((a): a is ConnectedAccount => a !== null),
        );
      }
      if (parsed.cryptoAmounts && typeof parsed.cryptoAmounts === "object") {
        setCryptoBook((prev) =>
          prev.map((c) => ({
            ...c,
            amount: Math.max(
              finiteOr(parsed.cryptoAmounts?.[c.token], c.amount),
              0,
            ),
          })),
        );
      }
      if (
        parsed.positionHoldings &&
        typeof parsed.positionHoldings === "object"
      ) {
        setPositionBook((prev) =>
          prev.map((p) => {
            const stored = parsed.positionHoldings?.[p.ticker];
            if (!stored) return p;
            return {
              ...p,
              shares: Math.max(finiteOr(stored.shares, p.shares), 0),
              avgCost: Math.max(finiteOr(stored.avgCost, p.avgCost), 0),
            };
          }),
        );
      }
      if (parsed.cashFlowProfile) {
        setCashFlowProfile(sanitizeCashFlowProfile(parsed.cashFlowProfile));
      }
      if (typeof parsed.cashFlowProfileSaved === "boolean") {
        setCashFlowProfileSaved(parsed.cashFlowProfileSaved);
      }
    } catch {
        /* Corrupt store — ignore it and keep the zero baseline. */
      }
    }, 0);
    return () => window.clearTimeout(restore);
  }, [cockpitActive]);

  /* ============ Phase 4 — linked-state persistence (debounced) ===========
     Mirrors the linked slice back to localStorage. Live market prices are
     deliberately excluded (they rehydrate from the API proxies); the Net
     Worth metric is persisted with the current crypto market value
     subtracted so hydration re-adds it exactly once. A JSON signature ref
     skips redundant writes when only prices moved. */
  const lastPersistRef = useRef("");
  useEffect(() => {
    if (authState !== "authenticated") return;
    /* Hold off until the rehydration pass has resolved — otherwise this
       zero-baseline snapshot would overwrite a just-restored linked store. */
    if (!rehydratedRef.current) return;
    const cryptoAmounts: Record<string, number> = {};
    for (const c of cryptoHoldings) cryptoAmounts[c.token] = c.amount;
    const positionHoldings: Record<
      string,
      { shares: number; avgCost: number }
    > = {};
    for (const p of trackedPositions) {
      positionHoldings[p.ticker] = { shares: p.shares, avgCost: p.avgCost };
    }
    const cryptoValueNow = cryptoHoldings.reduce(
      (s, c) => s + c.amount * c.price,
      0,
    );
    const payload: PersistedLinkedState = {
      version: 1,
      metrics: metrics.map((m) =>
        m.label === "Net Worth" ? { ...m, value: m.value - cryptoValueNow } : m,
      ),
      ledger,
      connected,
      cryptoAmounts,
      positionHoldings,
      cashFlowProfile,
      cashFlowProfileSaved,
    };
    const json = JSON.stringify(payload);
    if (json === lastPersistRef.current) return;
    lastPersistRef.current = json;
    try {
      window.localStorage.setItem(LINKED_STATE_KEY, json);
    } catch {
      /* Storage unavailable — the session simply stays in-memory. */
    }
  }, [
    authState,
    metrics,
    ledger,
    connected,
    cryptoHoldings,
    trackedPositions,
    cashFlowProfile,
    cashFlowProfileSaved,
  ]);

  /* ======== Phase 6 — market hydration moved to the shared store =========
     The two local polling loops that used to live here (a 3s `/api/crypto`
     fetch and a 30s `/api/stocks` fetch, each writing prices into local
     state) are gone. They were the source of the cent-drift: the marquee,
     the cards and the research terminal each held their own copy of a price
     captured at their own moment.

     `useMarketData()` now reads ONE dictionary owned by the provider in
     app/layout.tsx. Prices are merged into the holdings books at RENDER
     time (see the derived `cryptoHoldings` / `trackedPositions` memos
     above) rather than copied in through an effect — an effect would add a
     render of lag and quietly reintroduce the drift it was meant to fix. */

  /* Phase 2 — mirror the ⌘K research overlay's open/closed state. The root
     layout broadcasts on every toggle (open, Escape, backdrop-click), so
     this listener stays exactly in sync with the overlay. */
  useEffect(() => {
    const onVisibility = (e: Event) => {
      setIsTerminalOverlayOpen(
        (e as CustomEvent<boolean>).detail === true,
      );
    };
    window.addEventListener(TERMINAL_VISIBILITY_EVENT, onVisibility);
    return () =>
      window.removeEventListener(TERMINAL_VISIBILITY_EVENT, onVisibility);
  }, []);

  /* Rolls realized crypto market moves into the aggregate Net Worth metric:
     each holdings change applies its market-value delta to Net Worth so the
     headline card, metric tickers and finance view recalculate in lockstep
     with the live feed (and the Forecast Engine's start node moves with it).
     The rehydrated baseline intentionally excludes crypto value, so the
     first live quote adds it exactly once. */
  const lastCryptoValueRef = useRef<number | null>(null);
  useEffect(() => {
    const total = cryptoHoldings.reduce((s, c) => s + c.amount * c.price, 0);
    const prev = lastCryptoValueRef.current;
    lastCryptoValueRef.current = total;
    if (prev === null || prev === total) return; // mount / no-op
    const delta = total - prev;
    setMetrics((m) =>
      m.map((x) =>
        x.label === "Net Worth" ? { ...x, value: x.value + delta } : x,
      ),
    );
  }, [cryptoHoldings]);

  /* Monotonic id seed so repeated connections never collide on keys. */
  const idSeq = useRef(0);

  /**
   * Data ingestion hook — commits the newly synced institution into the
   * master dashboard state: balance rolls up into Net Worth (and Cash for
   * depositories), transactions prepend to the chronological ledger.
   */
  const handleInstitutionConnected = useCallback((inst: Institution) => {
    setMetrics((prev) =>
      prev.map((m) => {
        if (m.label === "Net Worth") {
          return { ...m, value: m.value + inst.balance };
        }
        if (m.label === "Cash" && inst.affectsCash) {
          return { ...m, value: m.value + inst.balance };
        }
        return m;
      }),
    );

    const base = idSeq.current;
    idSeq.current += inst.transactions.length + 1;
    setLedger((prev) => [
      ...inst.transactions.map((t, i) => ({
        id: `${inst.id}-${base + i}`,
        merchant: t.merchant,
        amount: t.amount,
        category: t.category,
        time: "Just now",
        institution: inst.name,
        color: inst.color,
      })),
      ...prev,
    ]);

    setConnected((prev) => [
      ...prev,
      { id: `${inst.id}-${base}`, name: inst.name, color: inst.color, balance: inst.balance },
    ]);

    /* Phase 3 — the advisor stays context-aware: new connected account
       records stream straight into the sidebar thread as they land. */
    pushAdvisorMessage(
      `Connected **${inst.name}** — **$${usd0(inst.balance)}** rolled into ${
        inst.affectsCash ? "Net Worth and Cash" : "Net Worth"
      }, with ${inst.transactions.length} transaction${
        inst.transactions.length === 1 ? "" : "s"
      } prepended to the ledger.`,
    );
  }, [pushAdvisorMessage]);

  /* Phase 2 — commit the manual cash-flow profile; the Monthly Cash Flow
     card re-renders from these values on the next paint. Phase 3 — the
     advisor parses the committed values into an event message. */
  const handleSaveCashFlowProfile = useCallback(
    (profile: CashFlowProfile) => {
      setCashFlowProfile(profile);
      setCashFlowProfileSaved(true);
      const outlays = profile.rent + profile.utilities + profile.discretionary;
      const capacity = profile.income - outlays;
      pushAdvisorMessage(
        `Cash flow profile saved — **$${usd0(profile.income)}** income against **$${usd0(
          outlays,
        )}** outlays → net capacity **${capacity >= 0 ? "+" : "−"}$${usd0(
          Math.abs(capacity),
        )}/mo**. The Monthly Cash Flow card now tracks these values.`,
      );
    },
    [pushAdvisorMessage],
  );

  /* Auth gateway resolution — "Create Account" / "Sign In" flips authState
     to "authenticated" and simultaneously initializes the workspace.
     Phase 4 zero-baseline enforcement: a registration wipes the persisted
     linked-state store and resets EVERY aggregate to the pristine $0
     baseline (fresh accounts must never inherit a previous session's
     balances); a sign-in keeps the states as-is — the rehydration effect
     restores the last linked snapshot if one exists, otherwise the cockpit
     stays strictly zeroed. */
  const handleAuthenticate = useCallback(
    (name: string, registered: boolean, email?: string) => {
      if (registered) {
        if (name.trim()) setUserName(name.trim());
        try {
          window.localStorage.removeItem(LINKED_STATE_KEY);
        } catch {
          /* Storage unavailable — nothing to wipe. */
        }
        lastPersistRef.current = "";
        setMetrics(ZERO_METRICS.map((m) => ({ ...m })));
        setLedger([]);
        setConnected([]);
        setCryptoBook(ZERO_CRYPTO_HOLDINGS.map((c) => ({ ...c })));
        setPositionBook(TRACKED_UNIVERSE.map((p) => ({ ...p })));
        setCashFlowProfile({ ...ZERO_CASH_FLOW_PROFILE });
        setCashFlowProfileSaved(false);
        setIsNewUser(true);
      }
      /* Phase 7 — issue the persistent session token. This is what keeps a
         window reload inside the cockpit instead of bouncing the user back
         to the gateway. A sign-in with no name reuses whatever the stored
         session last carried, so the greeting survives a re-auth. */
      const resolvedName = registered && name.trim() ? name.trim() : userName;
      writeSession(resolvedName, email ?? "");

      setAuthState("authenticated");
    },
    [userName],
  );

  /* Onboarding exit — "Skip for now" and "Continue" both land in the
     cockpit; skip leaves the blank state for manual entry later. */
  const handleOnboardingExit = useCallback(() => setIsNewUser(false), []);

  /* -------------------- Phase 7 — bill mutations ---------------------- */
  const handleAddBill = useCallback(
    (draft: Omit<Bill, "id">) => {
      const bill: Bill = {
        ...draft,
        id: `bill-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      };
      setBills((prev) => [...prev, bill]);
      /* Every commitment is announced into the advisor thread immediately —
         the calendar and the sidebar are two views of one ledger, and a
         silent add would leave the advisor reasoning over stale context. */
      pushAdvisorMessage(
        `**Reminder pinned.** ${bill.label}${
          bill.amount > 0 ? ` (**$${usd0(bill.amount)}**)` : ""
        } now recurs on the ${bill.day}${
          bill.day === 1 ? "st" : bill.day === 2 ? "nd" : bill.day === 3 ? "rd" : "th"
        } of every month — folded into your committed monthly outflow.`,
      );
    },
    [pushAdvisorMessage],
  );

  const handleRemoveBill = useCallback(
    (id: string) => {
      setBills((prev) => {
        const target = prev.find((b) => b.id === id);
        if (target) {
          pushAdvisorMessage(
            `**Reminder cleared.** ${target.label} no longer counts toward your recurring obligations.`,
          );
        }
        return prev.filter((b) => b.id !== id);
      });
    },
    [pushAdvisorMessage],
  );

  const billMonthlyTotal = useMemo(
    () => bills.reduce((sum, b) => sum + b.amount, 0),
    [bills],
  );

  /* The nearest liability inside a 14-day window, resolved against the
     CURRENT month's length so a 31st-of-the-month bill still resolves in
     February. */
  const nextBill = useMemo(() => {
    if (!clockDate) return null;
    const year = clockDate.getFullYear();
    const monthIndex = clockDate.getMonth();
    const today = clockDate.getDate();
    const upcoming = bills
      .map((b) => ({ bill: b, day: resolveBillDay(b, year, monthIndex) }))
      .filter((r) => r.day >= today && r.day <= today + 14)
      .sort((a, b) => a.day - b.day)[0];
    if (!upcoming) return null;
    return {
      label: upcoming.bill.label,
      amount: upcoming.bill.amount,
      daysOut: upcoming.day - today,
    };
  }, [bills, clockDate]);

  /* ============ Phase 7 — automated due-date notification logs ==========
     Once the clock and the restored bill list are both resolved, anything
     due within three days is announced into the advisor thread. The ref
     guard makes this fire once per session per bill: without it, every
     re-render of a 2-days-out bill would stack another duplicate line into
     the history. */
  const notifiedBillsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!cockpitActive || !clockDate || !billsHydratedRef.current) return;
    const year = clockDate.getFullYear();
    const monthIndex = clockDate.getMonth();
    const today = clockDate.getDate();

    for (const bill of bills) {
      const day = resolveBillDay(bill, year, monthIndex);
      const daysOut = day - today;
      if (daysOut < 0 || daysOut > 3) continue;
      if (notifiedBillsRef.current.has(bill.id)) continue;
      notifiedBillsRef.current.add(bill.id);
      pushAdvisorMessage(
        `**Due ${daysOut === 0 ? "today" : daysOut === 1 ? "tomorrow" : `in ${daysOut} days`}:** ${bill.label}${
          bill.amount > 0 ? ` — **$${usd0(bill.amount)}**` : ""
        }. Confirm the balance is there before it posts.`,
      );
    }
  }, [bills, clockDate, cockpitActive, pushAdvisorMessage]);

  /* ==================== Phase 5 — session teardown ======================
     "Log Out" in the identity menu drops `authState` straight back to
     "login", which re-arms Gate 1 and re-intercepts the entire workspace
     shell behind the credentials window.

     Deliberately NOT wiped: the persisted linked-state store. Logging out
     is not de-provisioning — signing back in rehydrates the exact snapshot
     (registration is the flow that wipes to a pristine $0 baseline). What
     IS reset is per-session UI: the view returns home, the advisor thread
     clears, every overlay closes, and `rehydratedRef` re-arms so the next
     sign-in performs a fresh restore instead of skipping it. */
  const handleLogOut = useCallback(() => {
    setIsProfileMenuOpen(false);
    /* Drop the persistent token FIRST — if anything below throws, the user
       must still land on the gateway on the next reload rather than being
       silently signed back in. */
    clearSession();
    setAuthState("login");
    setIsNewUser(false);
    setActiveView("home");
    setIsPrivateMode(false);
    setAdvisorMessages([]);
    setAdvisorDraft("");
    setAdvisorThinking(false);
    setModalSession(0);
    setIsCashFlowModalOpen(false);
    setIsPageLoading(false);
    rehydratedRef.current = false;
    lastPersistRef.current = "";
  }, []);

  /* Privacy veil — while private mode is active, wraps any sensitive figure
     (currency, percentage, crypto balance) in a uniform fluid blur instead
     of swapping the text, so layout never shifts. */
  const mask = useCallback(
    (figure: ReactNode) => (
      <span
        className={
          isPrivateMode
            ? "select-none blur-md transition-all duration-300"
            : "transition-all duration-300"
        }
      >
        {figure}
      </span>
    ),
    [isPrivateMode],
  );

  /* Same uniform veil, applied to line-graph point containers. */
  const chartVeil = isPrivateMode
    ? "blur-md transition-all duration-300"
    : "transition-all duration-300";

  /* ------------------------- Derived, data-driven values ------------------ */

  const netWorth = metrics.find((m) => m.label === "Net Worth") ?? metrics[0];
  const cashMetric = metrics.find((m) => m.label === "Cash") ?? metrics[1];

  /* ================= Phase 5 — the single zero-state predicate ============
     Aggregate Net Worth of exactly 0 is the one signal that gates: (a) the
     home hero CTA replacing the historical SVG canvas, (b) the Forecast
     soft-lock, and (c) the advisor's onboarding voice. Kept as one derived
     constant so those three surfaces can never disagree with each other. */
  const isZeroNetWorth = netWorth.value === 0;

  /* Spending tracker — derived strictly from real ledger debits against the
     category budgets. A clear cockpit spends $0 everywhere; linking an
     institution with transactions populates the bars honestly. */
  const expenseCategories: ExpenseCategory[] = useMemo(() => {
    const spentByCategory = new Map<string, number>();
    for (const t of ledger) {
      if (t.amount >= 0) continue;
      spentByCategory.set(
        t.category,
        (spentByCategory.get(t.category) ?? 0) - t.amount,
      );
    }
    const cats: ExpenseCategory[] = SPENDING_CATEGORIES.map((cat) => {
      const spent = spentByCategory.get(cat.name) ?? 0;
      const progress =
        cat.monthlyBudget > 0
          ? Math.round((spent / cat.monthlyBudget) * 100)
          : 0;
      return {
        name: cat.name,
        spent,
        progress,
        negative: progress > 100 || undefined,
      };
    });
    /* Off-budget categories seen in the ledger still report real spend. */
    for (const [name, spent] of spentByCategory) {
      if (!SPENDING_CATEGORIES.some((c) => c.name === name)) {
        cats.push({ name, spent, progress: 0 });
      }
    }
    return cats;
  }, [ledger]);

  const totalSpent = expenseCategories.reduce((s, c) => s + c.spent, 0);
  const overBudgetCats = expenseCategories.filter((c) => c.negative);
  const topCategories = [...expenseCategories]
    .sort((a, b) => b.progress - a.progress)
    .slice(0, 5);

  const cashFlow = ledger.reduce((s, t) => s + t.amount, 0);

  const portfolioValue = trackedPositions.reduce(
    (s, p) => s + p.shares * p.price,
    0,
  );
  const portfolioDayPct =
    trackedPositions.reduce((s, p) => s + p.shares * p.price * p.dayPct, 0) /
    (portfolioValue || 1);

  /* ---- Core financial metrics (all data-driven) -------------------------- */

  /* Monthly Cash Flow: total income minus total spending. Phase 2 — once a
     manual profile has been saved, the Monthly Cash Flow card derives from
     those committed values instead, updating the instant "Save" lands. */
  const monthlyIncome = ledger
    .filter((t) => t.amount > 0)
    .reduce((s, t) => s + t.amount, 0);
  const monthlyExpenses = totalSpent;
  const displayCashFlowIncome = cashFlowProfileSaved
    ? cashFlowProfile.income
    : monthlyIncome;
  const displayCashFlowOutlays = cashFlowProfileSaved
    ? cashFlowProfile.rent +
      cashFlowProfile.utilities +
      cashFlowProfile.discretionary
    : monthlyExpenses;
  const monthlyCashFlow = displayCashFlowIncome - displayCashFlowOutlays;
  const cashFlowPositive = monthlyCashFlow >= 0;

  /* Savings Rate: net income saved or invested, as % of monthly income. */
  const savingsRate =
    monthlyIncome > 0 ? (monthlyCashFlow / monthlyIncome) * 100 : 0;

  /* Emergency Fund Coverage: liquid cash ÷ essential monthly expenses. */
  const monthlyEssentials = expenseCategories
    .filter((c) => ESSENTIAL_CATEGORIES.includes(c.name))
    .reduce((s, c) => s + c.spent, 0);
  const emergencyMonths = cashMetric.value / (monthlyEssentials || 1);

  const creditRating =
    CREDIT_SCORE >= 800
      ? "Exceptional"
      : CREDIT_SCORE >= 740
        ? "Excellent"
        : CREDIT_SCORE >= 670
          ? "Good"
          : CREDIT_SCORE >= 580
            ? "Fair"
            : "Poor";

  /* Crypto hub totals: market value + balance-weighted day move. */
  const cryptoValue = cryptoHoldings.reduce(
    (s, c) => s + c.amount * c.price,
    0,
  );
  const cryptoDayPct =
    cryptoHoldings.reduce((s, c) => s + c.amount * c.price * c.deltaPct, 0) /
    (cryptoValue || 1);

  /* Phase 2/4 — marquee ticker stream: tokens first (BTC, ETH, SOL — live
     from the `/api/crypto` feed), then equities (NVDA, TSLA, AAPL, AMZN —
     live from `/api/stocks`). The bar mounts only on the stocks view or
     while the ⌘K overlay is open. */
  /* ============ Phase 6 — marquee stream from the SHARED store ===========
     The marquee is fed from the exact same `cryptoHoldings` /
     `trackedPositions` arrays the cards render, which are themselves
     derived from the shared price dictionary in the same render pass. There
     is now no code path by which the bar and the cards can disagree: they
     are two projections of one object, committed together. */
  const tickerQuotes: TickerQuote[] = useMemo(
    () => [
      ...cryptoHoldings.map((c) => ({
        symbol: c.token,
        name: c.name,
        price: c.price,
        dayPct: c.deltaPct,
      })),
      ...trackedPositions.map((p) => ({
        symbol: p.ticker,
        name: p.name,
        price: p.price,
        dayPct: p.dayPct,
      })),
    ],
    [cryptoHoldings, trackedPositions],
  );
  const showMarketTicker =
    activeView === "stocks" || isTerminalOverlayOpen;

  /* Phase 3 — stream the context-aware intro brief line-by-line once the
     cockpit activates. Lines are parsed from the live derived state at that
     moment (net worth, cash flow profile, connected accounts, crypto
     tickers), so post-onboarding connections are reflected; later context
     shifts (saves, new links) arrive as event messages instead. */
  /* ---- Equity book segmentation (drives upgrade #5 + the stocks brief) --- */
  const ownedPositions = useMemo(
    () => trackedPositions.filter((p) => p.shares > 0),
    [trackedPositions],
  );
  const watchlistPositions = useMemo(
    () => trackedPositions.filter((p) => p.shares <= 0),
    [trackedPositions],
  );
  /* True when NOTHING is held — the whole equity panel reframes as a
     Market Watchlist and every zero metric is withheld. */
  const isWatchlistOnly = ownedPositions.length === 0;
  const topMover = useMemo(() => {
    const quoted = trackedPositions.filter((p) => p.price > 0);
    if (quoted.length === 0) return null;
    const best = [...quoted].sort(
      (a, b) => Math.abs(b.dayPct) - Math.abs(a.dayPct),
    )[0];
    return { ticker: best.ticker, dayPct: best.dayPct };
  }, [trackedPositions]);

  /* ================ Phase 5 — view-scoped advisor context brief ==========
     The old implementation streamed one global block of zero metrics on
     cockpit activation and never changed again, so every page repeated the
     same "$0 / $0 / $0" wall. The brief is now a pure derivation of live
     state AND `activeView`: switching tabs re-frames the panel instantly
     (onboarding → margins → allocation → compounding), with no interval
     timers, no duplicated history and no stale snapshot. Conversation
     messages (events + composer replies) render underneath it untouched. */
  const advisorBrief = useMemo(() => {
    if (!cockpitActive) return [];
    return buildAdvisorBriefLines(
      {
        netWorthValue: netWorth.value,
        netWorthDelta: netWorth.delta,
        cash: cashMetric.value,
        income: displayCashFlowIncome,
        outlays: displayCashFlowOutlays,
        profileSaved: cashFlowProfileSaved,
        connectedCount: connected.length,
        crypto: cryptoHoldings,
        overBudget: overBudgetCats.map((c) => c.name),
        essentialsMonthly: monthlyEssentials,
        recentLedger: ledger,
        portfolioValue,
        ownedPositions: ownedPositions.length,
        watchlistTickers: watchlistPositions.map((p) => p.ticker),
        topMover,
        totalSpent,
        topCategory: topCategories[0] ?? null,
        billCount: bills.length,
        billMonthlyTotal,
        nextBill,
      },
      activeView,
    );
  }, [
    cockpitActive,
    activeView,
    netWorth.value,
    netWorth.delta,
    cashMetric.value,
    displayCashFlowIncome,
    displayCashFlowOutlays,
    cashFlowProfileSaved,
    connected.length,
    cryptoHoldings,
    overBudgetCats,
    monthlyEssentials,
    ledger,
    portfolioValue,
    ownedPositions.length,
    watchlistPositions,
    topMover,
    totalSpent,
    topCategories,
    bills.length,
    billMonthlyTotal,
    nextBill,
  ]);

  const advisorFrame = ADVISOR_VIEW_FRAME[activeView];

  /* Phase 3 — blur-and-focus navigation handler: flips `isPageLoading` true
     for exactly 450ms on every view change so the active center panel veils
     while the centered ring pulses, then releases back to crisp rendering.
     A ref-held timeout lets rapid clicks restart the window cleanly. */
  const navigate = useCallback(
    (view: DashboardView) => {
      if (view === activeView) return;
      setActiveView(view);
      setIsPageLoading(true);
      if (pageLoadTimeoutRef.current !== null) {
        window.clearTimeout(pageLoadTimeoutRef.current);
      }
      pageLoadTimeoutRef.current = window.setTimeout(() => {
        pageLoadTimeoutRef.current = null;
        setIsPageLoading(false);
      }, 450);
    },
    [activeView],
  );

  /* Phase 4 — offline fallback reply: a local markdown snapshot used only
     when the Gemini engine (`/api/chat`) is unreachable, so the advisor
     never strands the user without a response. */
  const offlineAdvisorReply = (question: string): string => {
    const capacity = displayCashFlowIncome - displayCashFlowOutlays;
    const runway = cashMetric.value / (monthlyEssentials || 1);
    const tokens = cryptoHoldings.map((c) => c.token).join(" · ");
    return `**Advisor offline** — the Gemini engine is unreachable, so here is the local snapshot for "${question}":\nNet worth **$${usd0(
      netWorth.value,
    )}** · cash **$${usd0(cashMetric.value)}** · crypto **$${usd0(
      cryptoValue,
    )}** (${tokens}) · net cash flow **${
      capacity >= 0 ? "+" : "−"
    }$${usd0(Math.abs(capacity))}/mo** · runway **${runway.toFixed(
      1,
    )} months**.\nLive market data keeps streaming in the meantime.`;
  };

  /* Phase 3/4 — composer submit: echo the question as a user bubble, then
     POST to the server-side Gemini engine with the recent thread + a live
     cockpit context payload; the contextual markdown reply streams back
     into the sidebar. Falls back to the local snapshot on failure. */
  const handleAdvisorSend = () => {
    const question = advisorDraft.trim();
    if (!question || advisorThinking) return;
    pushAdvisorMessage(question, "user");
    setAdvisorDraft("");
    setAdvisorThinking(true);

    const history = advisorMessages.slice(-8).map((m) => ({
      role: m.role,
      body: m.body,
    }));
    /* ========== Phase 6 — comprehensive live context payload =============
       The engine can only be as accurate as what it is handed, and its
       system instruction forbids inventing any figure not present here. So
       this payload carries the full picture: the balance sheet, the named
       linked institutions, the manual cash flow profile (and whether it was
       user-saved or ledger-derived), every spending category with its
       budget, the equity book with share counts, and the crypto book. */
    const context = {
      netWorth: netWorth.value,
      cash: cashMetric.value,
      liabilities:
        metrics.find((m) => m.label === "Liabilities")?.value ?? 0,
      monthlyIncome: displayCashFlowIncome,
      monthlyOutlays: displayCashFlowOutlays,
      emergencyMonths,
      cashFlowProfile: {
        income: cashFlowProfile.income,
        rent: cashFlowProfile.rent,
        utilities: cashFlowProfile.utilities,
        discretionary: cashFlowProfile.discretionary,
        saved: cashFlowProfileSaved,
      },
      /* Named institutions — lets the advisor say "your Chase balance"
         instead of "one of your accounts". */
      connectedInstitutions: connected.map((a) => ({
        name: a.name,
        balance: a.balance,
      })),
      connectedCount: connected.length,
      expenseCategories: expenseCategories.map((c) => ({
        name: c.name,
        spent: c.spent,
        budget:
          SPENDING_CATEGORIES.find((s) => s.name === c.name)?.monthlyBudget ?? 0,
        progress: c.progress,
      })),
      crypto: cryptoHoldings.map((c) => ({
        token: c.token,
        amount: c.amount,
        price: c.price,
        deltaPct: c.deltaPct,
      })),
      /* Share counts included so the engine can distinguish a held position
         from a watch-only quote — its instruction depends on that. */
      equities: trackedPositions
        .filter((p) => p.price > 0)
        .map((p) => ({
          symbol: p.ticker,
          shares: p.shares,
          price: p.price,
          dayPct: p.dayPct,
        })),
      ledgerCount: ledger.length,
      /* Phase 7 — committed recurring obligations. Distinct from spending
         history: these are forward commitments the advisor must not
         confuse with money already spent. */
      recurringBills: bills.map((b) => ({
        label: b.label,
        day: b.day,
        amount: b.amount,
      })),
      recurringMonthlyTotal: billMonthlyTotal,
      /* The view the user is looking at while asking. */
      activeView,
    };

    void (async () => {
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question, history, context }),
        });
        const data = (await res.json().catch(() => null)) as {
          reply?: string;
        } | null;
        pushAdvisorMessage(data?.reply ?? offlineAdvisorReply(question));
      } catch {
        pushAdvisorMessage(offlineAdvisorReply(question));
      } finally {
        setAdvisorThinking(false);
      }
    })();
  };

  /* ------------------- Layout routing state machine --------------------- */

  /* Gate 0 — Phase 7 session check. Until `readSession` has run in its
     mount effect we cannot know whether this browser is already signed in,
     so we render a bare canvas for that single frame. Painting the gateway
     first and then tearing it away produces a visible login flash on every
     reload, which is exactly the friction the session manager exists to
     remove. */
  if (!sessionChecked) {
    return (
      <div className="bg-background grid-lines min-h-screen" aria-hidden />
    );
  }

  /* Gate 1 — the App Authentication Screen Gateway intercepts the entire
     workspace shell while credentials have not yet been resolved. */
  if (authState !== "authenticated") {
    return (
      <AuthGateway
        mode={authState}
        onModeChange={setAuthState}
        onAuthenticate={handleAuthenticate}
      />
    );
  }

  /* Gate 2 — first-run onboarding: the Plaid-style multi-institution bank
     sync runs strictly AFTER register/login, before the homepage cockpit. */
  if (isNewUser) {
    return (
      <div className="bg-background text-foreground grid-lines min-h-screen font-sans animate-fey-fade">
        <OnboardingSync
          onInstitutionConnected={handleInstitutionConnected}
          onExit={handleOnboardingExit}
        />
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground font-sans grid-lines">
      {/* =================== CENTER COLUMN — header + views ================== */}
      <div className="flex min-w-0 flex-1 flex-col">
      {/* =====================================================================
          PHASE 7 — the global top navigation bar is GONE.

          Every control it carried has moved into the floating command dock
          rendered at the bottom of this component: view switching, the
          Privacy Veil eye, the advisor panel toggle, Add Account, and the
          identity menu. Removing the header buys back ~60px of vertical
          canvas on every screen and, more importantly, removes the last
          piece of conventional app chrome — which is precisely what makes
          the reference design read as a terminal rather than a web app.
          ===================================================================== */}

      {/* Phase 7 — the center column now owns the full viewport height (no
          header above it) and reserves bottom padding for the floating
          dock, plus extra when the marquee ticker is also mounted. Without
          this reservation the dock would sit on top of a card's last row. */}
      <main
        className={`mx-auto w-full max-w-[1600px] flex-1 overflow-y-auto px-6 pt-8 lg:px-8 ${
          showMarketTicker ? "pb-40" : "pb-28"
        }`}
      >
        {/* Phase 3 — blur-and-focus veil over the active center view panel:
            while `isPageLoading` is true the panel is forced into the smooth
            blur-md / opacity-60 / scale-[0.99] state, then dissolves back to
            full-focus crisp rendering when the 450ms window releases. */}
        <div
          className={`transition-all duration-300 ease-out ${
            isPageLoading
              ? "pointer-events-none scale-[0.99] opacity-60 blur-md"
              : "scale-100 opacity-100 blur-0"
          }`}
        >
        {/* ============================ HOME GRID ============================ */}
        {activeView === "home" && (
          <div className="grid gap-6 lg:grid-cols-3 animate-fey-fade">
            {/* -------- Personal Finance Snapshot (Net Worth center) -------- */}
            <BubbleCard
              onClick={() => navigate("finance")}
              hint="Open Personal Finance"
              ariaLabel="Open the Personal Finance view"
              className="lg:col-span-2 lg:row-span-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  Personal Finance
                </span>
                <span className="text-xs font-medium text-accent-green">
                  {mask(netWorth.delta)}
                </span>
              </div>
              <p className="mt-2 text-4xl font-semibold tabular-nums tracking-tight text-foreground">
                {mask(`$${usd0(netWorth.value)}`)}
              </p>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                Net worth · all linked accounts
              </p>

              {/* ========= PHASE 5 — ZERO-STATE HERO vs. LIVE SPARKLINE =====
                  At a precise $0 aggregate there is no history to draw, and
                  rendering the seeded SVG walk implies performance that does
                  not exist. The entire canvas node is swapped for the
                  "Connect First Portfolio" hero, whose action fires the
                  Plaid sync modal directly (stopPropagation keeps the parent
                  bubble's navigate-to-finance click from swallowing it). */}
              {isZeroNetWorth ? (
                <div
                  className="fey-hero-canvas animate-fey-fade relative mt-4 flex min-h-[140px] flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border-muted px-6 py-8 text-center"
                  onClick={(e) => e.stopPropagation()}
                >
                  <span
                    aria-hidden
                    className="flex h-9 w-9 items-center justify-center rounded-full border border-border-muted bg-card text-accent-green"
                  >
                    <SparkIcon className="h-3.5 w-3.5" />
                  </span>
                  <div>
                    <p className="text-sm font-medium tracking-tight text-foreground">
                      Connect First Portfolio
                    </p>
                    <p className="mx-auto mt-1 max-w-sm text-[11px] leading-relaxed tracking-tight text-text-muted">
                      No history to chart yet. Link an institution with a
                      read-only token and balances, holdings and historical
                      ledger lines populate this canvas instantly.
                    </p>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center justify-center gap-2">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        openModal();
                      }}
                      className="cursor-pointer rounded-lg bg-foreground px-4 py-2 text-xs font-medium tracking-tight text-background transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                    >
                      Connect First Portfolio
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setIsCashFlowModalOpen(true);
                      }}
                      className="cursor-pointer rounded-lg border border-border-muted px-4 py-2 text-xs tracking-tight text-text-muted transition-colors hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                    >
                      Enter manually
                    </button>
                  </div>
                  <p className="text-[10px] tracking-tight text-text-muted">
                    Read-only access · Credentials never stored · 256-bit
                    encryption
                  </p>
                </div>
              ) : (
                /* Multi-line sparkline: Net Worth (white) + Cash (violet) */
                <div className={`relative mt-4 min-h-[140px] flex-1 ${chartVeil}`}>
                  <svg
                    viewBox="0 0 600 180"
                    preserveAspectRatio="none"
                    className="absolute inset-0 h-full w-full"
                    aria-hidden
                  >
                    <path
                      d={CASH_SPARK_PATH}
                      fill="none"
                      stroke="#8b7ec8"
                      strokeWidth="1.25"
                      opacity="0.65"
                      vectorEffect="non-scaling-stroke"
                    />
                    <path
                      d={NET_WORTH_PATH}
                      fill="none"
                      stroke="var(--foreground)"
                      strokeWidth="1.5"
                      opacity="0.9"
                      vectorEffect="non-scaling-stroke"
                    />
                  </svg>
                </div>
              )}

              <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-border-muted pt-4 text-sm tracking-tight">
                {metrics.map((m) => (
                  <div key={m.label} className="flex items-center gap-2">
                    <span
                      className={`inline-block h-4 w-px ${
                        m.positive ? "bg-accent-green/70" : "bg-red-500/70"
                      }`}
                    />
                    <span className="text-text-muted">{m.label}</span>
                    <span className={m.positive ? "text-accent-green" : "text-red-500"}>
                      {mask(m.delta)}
                    </span>
                    <span className="tabular-nums text-foreground">
                      {mask(`$${usd0(m.value)}`)}
                    </span>
                  </div>
                ))}
              </div>
            </BubbleCard>

            {/* ---------- Stock Portfolio Snapshot (tracked modules) -------- */}
            <BubbleCard
              onClick={() => navigate("stocks")}
              hint="Open Research Terminal"
              ariaLabel="Open the Stock Portfolio research terminal"
            >
              {/* ======== PHASE 5 — WATCHLIST vs. HOLDINGS FRAMING =========
                  A "$0 / 0 sh @ $212.17" row reads as a rendering bug, not
                  as data. With zero shares held anywhere, the panel declares
                  itself a Market Watchlist and withholds EVERY position-
                  derived metric (portfolio value, day-weighted P/L, per-row
                  share counts and market values), surfacing only the live
                  quote — which is the one honest number available. Once any
                  position is funded the panel promotes itself to Active
                  Investment Holdings and unowned tickers drop into a
                  secondary watch-only strip. */}
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  {isWatchlistOnly
                    ? "Market Watchlist"
                    : "Active Investment Holdings"}
                </span>
                {isWatchlistOnly ? (
                  <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-text-muted">
                    <span className="anim-pulse-soft inline-block h-1 w-1 rounded-full bg-accent-green" />
                    Live quotes
                  </span>
                ) : (
                  <span
                    className={`text-xs font-medium ${
                      portfolioDayPct >= 0 ? "text-accent-green" : "text-red-500"
                    }`}
                  >
                    {mask(
                      `${portfolioDayPct >= 0 ? "+" : ""}${portfolioDayPct.toFixed(2)}% today`,
                    )}
                  </span>
                )}
              </div>

              {isWatchlistOnly ? (
                <p className="mt-2 text-sm leading-relaxed tracking-tight text-text-muted">
                  Tracking{" "}
                  <span className="font-medium text-foreground">
                    {trackedPositions.length} equities
                  </span>{" "}
                  · no positions held
                </p>
              ) : (
                <>
                  <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                    {mask(`$${usd0(portfolioValue)}`)}
                  </p>
                  <p className="mt-1 text-xs tracking-tight text-text-muted">
                    {ownedPositions.length} held ·{" "}
                    {watchlistPositions.length} watch-only · live quotes
                  </p>
                </>
              )}

              <div className="mt-4 flex flex-col divide-y divide-border-muted/60 border-t border-border-muted pt-2">
                {/* ---- Funded positions: full value + P/L disclosure ---- */}
                {ownedPositions.map((p) => {
                  const value = p.shares * p.price;
                  return (
                    <div
                      key={p.ticker}
                      className="flex items-center justify-between gap-3 py-2.5 text-sm tracking-tight"
                    >
                      <span className="flex min-w-0 items-center gap-2.5">
                        <span
                          className={`inline-block h-4 w-4 flex-shrink-0 rounded-full ${p.color} text-center text-[8px] font-bold leading-4 text-white`}
                        >
                          {p.ticker.charAt(0)}
                        </span>
                        <span className="font-medium text-foreground">
                          {p.ticker}
                        </span>
                        <span className="text-[11px] tabular-nums text-text-muted">
                          {p.shares} sh @{" "}
                          {p.price > 0 ? `$${usd2(p.price)}` : "—"}
                        </span>
                      </span>
                      <span className="flex flex-shrink-0 items-center gap-3">
                        <span
                          className={`text-xs tabular-nums ${
                            p.dayPct >= 0 ? "text-accent-green" : "text-red-500"
                          }`}
                        >
                          {mask(
                            `${p.dayPct >= 0 ? "+" : ""}${p.dayPct.toFixed(2)}%`,
                          )}
                        </span>
                        <span className="tabular-nums text-foreground">
                          {mask(`$${usd0(value)}`)}
                        </span>
                      </span>
                    </div>
                  );
                })}

                {/* ---- Watch-only rows: quote + day move ONLY. No share
                        count, no $0 market value, no weighting. ---- */}
                {watchlistPositions.map((p) => (
                  <div
                    key={p.ticker}
                    className="flex items-center justify-between gap-3 py-2.5 text-sm tracking-tight"
                  >
                    <span className="flex min-w-0 items-center gap-2.5">
                      <span
                        className={`inline-block h-4 w-4 flex-shrink-0 rounded-full ${p.color} text-center text-[8px] font-bold leading-4 text-white`}
                      >
                        {p.ticker.charAt(0)}
                      </span>
                      <span className="font-medium text-foreground">
                        {p.ticker}
                      </span>
                      <span className="hidden truncate text-[11px] tracking-tight text-text-muted sm:inline">
                        {p.name}
                      </span>
                    </span>
                    <span className="flex flex-shrink-0 items-center gap-3">
                      <span
                        className={`text-xs tabular-nums ${
                          p.dayPct >= 0 ? "text-accent-green" : "text-red-500"
                        }`}
                      >
                        {`${p.dayPct >= 0 ? "+" : ""}${p.dayPct.toFixed(2)}%`}
                      </span>
                      <span className="tabular-nums text-foreground">
                        {p.price > 0 ? `$${usd2(p.price)}` : "—"}
                      </span>
                    </span>
                  </div>
                ))}
              </div>

              {/* Watchlist mode gets its own funding affordance rather than
                  leaning on the header "Add Account" button. */}
              {isWatchlistOnly && (
                <div
                  className="mt-4 border-t border-border-muted pt-3"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      openModal();
                    }}
                    className="flex w-full cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-border-muted py-2 text-[11px] tracking-tight text-text-muted transition-colors hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                  >
                    <span
                      aria-hidden
                      className="inline-block h-3.5 w-3.5 rounded-full border border-border-muted text-center text-[9px] leading-[13px]"
                    >
                      +
                    </span>
                    Link a brokerage to convert quotes into holdings
                  </button>
                </div>
              )}
            </BubbleCard>

            {/* ---------- Expense Tracker Snapshot (micro bars) ------------ */}
            <BubbleCard
              onClick={() => navigate("spending")}
              hint="Open Expense Tracker"
              ariaLabel="Open the Expense Tracker view"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  Expense Tracker
                </span>
                {overBudgetCats.length > 0 && (
                  <span className="text-xs font-medium text-red-500">
                    {overBudgetCats.length} over budget
                  </span>
                )}
              </div>
              <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                {mask(`$${usd0(totalSpent)}`)}
              </p>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                Spent this month · {expenseCategories.length} categories
              </p>

              <div className="mt-4 flex flex-col gap-2.5 border-t border-border-muted pt-3">
                {topCategories.map((cat) => (
                  <div
                    key={cat.name}
                    className="flex items-center justify-between text-sm gap-4"
                  >
                    <span className="flex min-w-0 items-center gap-2 text-foreground tracking-tight">
                      <span className="inline-block h-3.5 w-3.5 flex-shrink-0 rounded-full border border-border-muted text-center text-[8px] leading-[12px] text-text-muted">
                        {cat.name.charAt(0)}
                      </span>
                      <span className="truncate">{cat.name}</span>
                    </span>
                    <span className="flex flex-shrink-0 items-center gap-3">
                      <span
                        className={cat.negative ? "text-red-500" : "text-text-muted"}
                      >
                        {mask(`$${usd0(cat.spent)}`)}
                      </span>
                      <TrackingBar progress={cat.progress} negative={cat.negative} />
                    </span>
                  </div>
                ))}
              </div>
            </BubbleCard>

            {/* ---------- Monthly Cash Flow — Phase 2 input calculator ----- */}
            <BubbleCard
              onClick={() => setIsCashFlowModalOpen(true)}
              hint="Configure cash flow inputs"
              ariaLabel="Open the personal cash flow input calculator"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  Monthly Cash Flow
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-tight ${
                    cashFlowPositive
                      ? "border-accent-green/30 bg-accent-green/10 text-accent-green"
                      : "border-red-500/30 bg-red-500/10 text-red-500"
                  }`}
                >
                  {cashFlowPositive ? "Surplus" : "Deficit"}
                </span>
              </div>
              <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                {mask(
                  `${cashFlowPositive ? "+" : "−"}$${usd0(Math.abs(monthlyCashFlow))}`,
                )}
              </p>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                {mask(`$${usd0(displayCashFlowIncome)}`)} income ·{" "}
                {mask(`$${usd0(displayCashFlowOutlays)}`)} spending
              </p>
              {cashFlowProfileSaved && (
                <p className="mt-3 inline-flex items-center gap-1.5 self-start rounded-full border border-border-muted bg-border-muted/40 px-2 py-0.5 text-[10px] tracking-tight text-text-muted">
                  <span className="inline-block h-1 w-1 rounded-full bg-accent-green" />
                  Manual profile active
                </p>
              )}
            </BubbleCard>

            {/* ---------- Savings Rate -------------------------------------- */}
            <BubbleCard
              onClick={() => navigate("finance")}
              hint="View savings breakdown"
              ariaLabel="Open savings rate details in Personal Finance"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  Savings Rate
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-tight ${
                    savingsRate >= 20
                      ? "border-accent-green/30 bg-accent-green/10 text-accent-green"
                      : "border-border-muted bg-border-muted/40 text-text-muted"
                  }`}
                >
                  {savingsRate >= 20 ? "On target" : "Building"}
                </span>
              </div>
              <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                {mask(`${savingsRate.toFixed(1)}%`)}
              </p>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                Net income saved or invested monthly
              </p>
              <div className="relative mt-3 h-px w-full bg-border-muted">
                <div
                  className="absolute left-0 top-0 h-px bg-accent-green transition-all duration-500"
                  style={{
                    width: `${Math.min(Math.max(savingsRate, 0), 100)}%`,
                  }}
                />
              </div>
            </BubbleCard>

            {/* ---------- Emergency Fund Coverage ---------------------------- */}
            <BubbleCard
              onClick={() => navigate("finance")}
              hint="View safety reserves"
              ariaLabel="Open emergency fund coverage in Personal Finance"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  Emergency Fund
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-tight ${
                    emergencyMonths >= 6
                      ? "border-accent-green/30 bg-accent-green/10 text-accent-green"
                      : emergencyMonths >= 3
                        ? "border-border-muted bg-border-muted/40 text-text-muted"
                        : "border-red-500/30 bg-red-500/10 text-red-500"
                  }`}
                >
                  {emergencyMonths >= 6
                    ? "Healthy"
                    : emergencyMonths >= 3
                      ? "Fair"
                      : "Low"}
                </span>
              </div>
              <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                {mask(emergencyMonths.toFixed(1))}{" "}
                <span className="text-sm font-medium text-text-muted">
                  Months Covered
                </span>
              </p>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                {mask(`$${usd0(cashMetric.value)}`)} liquid ·{" "}
                {mask(`$${usd0(monthlyEssentials)}`)} essentials/mo
              </p>
            </BubbleCard>

            {/* ---------- Credit Score Health -------------------------------- */}
            <BubbleCard
              onClick={() => navigate("finance")}
              hint="View borrowing profile"
              ariaLabel="Open credit score details in Personal Finance"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  Credit Score
                </span>
                <span className="text-[10px] font-medium uppercase tracking-wider text-accent-green">
                  {creditRating}
                </span>
              </div>
              <div className="mt-3 flex items-center gap-4">
                <CreditScoreRing score={CREDIT_SCORE} />
                <div className="min-w-0">
                  <p className="text-sm font-medium tracking-tight text-foreground">
                    {mask(String(CREDIT_SCORE))} of {CREDIT_SCORE_MAX}
                  </p>
                  <p className="mt-0.5 text-xs tracking-tight text-text-muted">
                    Borrowing rating monitor
                  </p>
                </div>
              </div>
            </BubbleCard>

            {/* ---------- Crypto Balances Hub -------------------------------- */}
            <BubbleCard
              onClick={() => navigate("stocks")}
              hint="Open assets terminal"
              ariaLabel="Open the crypto balances hub in the research terminal"
              className="lg:col-span-2"
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-text-muted">
                  Crypto Balances
                  <span
                    className="anim-pulse-soft inline-block h-1 w-1 rounded-full bg-accent-green"
                    title="Live prices · hydrated from /api/crypto every 3s"
                  />
                </span>
                <span
                  className={`text-xs font-medium ${
                    cryptoDayPct >= 0 ? "text-accent-green" : "text-red-500"
                  }`}
                >
                  {mask(
                    `${cryptoDayPct >= 0 ? "+" : ""}${cryptoDayPct.toFixed(2)}% today`,
                  )}
                </span>
              </div>
              <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
                {mask(`$${usd0(cryptoValue)}`)}
              </p>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                {cryptoHoldings.length} tokens · wallet + exchange
              </p>

              <div className="mt-4 grid gap-4 border-t border-border-muted pt-3 sm:grid-cols-3">
                {cryptoHoldings.map((c) => {
                  const value = c.amount * c.price;
                  return (
                    <div
                      key={c.token}
                      className="flex flex-col gap-1 text-sm tracking-tight"
                    >
                      <span className="flex items-center gap-2.5">
                        <span
                          className={`inline-block h-4 w-4 flex-shrink-0 rounded-full ${c.color} text-center text-[8px] font-bold leading-4 text-black`}
                        >
                          {c.token.charAt(0)}
                        </span>
                        <span className="font-medium text-foreground">
                          {c.token}
                        </span>
                        <span className="text-[11px] text-text-muted">
                          {c.name}
                        </span>
                      </span>
                      <span className="flex items-center gap-3">
                        <span
                          className={`text-xs tabular-nums ${
                            c.deltaPct >= 0 ? "text-accent-green" : "text-red-500"
                          }`}
                        >
                          {mask(
                            `${c.deltaPct >= 0 ? "+" : ""}${c.deltaPct.toFixed(2)}%`,
                          )}
                        </span>
                        <span className="tabular-nums text-foreground">
                          {mask(`$${usd0(value)}`)}
                        </span>
                      </span>
                    </div>
                  );
                })}
              </div>
            </BubbleCard>
          </div>
        )}

        {/* ========================= FINANCE VIEW ========================== */}
        {activeView === "finance" && (
          <div className="flex flex-col gap-6 lg:flex-row animate-fey-fade">
            {/* -------- Main Net Worth card -------- */}
            <section className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-border-muted bg-card">
              <div className="p-6 pb-4">
                <div className="mb-1 text-sm tracking-tight text-text-muted">
                  {clock ? clock.date : "—"}
                </div>
                <div className="flex flex-wrap items-baseline gap-3">
                  <h2 className="text-3xl font-semibold tracking-tight text-foreground">
                    Personal finance
                  </h2>
                  <span className="text-sm font-medium text-accent-green">
                    {mask(netWorth.delta)}
                  </span>
                </div>
                <p className="mt-2 text-4xl font-semibold tabular-nums tracking-tight text-foreground">
                  {mask(`$${usd0(netWorth.value)}`)}
                </p>
              </div>

              {/* Net Worth chart — Phase 5 applies the same zero-state rule
                  as the home canvas: no seeded historical walk at a $0
                  aggregate, since a rising line with no data behind it is
                  the single most misleading pixel in the cockpit. */}
              {isZeroNetWorth ? (
                <div className="mx-6 flex min-h-[180px] flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border-muted px-6 py-8 text-center">
                  <p className="text-sm font-medium tracking-tight text-foreground">
                    No trailing history yet
                  </p>
                  <p className="max-w-sm text-[11px] leading-relaxed tracking-tight text-text-muted">
                    Net worth is charted from linked balances. Connect an
                    institution and the trailing-year curve renders from real
                    historical data.
                  </p>
                  <button
                    type="button"
                    onClick={openModal}
                    className="mt-1 cursor-pointer rounded-lg bg-foreground px-4 py-2 text-xs font-medium tracking-tight text-background transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                  >
                    Connect First Portfolio
                  </button>
                </div>
              ) : (
                <div className={`relative min-h-[180px] flex-1 px-6 ${chartVeil}`}>
                  <svg
                    viewBox="0 0 600 180"
                    preserveAspectRatio="none"
                    className="absolute inset-x-6 bottom-8 top-4 h-[calc(100%-3rem)] w-[calc(100%-3rem)]"
                    aria-hidden
                  >
                    <path
                      d={NET_WORTH_PATH}
                      fill="none"
                      stroke="var(--foreground)"
                      strokeWidth="1.5"
                      opacity="0.9"
                    />
                  </svg>
                  <div className="absolute right-8 top-4 text-xs tracking-tight text-accent-green">
                    {mask(netWorth.delta)}
                  </div>
                  <div className="absolute bottom-2 left-6 text-xs tracking-tight text-text-muted">
                    Net worth · trailing year
                  </div>
                </div>
              )}

              {/* Cash flow sparkline — derived from real ledger activity */}
              <div className="relative h-24 px-6 pb-4 pt-2">
                <div
                  className={`absolute inset-x-6 top-2 w-[calc(100%-3rem)] ${chartVeil}`}
                >
                  <CashFlowSparkline
                    amounts={ledger.slice(0, 12).map((t) => t.amount)}
                  />
                </div>
                <div className="absolute right-8 top-2 text-xs tabular-nums tracking-tight text-[#8b7ec8]">
                  {mask(signedUsd2(cashFlow))}
                </div>
                <div className="absolute bottom-0 left-6 text-xs tracking-tight text-text-muted">
                  Cash flow · recent ledger activity
                </div>
              </div>

              {/* Metric tickers + range controls */}
              <div className="flex flex-wrap items-center gap-4 px-6 pb-4 text-sm tracking-tight">
                {metrics.map((m) => (
                  <div key={m.label} className="flex items-center gap-2">
                    <span
                      className={`inline-block h-4 w-px ${
                        activeMetric === m.label
                          ? "bg-foreground"
                          : m.positive
                            ? "bg-accent-green/70"
                            : "bg-red-500/70"
                      }`}
                    />
                    <span
                      className={`cursor-pointer ${
                        activeMetric === m.label
                          ? "font-medium text-foreground"
                          : "text-text-muted hover:text-foreground"
                      }`}
                      onClick={() => setActiveMetric(m.label)}
                    >
                      {m.label}
                    </span>
                    <span className={m.positive ? "text-accent-green" : "text-red-500"}>
                      {mask(m.delta)}
                    </span>
                    <span className="tabular-nums text-foreground">
                      {mask(`$${usd0(m.value)}`)}
                    </span>
                  </div>
                ))}
                <AddAction label="Connect Portfolio" onClick={openModal} />
                <div className="ml-auto flex gap-1 text-sm">
                  {ranges.map((r) => (
                    <button
                      key={r}
                      onClick={() => setRange(r)}
                      className={`cursor-pointer rounded-md px-2 py-0.5 tracking-tight ${
                        r === range
                          ? "bg-border-muted/60 text-foreground"
                          : "text-text-muted hover:text-foreground"
                      }`}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>

              {/* Connected institutions — hydrated by the sync flow */}
              {connected.length > 0 && (
                <div className="flex flex-wrap items-center gap-2 px-6 pb-4">
                  <span className="text-[11px] uppercase tracking-wider text-text-muted">
                    Connected
                  </span>
                  {connected.map((acc) => (
                    <span
                      key={acc.id}
                      className="inline-flex items-center gap-1.5 rounded-full border border-border-muted bg-border-muted/40 py-0.5 pl-1 pr-2.5"
                    >
                      <span
                        className={`inline-block h-3.5 w-3.5 rounded-full ${acc.color} text-center text-[7px] font-bold leading-[14px] text-white`}
                      >
                        {acc.name.charAt(0)}
                      </span>
                      <span className="text-[11px] tracking-tight text-foreground">
                        {acc.name}
                      </span>
                      <span className="text-[11px] tabular-nums text-accent-green">
                        {mask(`$${usd0(acc.balance)}`)}
                      </span>
                    </span>
                  ))}
                </div>
              )}
            </section>

            {/* -------- Ledger column (brief moved to AI advisor sidebar) --- */}
            <aside className="flex min-w-0 flex-1 flex-col gap-4">
              <div className="rounded-xl border border-border-muted bg-card p-6">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-xs font-medium uppercase tracking-wider text-foreground">
                    Ledger
                  </span>
                  <span className="text-xs tracking-tight text-text-muted">
                    {ledger.length} transactions
                  </span>
                </div>
                <ul className="flex flex-col divide-y divide-border-muted/60">
                  {ledger.length === 0 ? (
                    /* Phase 4 — elegant clear-cockpit empty state. */
                    <li className="flex flex-col items-center gap-2 py-10 text-center">
                      <span className="flex h-7 w-7 items-center justify-center rounded-full border border-border-muted text-accent-green">
                        <SparkIcon className="h-3 w-3" />
                      </span>
                      <p className="text-xs tracking-tight text-foreground">
                        Your cockpit is clear.
                      </p>
                      <p className="text-[11px] tracking-tight text-text-muted">
                        Link an account or search terminal to begin.
                      </p>
                    </li>
                  ) : (
                    ledger.slice(0, 8).map((t) => (
                    <li
                      key={t.id}
                      className="flex items-center justify-between gap-4 py-2.5 first:pt-0 last:pb-0"
                    >
                      <span className="flex min-w-0 items-center gap-2.5">
                        <span
                          className={`inline-block h-4 w-4 flex-shrink-0 rounded-full ${t.color} text-center text-[8px] font-bold leading-4 text-white`}
                        >
                          {t.institution.charAt(0)}
                        </span>
                        <span className="truncate text-sm tracking-tight text-foreground">
                          {t.merchant}
                        </span>
                        <span className="hidden flex-shrink-0 text-[11px] text-text-muted sm:inline">
                          {t.category}
                        </span>
                      </span>
                      <span className="flex flex-shrink-0 items-center gap-3">
                        <span className="text-[11px] tracking-tight text-text-muted">
                          {t.time}
                        </span>
                        <span
                          className={`text-sm tabular-nums ${
                            t.amount >= 0 ? "text-accent-green" : "text-foreground"
                          }`}
                        >
                          {mask(signedUsd2(t.amount))}
                        </span>
                      </span>
                    </li>
                    ))
                  )}
                </ul>
              </div>

              {/* Phase 3 — the standalone Assistant brief card was removed;
                  its context-aware brief now streams in the floating AI
                  Advisor sidebar on the right-hand side. */}
            </aside>
          </div>
        )}

        {/* ========================== STOCKS VIEW ========================== */}
        {activeView === "stocks" && (
          /* Dedicated single-equity research center. Privacy mode blurs the
             whole panel — holdings inside are personal data. */
          <div
            className={`animate-fey-fade ${
              isPrivateMode
                ? "select-none blur-md transition-all duration-300"
                : "transition-all duration-300"
            }`}
            aria-hidden={isPrivateMode}
          >
            <StockResearchTerminal variant="page" />
          </div>
        )}

        {/* ========================= FORECAST VIEW ========================= */}
        {activeView === "forecast" && (
          /* Wealth Forecasting Engine — 60/40 canvas: compounding projection
             chart + interactive Simulation Control Center.

             PHASE 5 — SOFT LOCK: compounding is `FV = PV × (1 + r/12)^(12t)`.
             With PV = 0 every horizon, every return preset and every target
             resolves to $0 / 0.00× / "Not within horizon" — technically
             correct, but it reads as a broken tool. At a precise $0 net
             worth the whole canvas is gated behind a translucent lock: the
             engine still renders underneath (blurred, inert, aria-hidden) so
             the user sees exactly what they are unlocking, while the banner
             carries the two real paths forward. */
          <div className="relative animate-fey-fade">
            <div
              className={`grid gap-6 lg:grid-cols-5 ${
                isZeroNetWorth
                  ? "pointer-events-none select-none blur-[6px] opacity-40 transition-all duration-500"
                  : "transition-all duration-500"
              }`}
              aria-hidden={isZeroNetWorth || undefined}
            >
              <WealthForecastEngine
                netWorth={netWorth.value}
                mask={mask}
                chartVeil={chartVeil}
              />
            </div>

            {isZeroNetWorth && (
              <div className="absolute inset-0 z-20 flex items-center justify-center rounded-xl bg-background/50 p-6 backdrop-blur-[2px]">
                <div className="fey-lock-banner w-full max-w-md rounded-xl border border-border-muted bg-card/95 p-6 text-center shadow-2xl">
                  <span
                    aria-hidden
                    className="mx-auto flex h-10 w-10 items-center justify-center rounded-full border border-border-muted bg-background/60 text-text-muted"
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-4 w-4"
                    >
                      <rect x="4" y="11" width="16" height="9" rx="2" />
                      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
                    </svg>
                  </span>
                  <h2 className="mt-4 text-base font-semibold tracking-tight text-foreground">
                    Unlock Forecasting Engine
                  </h2>
                  <p className="mx-auto mt-2 max-w-sm text-xs leading-relaxed tracking-tight text-text-muted">
                    Sync a portfolio balance or use manual entry sliders to
                    begin simulating future wealth trends.
                  </p>
                  <p className="mx-auto mt-3 max-w-sm text-[11px] leading-relaxed tracking-tight text-text-muted">
                    Compounding multiplies a principal — anchored at{" "}
                    <span className="tabular-nums text-foreground">$0</span>,
                    every horizon mathematically returns zero.
                  </p>
                  <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
                    <button
                      type="button"
                      onClick={openModal}
                      className="cursor-pointer rounded-lg bg-foreground px-4 py-2 text-xs font-medium tracking-tight text-background transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                    >
                      Sync a portfolio balance
                    </button>
                    <button
                      type="button"
                      onClick={() => setIsCashFlowModalOpen(true)}
                      className="cursor-pointer rounded-lg border border-border-muted px-4 py-2 text-xs tracking-tight text-text-muted transition-colors hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
                    >
                      Use manual entry sliders
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================= SPENDING VIEW ========================= */}
        {/* ========================= CALENDAR VIEW ========================= */}
        {activeView === "calendar" && (
          <BillCalendar
            bills={bills}
            onAddBill={handleAddBill}
            onRemoveBill={handleRemoveBill}
            mask={mask}
            today={clockDate}
          />
        )}

        {activeView === "spending" && (
          <section className="animate-fey-fade rounded-xl border border-border-muted bg-card p-6">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <div>
                <h2 className="text-2xl font-semibold tracking-tight text-foreground">
                  Spending
                </h2>
                <p className="mt-1 text-xs tracking-tight text-text-muted">
                  This month · {expenseCategories.length} categories ·{" "}
                  {mask(`$${usd0(totalSpent)}`)} total
                </p>
              </div>
              {overBudgetCats.length > 0 && (
                <span className="rounded-md border border-red-500/30 bg-red-500/10 px-2 py-1 text-xs font-medium text-red-500">
                  {overBudgetCats.map((c) => c.name).join(", ")} over budget
                </span>
              )}
            </div>

            <div className="mt-6 grid gap-x-10 gap-y-1 sm:grid-cols-2">
              {expenseCategories.map((cat) => (
                <div
                  key={cat.name}
                  className="flex items-center justify-between gap-4 border-b border-border-muted/60 py-3 text-sm last:border-b-0"
                >
                  <span className="flex min-w-0 items-center gap-2.5 text-foreground tracking-tight">
                    <span className="inline-block h-4 w-4 flex-shrink-0 rounded-full border border-border-muted text-center text-[8px] leading-[14px] text-text-muted">
                      {cat.name.charAt(0)}
                    </span>
                    <span className="truncate">{cat.name}</span>
                  </span>
                  <span className="flex flex-shrink-0 items-center gap-3">
                    <span
                      className={`text-xs tabular-nums ${
                        cat.negative ? "text-red-500" : "text-text-muted"
                      }`}
                    >
                      {mask(`${cat.progress}% used`)}
                    </span>
                    <TrackingBar progress={cat.progress} negative={cat.negative} />
                    <span className="w-14 text-right tabular-nums text-foreground">
                      {mask(`$${usd0(cat.spent)}`)}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}
        </div>{/* /center-view blur-and-focus veil */}
      </main>
      </div>{/* /center column */}

      {/* ============ PHASE 3 — FLOATING AI ADVISOR SIDEBAR =============
          Dedicated right-hand column anchored beside the cockpit. The
          header trigger slides it fully open / collapsed: the outer rail
          collapses its width while the inner panel translates out with
          transition-transform duration-300. Phase 4 — composer questions
          are answered by the server-side Gemini engine via `/api/chat`. */}
      <aside
        id="ai-advisor-sidebar"
        aria-label="AI advisor panel"
        aria-hidden={!isSidebarExpanded || undefined}
        className={`w-80 h-full hidden lg:flex flex-col flex-shrink-0 overflow-hidden border-l border-border-muted bg-card/20 backdrop-blur-sm z-30 transition-all duration-300 ease-out ${
          isSidebarExpanded ? "opacity-100" : "w-0 border-l-0 opacity-0"
        }`}
      >
        <div
          className={`flex h-full w-80 flex-col transition-transform duration-300 ease-out ${
            isSidebarExpanded ? "translate-x-0" : "translate-x-full"
          }`}
        >
          {/* Panel header */}
          <div className="flex flex-shrink-0 items-center justify-between border-b border-border-muted px-4 py-3">
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-foreground">
                <span className="flex h-5 w-5 items-center justify-center rounded-full border border-border-muted bg-card text-accent-green">
                  <SparkIcon className="h-2.5 w-2.5" />
                </span>
                AI Advisor
              </span>
              {/* Phase 5 — the panel declares which lens it is reasoning
                  through, so a context switch is legible, not implicit. */}
              <span className="truncate pl-7 text-[10px] tracking-tight text-text-muted">
                {advisorFrame.scope}
              </span>
            </span>
            <button
              type="button"
              onClick={() => setIsSidebarExpanded(false)}
              aria-label="Collapse AI advisor panel"
              title="Collapse panel"
              className="rounded-md p-1 text-text-muted transition-colors hover:bg-foreground/5 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
            >
              <ChevronRightIcon />
            </button>
          </div>

          {/* Scrollable messaging bubble history — auto-scrolls to the
              absolute bottom on every addition, and every currency/percent
              figure streams behind the mask() veil while Privacy Mode is
              active. */}
          <div
            ref={advisorScrollRef}
            className="fey-scroll flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-4"
          >
            {/* ---- Phase 5 · view-scoped context brief -------------------
                Re-derived (not re-streamed) on every `activeView` change.
                Keying on the view remounts the block so the cascade replays
                as a stream, while the conversation thread below survives
                the switch untouched. */}
            {advisorBrief.length === 0 ? (
              <p className="mt-8 text-center text-[11px] leading-relaxed tracking-tight text-text-muted">
                Parsing your cash flow profile, connected accounts and live
                crypto tickers…
              </p>
            ) : (
              <div key={activeView} className="flex flex-col gap-3">
                <span className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-text-muted">
                  <span className="h-px flex-1 bg-border-muted" />
                  {advisorFrame.label} brief
                  <span className="h-px flex-1 bg-border-muted" />
                </span>
                {advisorBrief.map((line, i) => (
                  <AdvisorBubble
                    key={`${activeView}-brief-${i}`}
                    role="advisor"
                    body={line}
                    mask={mask}
                    delayMs={i * 90}
                  />
                ))}
              </div>
            )}

            {/* ---- Event + conversation thread (persists across views) ---- */}
            {(advisorMessages.length > 0 || advisorThinking) && (
              <div className="flex flex-col gap-3">
                <span className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-text-muted">
                  <span className="h-px flex-1 bg-border-muted" />
                  Thread
                  <span className="h-px flex-1 bg-border-muted" />
                </span>
                {advisorMessages.map((m) => (
                  <AdvisorBubble
                    key={m.id}
                    role={m.role}
                    body={m.body}
                    mask={mask}
                  />
                ))}
                {/* Phase 6 — skeleton holds the reply's shape while the
                    server-side Gemini round trip is in flight. */}
                {advisorThinking && <AdvisorSkeleton />}
              </div>
            )}
          </div>

          {/* Composer */}
          <form
            className="flex-shrink-0 border-t border-border-muted p-3"
            onSubmit={(e) => {
              e.preventDefault();
              handleAdvisorSend();
            }}
          >
            <div className="flex items-center gap-2 rounded-lg border border-border-muted bg-card px-2.5 py-1.5 transition-colors focus-within:border-foreground/30">
              <input
                value={advisorDraft}
                onChange={(e) => setAdvisorDraft(e.target.value)}
                placeholder="Ask about cash flow, crypto, budgets…"
                aria-label="Message the AI advisor"
                className="min-w-0 flex-1 bg-transparent text-xs tracking-tight text-foreground placeholder:text-text-muted focus:outline-none"
              />
              <button
                type="submit"
                disabled={!advisorDraft.trim() || advisorThinking}
                aria-label={advisorThinking ? "Advisor is thinking" : "Send message"}
                className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md border border-border-muted text-text-muted transition-colors enabled:hover:border-foreground/40 enabled:hover:text-foreground disabled:opacity-40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
              >
                {advisorThinking ? (
                  <span className="anim-pulse-soft inline-block h-2 w-2 rounded-full bg-accent-green" />
                ) : (
                  <SendIcon />
                )}
              </button>
            </div>
            {/* Phase 6 — feed transparency. The budget readout is not
                decoration: free equity tiers are hard-capped, and a user
                watching prices freeze deserves to know it's a quota wall
                rather than a broken app. */}
            <p className="mt-2 flex items-center gap-1.5 text-[10px] tracking-tight text-text-muted">
              <span
                className={`inline-block h-1 w-1 flex-shrink-0 rounded-full ${
                  marketHydrating
                    ? "anim-pulse-soft bg-text-muted"
                    : "bg-accent-green"
                }`}
              />
              {marketHydrating ? "Connecting market feeds" : "Live feeds"} ·
              Powered by Gemini
              {equityBudgetRemaining !== null && equityBudgetRemaining <= 5
                ? ` · ${equityBudgetRemaining} equity calls left today`
                : ""}
            </p>
          </form>
        </div>
      </aside>

      {/* Phase 3 — blur-and-focus transition ring: a minimal geometric ring
          pulsing (animate-pulse) dead-center of the viewport during the
          450ms window, dissolving as the veiled panel refocuses crisp. */}
      {isPageLoading && (
        <div
          role="status"
          aria-label="Loading view"
          className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center"
        >
          <span className="relative flex items-center justify-center">
            <span className="animate-pulse absolute h-12 w-12 rounded-full border border-foreground/20" />
            <span className="animate-pulse block h-8 w-8 rounded-full border-[1.5px] border-foreground/70" />
          </span>
        </div>
      )}

      {/* =====================================================================
          PHASE 7 — FLOATING COMMAND DOCK

          Fixed, centered, pill-shaped, glass. It replaces the top header
          entirely and carries every global control. `bottom-6` clears the
          marquee ticker when that is mounted (the ticker sits at the
          viewport floor), and the main content column reserves matching
          bottom padding so the dock never occludes a card's final row.
          ===================================================================== */}
      <nav
        aria-label="Primary navigation"
        className="bg-card/60 backdrop-blur-xl border border-border-muted px-4 py-2 rounded-full fixed bottom-6 left-1/2 -translate-x-1/2 shadow-2xl flex items-center gap-6 z-50"
      >
        {/* ---- Four primary view states ---- */}
        <div className="flex items-center gap-1">
          {DOCK_PRIMARY.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => navigate(item.key)}
              aria-current={activeView === item.key ? "page" : undefined}
              aria-label={item.label}
              title={item.label}
              className={`fey-dock-btn flex h-9 w-9 cursor-pointer items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                activeView === item.key
                  ? "bg-foreground/10 text-foreground"
                  : "text-text-muted hover:bg-foreground/5 hover:text-foreground"
              }`}
            >
              <DockIcon name={item.icon} />
              <span className="fey-dock-tip">{item.label}</span>
            </button>
          ))}
        </div>

        <span aria-hidden className="h-5 w-px bg-border-muted" />

        {/* ---- Secondary surfaces ---- */}
        <div className="flex items-center gap-1">
          {DOCK_SECONDARY.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => navigate(item.key)}
              aria-current={activeView === item.key ? "page" : undefined}
              aria-label={item.label}
              title={item.label}
              className={`fey-dock-btn flex h-9 w-9 cursor-pointer items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                activeView === item.key
                  ? "bg-foreground/10 text-foreground"
                  : "text-text-muted hover:bg-foreground/5 hover:text-foreground"
              }`}
            >
              <DockIcon name={item.icon} />
              <span className="fey-dock-tip">{item.label}</span>
            </button>
          ))}
        </div>

        <span aria-hidden className="h-5 w-px bg-border-muted" />

        {/* ---- Global toggles: Privacy Veil, advisor panel, add account ---- */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setIsPrivateMode((p) => !p)}
            aria-pressed={isPrivateMode}
            aria-label={
              isPrivateMode ? "Show account balances" : "Hide account balances"
            }
            title={isPrivateMode ? "Show balances" : "Hide balances"}
            className={`fey-dock-btn flex h-9 w-9 cursor-pointer items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
              isPrivateMode
                ? "bg-foreground/10 text-foreground"
                : "text-text-muted hover:bg-foreground/5 hover:text-foreground"
            }`}
          >
            <EyeIcon closed={isPrivateMode} />
            <span className="fey-dock-tip">
              {isPrivateMode ? "Show balances" : "Hide balances"}
            </span>
          </button>

          <button
            type="button"
            onClick={() => setIsSidebarExpanded((v) => !v)}
            aria-pressed={isSidebarExpanded}
            aria-controls="ai-advisor-sidebar"
            aria-label={
              isSidebarExpanded
                ? "Collapse AI advisor panel"
                : "Expand AI advisor panel"
            }
            title={isSidebarExpanded ? "Collapse advisor" : "Expand advisor"}
            className={`fey-dock-btn hidden h-9 w-9 cursor-pointer items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 lg:flex ${
              isSidebarExpanded
                ? "bg-foreground/10 text-foreground"
                : "text-text-muted hover:bg-foreground/5 hover:text-foreground"
            }`}
          >
            <AdvisorPanelIcon open={isSidebarExpanded} />
            <span className="fey-dock-tip">AI Advisor</span>
          </button>

          <button
            type="button"
            onClick={openModal}
            aria-label="Add account"
            title="Add account"
            className="fey-dock-btn flex h-9 w-9 cursor-pointer items-center justify-center rounded-full text-text-muted transition-colors hover:bg-foreground/5 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              aria-hidden
              className="h-[18px] w-[18px]"
            >
              <circle cx="12" cy="12" r="8.5" />
              <path d="M12 8.5v7M8.5 12h7" />
            </svg>
            <span className="fey-dock-tip">Add account</span>
          </button>
        </div>

        <span aria-hidden className="h-5 w-px bg-border-muted" />

        {/* ---- Identity pill + settings menu, nested in the dock ---- */}
        <div ref={profileMenuRef} className="relative">
          <button
            type="button"
            onClick={() => setIsProfileMenuOpen((o) => !o)}
            aria-haspopup="menu"
            aria-expanded={isProfileMenuOpen}
            aria-label="Open account menu"
            className={`flex cursor-pointer items-center gap-2 rounded-full py-1 pl-1 pr-2.5 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
              isProfileMenuOpen
                ? "bg-foreground/10"
                : "hover:bg-foreground/5"
            }`}
          >
            <span
              aria-hidden
              className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border border-border-muted bg-background text-[11px] font-bold uppercase text-foreground"
            >
              {userName.charAt(0)}
            </span>
            <span className="hidden max-w-[7rem] truncate text-xs tracking-tight text-foreground sm:inline">
              {userName}
            </span>
          </button>

          {isProfileMenuOpen && (
            /* Opens UPWARD (bottom-full) — the dock is at the viewport
               floor, so a downward menu would render off-screen. */
            <div
              role="menu"
              aria-label="Account menu"
              className="terminal-overlay-card absolute bottom-full right-0 z-50 mb-3 w-48 rounded-xl border border-border-muted bg-card p-2 shadow-2xl"
            >
              <div className="border-b border-border-muted px-2 pb-2">
                <p className="truncate text-xs font-medium tracking-tight text-foreground">
                  {userName}
                </p>
                <p className="mt-0.5 text-[10px] tracking-tight text-text-muted">
                  {connected.length} linked ·{" "}
                  {isZeroNetWorth ? "$0 baseline" : "Synced"}
                </p>
              </div>

              <div className="pt-1">
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setIsProfileMenuOpen(false);
                    /* Privacy Veil is the one live setting in the shell —
                       the menu exposes it directly rather than routing to
                       a settings surface that does not exist yet. */
                    setIsPrivateMode((p) => !p);
                  }}
                  className={menuItemCls}
                >
                  <span>Settings</span>
                  <span className="text-[10px] tracking-tight text-text-muted">
                    {isPrivateMode ? "Private" : "Visible"}
                  </span>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setIsProfileMenuOpen(false);
                    /* Nothing linked yet → go straight to the Plaid sync
                       flow; otherwise show the linked roster on finance. */
                    if (connected.length === 0) openModal();
                    else navigate("finance");
                  }}
                  className={menuItemCls}
                >
                  <span>Account Info</span>
                  <span className="text-[10px] tabular-nums tracking-tight text-text-muted">
                    {connected.length}
                  </span>
                </button>

                <div className="my-1 h-px bg-border-muted" />

                <button
                  type="button"
                  role="menuitem"
                  onClick={handleLogOut}
                  className="flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left text-xs tracking-tight text-red-500 transition-colors hover:bg-red-500/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-red-500/40"
                >
                  <span>Log Out</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Phase 2 — running market marquee ticker: mounts only while the
          stocks view is active or the ⌘K research overlay is open; navigating
          home/spending unmounts it from the DOM entirely. Phase 4 — the
          stream carries real `/api/crypto` + `/api/stocks` quotes. */}
      {showMarketTicker && (
        <MarketMarqueeTicker quotes={tickerQuotes} masked={isPrivateMode} />
      )}

      {/* Plaid-style institution syncing overlay — fresh mount per session */}
      {modalSession > 0 && (
        <ConnectAccountModal
          key={modalSession}
          onClose={closeModal}
          onComplete={handleInstitutionConnected}
        />
      )}

      {/* Phase 2 — Personal Cash Flow Input Calculator overlay, opened from
          the Monthly Cash Flow card. */}
      {isCashFlowModalOpen && (
        <CashFlowModal
          profile={cashFlowProfile}
          onSave={handleSaveCashFlowProfile}
          onClose={() => setIsCashFlowModalOpen(false)}
        />
      )}
    </div>
  );
}