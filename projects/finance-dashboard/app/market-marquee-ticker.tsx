"use client";

/**
 * Running Market Marquee Ticker — Phase 2.
 *
 * A sleek quote bar pinned to the absolute bottom of the viewport. The track
 * is a double-looped flex row: two identical copies of the quote sequence are
 * rendered back-to-back inside a single `.animate-fey-marquee` track
 * (globals.css), which slides to exactly −50% of its own width — landing one
 * full copy later — for a seamless, infinite right-to-left glide. Quotes are
 * passed in already live: crypto prices mirror the dashboard's 3-second
 * holdings tick and equity quotes drift on the same 3-second cadence.
 *
 * The mount condition lives in the parent: activeView === "stocks" or the ⌘K
 * research panel overlay open. Anything else unmounts it from the DOM.
 */

export type TickerQuote = {
  symbol: string;
  name: string;
  price: number;
  dayPct: number;
};

function usd2(n: number): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/** One identical copy of the quote sequence. */
function QuoteRow({
  copy,
  quotes,
  hidden,
}: {
  copy: string;
  quotes: TickerQuote[];
  hidden: boolean;
}) {
  return (
    <div
      className="flex shrink-0 items-center"
      aria-hidden={hidden || undefined}
    >
      {quotes.map((q) => (
        <span
          key={`${copy}-${q.symbol}`}
          className="flex items-center gap-2 whitespace-nowrap px-5 text-xs tracking-tight"
        >
          <span className="font-medium text-foreground">{q.symbol}</span>
          <span className="tabular-nums text-text-muted">
            ${usd2(q.price)}
          </span>
          <span
            className={`tabular-nums ${
              q.dayPct >= 0 ? "text-accent-green" : "text-red-500"
            }`}
          >
            {q.dayPct >= 0 ? "+" : ""}
            {q.dayPct.toFixed(2)}%
          </span>
        </span>
      ))}
    </div>
  );
}

export default function MarketMarqueeTicker({
  quotes,
  masked,
}: {
  quotes: TickerQuote[];
  masked: boolean;
}) {
  return (
    <div
      role="marquee"
      aria-label="Live market ticker"
      className={`fixed bottom-0 left-0 right-0 h-8 bg-card/80 backdrop-blur border-t border-border-muted z-40 overflow-hidden select-none flex items-center transition-all duration-300 ${
        masked ? "blur-md" : ""
      }`}
    >
      {/* Double-looped track: two identical copies + the translateX(-50%)
          keyframe = seamless infinite glide. `w-max` sizes the track to its
          content so −50% lands exactly one copy later; `shrink-0` keeps both
          halves intact inside the overflow-hidden bar. */}
      <div className="animate-fey-marquee w-max">
        <QuoteRow copy="a" quotes={quotes} hidden={false} />
        <QuoteRow copy="b" quotes={quotes} hidden />
      </div>
    </div>
  );
}