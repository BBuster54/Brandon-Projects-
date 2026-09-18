"use client";

/**
 * Bill Reminder Calendar — the cockpit's fourth primary view.
 *
 * A dense, hairline monthly grid in the Fey idiom: no cell borders, no
 * weight, just a 7-column lattice where the DATA is the only thing that
 * carries color. Each recurring liability renders as a razor-thin tinted
 * line beneath its date, so a month's obligations are legible as a pattern
 * before any single entry is read.
 *
 * Liabilities are recurring by day-of-month (rent on the 1st, a
 * subscription on the 14th), which is how consumer bills actually behave —
 * storing absolute dates would mean re-entering everything each month.
 *
 * The parent owns the bill list and persistence; this component is a
 * controlled surface that reports adds and removes upward. Advisor
 * notification injection also happens in the parent, for the same reason:
 * one owner for one piece of state.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* -------------------------------------------------------------------------- */
/*                                   Types                                    */
/* -------------------------------------------------------------------------- */

/** A recurring monthly liability pinned to a day-of-month. */
export type Bill = {
  id: string;
  /** Display label, e.g. "Rent" or "Netflix". */
  label: string;
  /** 1–31. Months shorter than the day clamp to the final day. */
  day: number;
  /** Monthly amount in dollars. 0 is allowed — some reminders aren't bills. */
  amount: number;
  /** One of BILL_ACCENTS — the tint of the razor-thin line. */
  accent: BillAccent;
};

export type BillAccent = "green" | "violet" | "amber" | "rose" | "sky";

/** Accent tints. Kept as a closed map so a bad value can't leak into class
    names — Tailwind v4 needs literal class strings to emit the utility. */
export const BILL_ACCENTS: Record<
  BillAccent,
  { line: string; dot: string; text: string; label: string }
> = {
  green: {
    line: "bg-accent-green/70",
    dot: "bg-accent-green",
    text: "text-accent-green",
    label: "Green",
  },
  violet: {
    line: "bg-[#8b7ec8]/70",
    dot: "bg-[#8b7ec8]",
    text: "text-[#8b7ec8]",
    label: "Violet",
  },
  amber: {
    line: "bg-amber-500/70",
    dot: "bg-amber-500",
    text: "text-amber-500",
    label: "Amber",
  },
  rose: {
    line: "bg-rose-500/70",
    dot: "bg-rose-500",
    text: "text-rose-500",
    label: "Rose",
  },
  sky: {
    line: "bg-sky-500/70",
    dot: "bg-sky-500",
    text: "text-sky-500",
    label: "Sky",
  },
};

const ACCENT_KEYS = Object.keys(BILL_ACCENTS) as BillAccent[];

const WEEKDAYS = ["S", "M", "T", "W", "T", "F", "S"];

const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/* -------------------------------------------------------------------------- */
/*                                  Helpers                                   */
/* -------------------------------------------------------------------------- */

export function daysInMonth(year: number, monthIndex: number): number {
  /* Day 0 of the NEXT month is the last day of this one. */
  return new Date(year, monthIndex + 1, 0).getDate();
}

/**
 * Resolves a bill's day-of-month against a specific month. A bill due on
 * the 31st still lands in February — on the 28th or 29th — rather than
 * silently vanishing for three months of the year.
 */
export function resolveBillDay(
  bill: Bill,
  year: number,
  monthIndex: number,
): number {
  return Math.min(bill.day, daysInMonth(year, monthIndex));
}

function usd(n: number): string {
  return `$${Math.round(n).toLocaleString("en-US")}`;
}

/* -------------------------------------------------------------------------- */
/*                                 Component                                  */
/* -------------------------------------------------------------------------- */

export default function BillCalendar({
  bills,
  onAddBill,
  onRemoveBill,
  mask,
  /** Today's date, resolved by the parent after mount to avoid hydration
      drift — a server-rendered `new Date()` disagrees with the client. */
  today,
}: {
  bills: Bill[];
  onAddBill: (bill: Omit<Bill, "id">) => void;
  onRemoveBill: (id: string) => void;
  mask: (figure: React.ReactNode) => React.ReactNode;
  today: Date | null;
}) {
  /* The month being viewed, as an offset from the current month. Kept as an
     offset rather than a concrete date so it stays correct if `today`
     resolves a beat after mount. */
  const [monthOffset, setMonthOffset] = useState(0);
  const [selectedDay, setSelectedDay] = useState<number | null>(null);

  /* Composer state for the selected day's tile. */
  const [draftLabel, setDraftLabel] = useState("");
  const [draftAmount, setDraftAmount] = useState("");
  const [draftAccent, setDraftAccent] = useState<BillAccent>("green");
  const labelInputRef = useRef<HTMLInputElement>(null);

  const anchor = today ?? new Date(2026, 8, 16);
  const viewDate = new Date(
    anchor.getFullYear(),
    anchor.getMonth() + monthOffset,
    1,
  );
  const year = viewDate.getFullYear();
  const monthIndex = viewDate.getMonth();
  const isCurrentMonth = monthOffset === 0;
  const todayDate = today ? today.getDate() : null;

  const totalDays = daysInMonth(year, monthIndex);
  const leadingBlanks = new Date(year, monthIndex, 1).getDay();

  /* day-of-month → bills falling on it, resolved for THIS month's length. */
  const billsByDay = useMemo(() => {
    const map = new Map<number, Bill[]>();
    for (const bill of bills) {
      const day = resolveBillDay(bill, year, monthIndex);
      const existing = map.get(day);
      if (existing) existing.push(bill);
      else map.set(day, [bill]);
    }
    return map;
  }, [bills, year, monthIndex]);

  const monthlyTotal = useMemo(
    () => bills.reduce((sum, b) => sum + b.amount, 0),
    [bills],
  );

  /* Focus the composer the moment a tile opens — the whole interaction is
     click-tile-then-type, and a manual second click would break that. */
  useEffect(() => {
    if (selectedDay === null) return;
    const t = window.setTimeout(() => labelInputRef.current?.focus(), 60);
    return () => window.clearTimeout(t);
  }, [selectedDay]);

  /* Escape closes the composer without committing. */
  useEffect(() => {
    if (selectedDay === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedDay(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedDay]);

  const resetDraft = useCallback(() => {
    setDraftLabel("");
    setDraftAmount("");
    setDraftAccent("green");
  }, []);

  const commit = useCallback(() => {
    if (selectedDay === null) return;
    const label = draftLabel.trim();
    if (!label) return;

    /* An empty or malformed amount is a zero-dollar reminder, not an error:
       "Review insurance — 12th" is a legitimate calendar entry. */
    const parsed = Number(draftAmount.replace(/[^0-9.]/g, ""));
    const amount = Number.isFinite(parsed) && parsed > 0 ? parsed : 0;

    onAddBill({ label, day: selectedDay, amount, accent: draftAccent });
    resetDraft();
    setSelectedDay(null);
  }, [selectedDay, draftLabel, draftAmount, draftAccent, onAddBill, resetDraft]);

  /* Upcoming obligations in the next 14 days of the current month — the
     roster the advisor sidebar mirrors. */
  const upcoming = useMemo(() => {
    if (!isCurrentMonth || todayDate === null) return [];
    return bills
      .map((b) => ({ bill: b, day: resolveBillDay(b, year, monthIndex) }))
      .filter((r) => r.day >= todayDate && r.day <= todayDate + 14)
      .sort((a, b) => a.day - b.day);
  }, [bills, isCurrentMonth, todayDate, year, monthIndex]);

  return (
    <div className="animate-fey-fade grid gap-6 lg:grid-cols-3">
      {/* ============================ CALENDAR GRID ========================== */}
      <section className="rounded-2xl border border-border-muted bg-card p-6 lg:col-span-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-foreground">
              {MONTH_NAMES[monthIndex]}{" "}
              <span className="text-text-muted">{year}</span>
            </h2>
            <p className="mt-1 text-xs tracking-tight text-text-muted">
              {bills.length} recurring{" "}
              {bills.length === 1 ? "liability" : "liabilities"} ·{" "}
              {mask(usd(monthlyTotal))}/mo
            </p>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setMonthOffset((m) => m - 1)}
              aria-label="Previous month"
              className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-lg border border-border-muted text-text-muted transition-colors hover:border-foreground/25 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
                className="h-3.5 w-3.5"
              >
                <path d="M15 18l-6-6 6-6" />
              </svg>
            </button>
            {monthOffset !== 0 && (
              <button
                type="button"
                onClick={() => setMonthOffset(0)}
                className="cursor-pointer rounded-lg border border-border-muted px-2.5 py-1 text-[11px] tracking-tight text-text-muted transition-colors hover:border-foreground/25 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
              >
                Today
              </button>
            )}
            <button
              type="button"
              onClick={() => setMonthOffset((m) => m + 1)}
              aria-label="Next month"
              className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-lg border border-border-muted text-text-muted transition-colors hover:border-foreground/25 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
                className="h-3.5 w-3.5"
              >
                <path d="M9 18l6-6-6-6" />
              </svg>
            </button>
          </div>
        </div>

        {/* Weekday rail */}
        <div className="mt-5 grid grid-cols-7 gap-1.5 border-b border-border-muted pb-2">
          {WEEKDAYS.map((d, i) => (
            <span
              key={`${d}-${i}`}
              className="text-center text-[10px] uppercase tracking-wider text-text-muted"
            >
              {d}
            </span>
          ))}
        </div>

        {/* Day lattice */}
        <div className="mt-1.5 grid grid-cols-7 gap-1.5">
          {Array.from({ length: leadingBlanks }).map((_, i) => (
            <span key={`blank-${i}`} aria-hidden className="min-h-[76px]" />
          ))}

          {Array.from({ length: totalDays }).map((_, i) => {
            const day = i + 1;
            const dayBills = billsByDay.get(day) ?? [];
            const isToday = isCurrentMonth && day === todayDate;
            const isSelected = day === selectedDay;

            return (
              <button
                key={day}
                type="button"
                onClick={() => {
                  setSelectedDay((prev) => (prev === day ? null : day));
                  resetDraft();
                }}
                aria-label={`${MONTH_NAMES[monthIndex]} ${day}${
                  dayBills.length > 0
                    ? `, ${dayBills.length} liability due`
                    : ", no liabilities"
                }`}
                aria-pressed={isSelected}
                className={`flex min-h-[76px] cursor-pointer flex-col gap-1 rounded-lg border p-1.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                  isSelected
                    ? "border-foreground/30 bg-foreground/[0.05]"
                    : "border-transparent hover:border-border-muted hover:bg-foreground/[0.02]"
                }`}
              >
                <span
                  className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] tabular-nums tracking-tight ${
                    isToday
                      ? "bg-foreground font-semibold text-background"
                      : "text-text-muted"
                  }`}
                >
                  {day}
                </span>

                {/* Razor-thin liability lines — the data IS the decoration. */}
                <span className="flex flex-col gap-1">
                  {dayBills.slice(0, 3).map((b) => (
                    <span key={b.id} className="flex flex-col gap-0.5">
                      <span
                        className={`block h-[2px] w-full rounded-full ${BILL_ACCENTS[b.accent].line}`}
                      />
                      <span className="truncate text-[9px] leading-tight tracking-tight text-text-muted">
                        {b.label}
                      </span>
                    </span>
                  ))}
                  {dayBills.length > 3 && (
                    <span className="text-[9px] tracking-tight text-text-muted">
                      +{dayBills.length - 3} more
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </div>

        {/* ---------------------- Day composer / editor ---------------------- */}
        {selectedDay !== null && (
          <div className="animate-fey-fade mt-5 rounded-xl border border-border-muted bg-background/40 p-4">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium tracking-tight text-foreground">
                {MONTH_NAMES[monthIndex]} {selectedDay}
              </p>
              <button
                type="button"
                onClick={() => setSelectedDay(null)}
                aria-label="Close day editor"
                className="cursor-pointer text-[11px] tracking-tight text-text-muted transition-colors hover:text-foreground"
              >
                Close
              </button>
            </div>
            <p className="mt-0.5 text-[11px] tracking-tight text-text-muted">
              Recurs on the {selectedDay}
              {selectedDay === 1
                ? "st"
                : selectedDay === 2
                  ? "nd"
                  : selectedDay === 3
                    ? "rd"
                    : "th"}{" "}
              of every month.
            </p>

            {/* Existing entries on this day */}
            {(billsByDay.get(selectedDay) ?? []).length > 0 && (
              <ul className="mt-3 flex flex-col divide-y divide-border-muted/60 border-y border-border-muted">
                {(billsByDay.get(selectedDay) ?? []).map((b) => (
                  <li
                    key={b.id}
                    className="flex items-center justify-between gap-3 py-2"
                  >
                    <span className="flex min-w-0 items-center gap-2">
                      <span
                        className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${BILL_ACCENTS[b.accent].dot}`}
                      />
                      <span className="truncate text-xs tracking-tight text-foreground">
                        {b.label}
                      </span>
                    </span>
                    <span className="flex flex-shrink-0 items-center gap-3">
                      {b.amount > 0 && (
                        <span className="text-xs tabular-nums tracking-tight text-text-muted">
                          {mask(usd(b.amount))}
                        </span>
                      )}
                      <button
                        type="button"
                        onClick={() => onRemoveBill(b.id)}
                        aria-label={`Remove ${b.label}`}
                        className="cursor-pointer text-[11px] tracking-tight text-text-muted transition-colors hover:text-red-500"
                      >
                        Remove
                      </button>
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {/* New entry composer */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <input
                ref={labelInputRef}
                type="text"
                value={draftLabel}
                onChange={(e) => setDraftLabel(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commit();
                }}
                placeholder="Rent, Netflix, car payment…"
                aria-label="Liability name"
                className="min-w-[10rem] flex-1 rounded-lg border border-border-muted bg-background/60 px-3 py-2 text-sm tracking-tight text-foreground placeholder:text-text-muted/60 focus:border-foreground/30 focus:outline-none"
              />
              <input
                type="text"
                inputMode="decimal"
                value={draftAmount}
                onChange={(e) => setDraftAmount(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commit();
                }}
                placeholder="$0"
                aria-label="Monthly amount"
                className="w-24 rounded-lg border border-border-muted bg-background/60 px-3 py-2 text-sm tabular-nums tracking-tight text-foreground placeholder:text-text-muted/60 focus:border-foreground/30 focus:outline-none"
              />

              {/* Accent picker — five swatches, no dropdown. */}
              <span className="flex items-center gap-1.5">
                {ACCENT_KEYS.map((key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setDraftAccent(key)}
                    aria-label={`${BILL_ACCENTS[key].label} accent`}
                    aria-pressed={draftAccent === key}
                    className={`h-5 w-5 cursor-pointer rounded-full border transition-transform focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                      draftAccent === key
                        ? "scale-110 border-foreground/50"
                        : "border-transparent hover:scale-105"
                    }`}
                  >
                    <span
                      className={`block h-full w-full rounded-full ${BILL_ACCENTS[key].dot}`}
                    />
                  </button>
                ))}
              </span>

              <button
                type="button"
                onClick={commit}
                disabled={draftLabel.trim().length === 0}
                className="cursor-pointer rounded-lg bg-foreground px-4 py-2 text-xs font-medium tracking-tight text-background transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
              >
                Add reminder
              </button>
            </div>
          </div>
        )}
      </section>

      {/* =========================== UPCOMING RAIL =========================== */}
      <aside className="flex flex-col gap-6">
        <section className="rounded-2xl border border-border-muted bg-card p-6">
          <span className="text-[11px] uppercase tracking-wider text-text-muted">
            Next 14 days
          </span>

          {upcoming.length === 0 ? (
            <p className="mt-3 text-xs leading-relaxed tracking-tight text-text-muted">
              {bills.length === 0
                ? "No liabilities scheduled yet. Click any day tile to pin a recurring bill — rent, a subscription, a loan payment."
                : "Nothing due in the next two weeks."}
            </p>
          ) : (
            <ul className="mt-3 flex flex-col divide-y divide-border-muted/60 border-t border-border-muted pt-1">
              {upcoming.map(({ bill, day }) => {
                const daysOut = todayDate === null ? 0 : day - todayDate;
                return (
                  <li
                    key={bill.id}
                    className="flex items-center justify-between gap-3 py-2.5"
                  >
                    <span className="flex min-w-0 items-center gap-2.5">
                      <span
                        className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${BILL_ACCENTS[bill.accent].dot}`}
                      />
                      <span className="min-w-0">
                        <span className="block truncate text-sm tracking-tight text-foreground">
                          {bill.label}
                        </span>
                        <span className="block text-[11px] tracking-tight text-text-muted">
                          {daysOut === 0
                            ? "Due today"
                            : daysOut === 1
                              ? "Due tomorrow"
                              : `In ${daysOut} days`}
                        </span>
                      </span>
                    </span>
                    {bill.amount > 0 && (
                      <span className="flex-shrink-0 text-sm tabular-nums tracking-tight text-foreground">
                        {mask(usd(bill.amount))}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="rounded-2xl border border-border-muted bg-card p-6">
          <span className="text-[11px] uppercase tracking-wider text-text-muted">
            Monthly obligation
          </span>
          <p className="mt-2 text-2xl font-semibold tabular-nums tracking-tight text-foreground">
            {mask(usd(monthlyTotal))}
          </p>
          <p className="mt-1 text-xs tracking-tight text-text-muted">
            Committed across {bills.length}{" "}
            {bills.length === 1 ? "liability" : "liabilities"} every month
          </p>

          {bills.length > 0 && (
            <ul className="mt-4 flex flex-col gap-2 border-t border-border-muted pt-3">
              {[...bills]
                .sort((a, b) => b.amount - a.amount)
                .slice(0, 5)
                .map((b) => {
                  const share =
                    monthlyTotal > 0 ? (b.amount / monthlyTotal) * 100 : 0;
                  return (
                    <li key={b.id} className="flex flex-col gap-1">
                      <span className="flex items-center justify-between text-xs tracking-tight">
                        <span className="truncate text-text-muted">
                          {b.label}
                        </span>
                        <span className="tabular-nums text-foreground">
                          {mask(usd(b.amount))}
                        </span>
                      </span>
                      <span className="h-[2px] w-full rounded-full bg-border-muted">
                        <span
                          className={`block h-full rounded-full ${BILL_ACCENTS[b.accent].line}`}
                          style={{ width: `${Math.max(share, 2)}%` }}
                        />
                      </span>
                    </li>
                  );
                })}
            </ul>
          )}
        </section>
      </aside>
    </div>
  );
}