"use client";

/**
 * Personal Cash Flow Input Calculator — Phase 2 overlay configuration window.
 *
 * Slides open over the dashboard (`max-w-md`) when the Monthly Cash Flow card
 * is clicked. The user enters or slides four raw monthly metrics:
 *
 *   Monthly Take-Home Income · Rent/Mortgage · Utilities & Insurance ·
 *   Discretionary Spend
 *
 * A reactive calculation panel recomputes on every keystroke/drag:
 *   - Total Monthly Outlays   (housing + bills + discretionary)
 *   - Net Savings Capacity    (income − outlays)
 *   - Cash Flow Health Score  (0–100 rating scored against the standard
 *     50% Essentials / 30% Savings / 20% Discretionary baseline allocation)
 *
 * "Save Cash Flow Profile" commits the draft to the dashboard's global state,
 * which instantly updates the Monthly Cash Flow card's headline figure.
 */

import { useEffect, useState } from "react";

export type CashFlowProfile = {
  income: number;
  rent: number;
  utilities: number;
  discretionary: number;
};

/** Healthy baseline allocation model — 50% essentials / 30% savings / 20% wants. */
const BASELINE_ESSENTIALS = 0.5;
const BASELINE_SAVINGS = 0.3;
const BASELINE_DISCRETIONARY = 0.2;

/**
 * Health score against the 50/30/20 model. Each pillar is scored 0–1 by how
 * far its actual share deviates from the baseline (measured as share of
 * income), penalized with a 2× weight for overruns (overspending hurts more
 * than underspending helps); savings over the 30% target is pure upside.
 * The weighted pillars scale to a 0–100 rating.
 */
function healthScore(p: CashFlowProfile): number {
  if (p.income <= 0) return 0;
  const essentials = (p.rent + p.utilities) / p.income;
  const savings = Math.max(0, (p.income - (p.rent + p.utilities + p.discretionary)) / p.income);
  const discretionary = Math.min(1, p.discretionary / p.income);

  const pillar = (actual: number, target: number) => {
    const gap = actual - target;
    const penalty = gap > 0 ? gap * 2 : -gap; // overruns weighted 2×
    return Math.max(0, 1 - penalty / target);
  };

  const score =
    pillar(essentials, BASELINE_ESSENTIALS) * 0.4 +
    Math.min(savings / BASELINE_SAVINGS, 1) * 0.4 +
    pillar(discretionary, BASELINE_DISCRETIONARY) * 0.2;
  return Math.round(score * 100);
}

function ratingLabel(score: number): { label: string; tone: string } {
  if (score >= 85) return { label: "Excellent", tone: "text-accent-green" };
  if (score >= 70) return { label: "Strong", tone: "text-accent-green" };
  if (score >= 55) return { label: "Stable", tone: "text-foreground" };
  if (score >= 35) return { label: "Strained", tone: "text-amber-500" };
  return { label: "At Risk", tone: "text-red-500" };
}

function usd0(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

/** One labeled row: number input + hairline slider, mirroring the fey-range aesthetic. */
function MetricRow({
  id,
  label,
  value,
  max,
  onChange,
  hint,
}: {
  id: string;
  label: string;
  value: number;
  max: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label
          htmlFor={id}
          className="text-[11px] uppercase tracking-wider text-text-muted"
        >
          {label}
        </label>
        <div className="flex items-center gap-1">
          <span className="text-xs text-text-muted">$</span>
          <input
            id={id}
            type="number"
            min={0}
            step={10}
            value={value}
            onChange={(e) =>
              onChange(Math.max(0, Number(e.target.value) || 0))
            }
            className="w-24 rounded-md border border-border-muted bg-background/60 px-2 py-1 text-right text-sm tabular-nums text-foreground focus:border-foreground/30 focus:outline-none"
          />
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={max}
        step={10}
        value={Math.min(value, max)}
        onChange={(e) => onChange(Number(e.target.value))}
        className="fey-range mt-2.5 w-full"
        aria-label={`${label} slider`}
        aria-valuetext={`$${usd0(value)} per month`}
      />
      {hint && (
        <p className="mt-1.5 text-[11px] tracking-tight text-text-muted">{hint}</p>
      )}
    </div>
  );
}

export default function CashFlowModal({
  profile,
  onSave,
  onClose,
}: {
  profile: CashFlowProfile;
  onSave: (profile: CashFlowProfile) => void;
  onClose: () => void;
}) {
  /* Local draft — edits stay in the modal until "Save" commits them upward. */
  const [income, setIncome] = useState(profile.income);
  const [rent, setRent] = useState(profile.rent);
  const [utilities, setUtilities] = useState(profile.utilities);
  const [discretionary, setDiscretionary] = useState(profile.discretionary);

  /* Escape dismisses, matching every other overlay in the shell. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const draft: CashFlowProfile = { income, rent, utilities, discretionary };
  const outlays = rent + utilities + discretionary;
  const net = income - outlays;
  const score = healthScore(draft);
  const { label, tone } = ratingLabel(score);

  const save = () => {
    onSave(draft);
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Monthly cash flow calculator"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-6 backdrop-blur-sm"
    >
      <div className="terminal-overlay-card flex w-full max-w-md flex-col overflow-hidden rounded-xl border border-border-muted bg-card shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between p-5 pb-0">
          <div>
            <h2 className="text-base font-semibold tracking-tight text-foreground">
              Monthly Cash Flow
            </h2>
            <p className="mt-1 text-xs tracking-tight text-text-muted">
              Enter your raw monthly figures — the panel recalculates live
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="cursor-pointer p-1 text-sm leading-none text-text-muted hover:text-foreground"
          >
            ✕
          </button>
        </div>

        {/* Inputs */}
        <div className="flex flex-col gap-5 p-5">
          <MetricRow
            id="cf-income"
            label="Monthly Take-Home Income"
            value={income}
            max={20_000}
            onChange={setIncome}
          />
          <MetricRow
            id="cf-rent"
            label="Rent / Mortgage"
            value={rent}
            max={8_000}
            onChange={setRent}
          />
          <MetricRow
            id="cf-utilities"
            label="Utilities & Insurance"
            value={utilities}
            max={2_000}
            onChange={setUtilities}
          />
          <MetricRow
            id="cf-discretionary"
            label="Discretionary Spend"
            value={discretionary}
            max={6_000}
            onChange={setDiscretionary}
            hint="Dining, shopping, travel, subscriptions — everything non-essential."
          />
        </div>

        {/* Reactive calculation panel */}
        <div className="mx-5 mb-5 rounded-lg border border-border-muted bg-background/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-foreground">
              Calculation
            </span>
            <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-text-muted">
              <span className="anim-pulse-soft inline-block h-1 w-1 rounded-full bg-accent-green" />
              Live
            </span>
          </div>

          <dl className="mt-3 flex flex-col gap-2.5 text-sm tracking-tight">
            <div className="flex items-center justify-between">
              <dt className="text-text-muted">Total Monthly Outlays</dt>
              <dd className="font-medium tabular-nums text-foreground">
                ${usd0(outlays)}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-text-muted">Net Savings Capacity</dt>
              <dd
                className={`font-medium tabular-nums ${
                  net >= 0 ? "text-accent-green" : "text-red-500"
                }`}
              >
                {net >= 0 ? "+" : "−"}${usd0(Math.abs(net))}
              </dd>
            </div>
          </dl>

          {/* Health score ring vs the 50/30/20 baseline */}
          <div className="mt-4 flex items-center gap-4 border-t border-border-muted pt-4">
            <div className="relative h-14 w-14 flex-shrink-0">
              <svg viewBox="0 0 64 64" className="h-14 w-14 -rotate-90" aria-hidden>
                <circle
                  cx="32"
                  cy="32"
                  r="26"
                  fill="none"
                  stroke="var(--border-muted)"
                  strokeWidth="4"
                />
                <circle
                  cx="32"
                  cy="32"
                  r="26"
                  fill="none"
                  stroke="var(--accent-green)"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={2 * Math.PI * 26}
                  strokeDashoffset={2 * Math.PI * 26 * (1 - score / 100)}
                  className="transition-all duration-500"
                />
              </svg>
              <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold tabular-nums text-foreground">
                {score}
              </span>
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium tracking-tight text-foreground">
                Cash Flow Health · <span className={tone}>{label}</span>
              </p>
              <p className="mt-0.5 text-[11px] leading-relaxed tracking-tight text-text-muted">
                Scored against the 50% essentials / 30% savings / 20%
                discretionary baseline allocation model.
              </p>
            </div>
          </div>
        </div>

        {/* Commit */}
        <div className="border-t border-border-muted p-5 pt-4">
          <button
            type="button"
            onClick={save}
            className="w-full cursor-pointer rounded-lg bg-foreground py-2.5 text-sm font-medium tracking-tight text-background transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
          >
            Save Cash Flow Profile
          </button>
          <button
            type="button"
            onClick={onClose}
            className="mt-2 w-full cursor-pointer text-[11px] tracking-tight text-text-muted hover:text-foreground"
          >
            Discard changes
          </button>
        </div>
      </div>
    </div>
  );
}