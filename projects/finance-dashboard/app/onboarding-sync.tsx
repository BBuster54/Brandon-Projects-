"use client";

/**
 * First-run onboarding — the bank-sync gate shown strictly AFTER a user
 * registers/logs in for the first time (`isNewUser`), before the homepage
 * cockpit mounts.
 *
 * Screen copy: "Welcome to your financial cockpit. Let's securely link your
 * institutions to aggregate your net worth."
 *
 * Embedded here is the full 3-step Plaid-style multi-institution sync flow
 * (institution grid → secure encrypted credential handshake → success +
 * hydration marquee). Data behaves like Origin: a secure, READ-ONLY token
 * connection that safely ingests bank details, historical balances and
 * historical ledger transaction lines — no corporate overhead, credentials
 * never stored. Every completed sync is committed upward immediately via
 * `onInstitutionConnected`, so linked balances/transactions are already in
 * dashboard state by the time the user continues.
 *
 * "Skip for now" routes to the blank dashboard state for manual entry later.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { INSTITUTIONS, type Institution } from "./institutions";

type SyncStep = "institution" | "credentials" | "success";

const STEP_COPY: Record<SyncStep, { title: string; sub: string }> = {
  institution: {
    title: "Link your institutions",
    sub: "Select any institutions you hold accounts with",
  },
  credentials: {
    title: "Secure login",
    sub: "Establishing a read-only token connection...",
  },
  success: {
    title: "Sync complete",
    sub: "Ingesting balances and historical transactions...",
  },
};

/** Compact step dots — filled as the sync flow advances. */
function StepDots({ step }: { step: SyncStep }) {
  const order: SyncStep[] = ["institution", "credentials", "success"];
  return (
    <div className="flex items-center gap-1.5">
      {order.map((s) => (
        <span
          key={s}
          className={`h-1 w-4 rounded-full transition-colors duration-300 ${
            order.indexOf(s) <= order.indexOf(step)
              ? "bg-accent-green"
              : "bg-border-muted"
          }`}
        />
      ))}
    </div>
  );
}

/** Animated success check — stroke draws in via CSS keyframes. */
function SuccessCheck() {
  return (
    <div className="flex justify-center">
      <span className="anim-check-ring flex h-16 w-16 items-center justify-center rounded-full border border-accent-green/40 bg-accent-green/10">
        <svg
          viewBox="0 0 32 32"
          fill="none"
          aria-hidden
          className="h-8 w-8 text-accent-green"
        >
          <path
            d="M8 17 L14 23 L24 10"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="anim-check-draw"
          />
        </svg>
      </span>
    </div>
  );
}

/**
 * Embedded (non-overlay) variant of the Plaid-style sync flow, rendered
 * inside the onboarding screen. Each completed institution is committed
 * upward instantly; `onExit` fires for both "Skip for now" and "Continue".
 */
export default function OnboardingSync({
  onInstitutionConnected,
  onExit,
}: {
  onInstitutionConnected: (institution: Institution) => void;
  /** Routes onward — blank state when nothing was linked. */
  onExit: () => void;
}) {
  const [step, setStep] = useState<SyncStep>("institution");
  const [selected, setSelected] = useState<Institution | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [linked, setLinked] = useState<Institution[]>([]);

  /* Escape at step 1 exits onboarding entirely; deeper steps unwind one level. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (step === "institution") onExit();
      if (step === "credentials") {
        setSelected(null);
        setStep("institution");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, onExit]);

  /* Every simulated timer is tracked so unmount never leaks a callback. */
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach(clearTimeout);
  }, []);

  const submitCredentials = useCallback(() => {
    if (!selected) return;
    setSubmitting(true);
    /* Simulated token round-trip, then success + instant data ingestion. */
    timers.current.push(
      setTimeout(() => {
        setSubmitting(false);
        setStep("success");
        onInstitutionConnected(selected);
        setLinked((prev) => [...prev, selected]);
        /* Hydration beat on the success card, then back to the grid. */
        timers.current.push(
          setTimeout(() => {
            setSelected(null);
            setUsername("");
            setPassword("");
            setStep("institution");
          }, 1800),
        );
      }, 900),
    );
  }, [selected, onInstitutionConnected]);

  const copy = STEP_COPY[step];

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        {/* ---------- Welcome headline ---------- */}
        <div className="text-center">
          <span
            aria-hidden
            className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-foreground text-background"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
              <path d="M12 2 L22 12 L12 22 L2 12 Z" />
            </svg>
          </span>
          <h1 className="mt-5 text-2xl font-semibold tracking-tight text-foreground">
            {
              "Welcome to your financial cockpit. Let's securely link your institutions to aggregate your net worth."
            }
          </h1>
          <p className="mx-auto mt-3 max-w-md text-xs leading-relaxed tracking-tight text-text-muted">
            A secure, read-only token connection safely ingests your bank
            details, historical balances and historical ledger transaction
            lines — no corporate overhead, credentials never stored.
          </p>
        </div>

        {/* ---------- Embedded 3-step sync card ---------- */}
        <div className="terminal-overlay-card mt-8 rounded-xl border border-border-muted bg-card tracking-tight">
          {/* Card header — persistent across every step */}
          <div className="flex items-center justify-between p-5 pb-0">
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-base font-semibold tracking-tight text-foreground">
                  {copy.title}
                </h2>
                <StepDots step={step} />
              </div>
              <p className="mt-1 text-xs tracking-tight text-text-muted">
                {copy.sub}
              </p>
            </div>
            <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-text-muted">
              <span className="anim-pulse-soft inline-block h-1 w-1 rounded-full bg-accent-green" />
              Encrypted
            </span>
          </div>

          {/* ===== STEP 1 · Multi-institution selection grid ===== */}
          {step === "institution" && (
            <div className="p-5">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {INSTITUTIONS.map((inst) => {
                  const alreadyLinked = linked.some((l) => l.id === inst.id);
                  return (
                    <button
                      key={inst.id}
                      type="button"
                      disabled={alreadyLinked}
                      onClick={() => {
                        setSelected(inst);
                        setStep("credentials");
                      }}
                      className={`flex items-center gap-3 rounded-lg border border-border-muted p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40 ${
                        alreadyLinked
                          ? "cursor-default border-accent-green/30 bg-accent-green/[0.06] opacity-80"
                          : "cursor-pointer hover:border-foreground/25 hover:bg-foreground/[0.04]"
                      }`}
                    >
                      <span
                        className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-xs font-bold text-white ${inst.color}`}
                      >
                        {inst.initials}
                      </span>
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium tracking-tight text-foreground">
                          {inst.name}
                        </span>
                        <span className="block truncate text-[11px] tracking-tight text-text-muted">
                          {alreadyLinked ? "Linked ✓" : inst.kind}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
              <p className="mt-4 flex items-center gap-1.5 text-[11px] text-text-muted">
                <span className="inline-block h-3 w-3 rounded-full border border-border-muted text-[8px] leading-[10px] text-center">
                  🔒
                </span>
                Read-only access · Historical balances + transactions ingested
                safely
              </p>
            </div>
          )}

          {/* ===== STEP 2 · Secure token handshake interface ===== */}
          {step === "credentials" && selected && (
            <div className="p-5">
              <div className="flex flex-col items-center text-center">
                <span
                  className={`flex h-12 w-12 items-center justify-center rounded-full text-lg font-bold text-white ${selected.color}`}
                >
                  {selected.initials}
                </span>
                <span className="mt-2.5 text-sm font-medium tracking-tight text-foreground">
                  {selected.name}
                </span>
                <span className="mt-1 flex items-center gap-1.5 text-[11px] text-text-muted">
                  <span className="anim-pulse-soft inline-block h-1 w-1 rounded-full bg-accent-green" />
                  Establishing a read-only token connection...
                </span>
              </div>

              <div className="mt-5 flex flex-col gap-2.5">
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] uppercase tracking-wider text-text-muted">
                    Username
                  </span>
                  <input
                    type="text"
                    autoComplete="off"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="user@example.com"
                    className="rounded-lg border border-border-muted bg-background/60 px-3 py-2 text-sm tracking-tight text-foreground placeholder:text-text-muted/60 focus:border-foreground/30 focus:outline-none"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] uppercase tracking-wider text-text-muted">
                    Password
                  </span>
                  <input
                    type="password"
                    autoComplete="off"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    onKeyDown={(e) => {
                      if (e.key === "Enter") submitCredentials();
                    }}
                    className="rounded-lg border border-border-muted bg-background/60 px-3 py-2 text-sm tracking-tight text-foreground placeholder:text-text-muted/60 focus:border-foreground/30 focus:outline-none"
                  />
                </label>
              </div>

              <button
                type="button"
                onClick={submitCredentials}
                disabled={submitting}
                className="mt-4 w-full cursor-pointer rounded-lg bg-foreground py-2.5 text-sm font-medium tracking-tight text-background transition-opacity hover:opacity-90 disabled:cursor-wait disabled:opacity-60"
              >
                {submitting ? "Connecting..." : "Submit Securely"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setSelected(null);
                  setStep("institution");
                }}
                className="mt-2 w-full cursor-pointer text-[11px] tracking-tight text-text-muted hover:text-foreground"
              >
                ← Choose a different institution
              </button>
            </div>
          )}

          {/* ===== STEP 3 · Sync success + live hydration ===== */}
          {step === "success" && selected && (
            <div className="flex flex-col gap-4 p-5">
              <SuccessCheck />
              <div className="text-center">
                <p className="text-sm font-medium tracking-tight text-foreground">
                  {selected.name} connected
                </p>
                <p className="mt-1 text-xs tabular-nums text-accent-green">
                  +${selected.balance.toLocaleString("en-US")} ·{" "}
                  {selected.accountLabel}
                </p>
              </div>
              <div className="overflow-hidden border-t border-border-muted pt-3">
                <p className="anim-marquee whitespace-nowrap text-center text-[11px] tracking-tight text-text-muted">
                  Ingesting balances and historical transactions... · Ingesting
                  balances and historical transactions... · Ingesting balances
                  and historical transactions...
                </p>
              </div>
            </div>
          )}
        </div>

        {/* ---------- Footer actions ---------- */}
        <div className="mt-6 flex flex-col items-center gap-3">
          <button
            type="button"
            onClick={onExit}
            disabled={linked.length === 0}
            className="w-full max-w-sm cursor-pointer rounded-lg bg-foreground py-2.5 text-sm font-medium tracking-tight text-background transition-opacity hover:opacity-90 disabled:pointer-events-none disabled:opacity-30"
          >
            {linked.length > 0
              ? `Continue to dashboard · ${linked.length} linked`
              : "Link an institution to continue"}
          </button>
          <button
            type="button"
            onClick={onExit}
            className="cursor-pointer text-[11px] tracking-tight text-text-muted hover:text-foreground"
          >
            Skip for now — you can enter data manually later
          </button>
        </div>
      </div>
    </div>
  );
}