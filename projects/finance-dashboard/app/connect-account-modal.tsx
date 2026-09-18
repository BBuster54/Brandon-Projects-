"use client";

/**
 * Plaid-style institution syncing overlay — a three-step connect flow:
 *
 *   1. Institution selection grid (Chase, BofA, Wells Fargo, Fidelity, …)
 *   2. Secure mock login (encrypted-handshake notice + credential fields)
 *   3. Sync success (animated check + parsing marquee, auto-closes in 2s)
 *
 * Step transitions are pure `useState` index tracking; step 3 commits the
 * institution's mock balance + transactions upward via `onComplete` right
 * before the modal tears itself down, so the dashboard hydrates live data.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { INSTITUTIONS, type Institution } from "./institutions";

type ConnectStep = "institution" | "credentials" | "success";

const STEP_COPY: Record<ConnectStep, { title: string; sub: string }> = {
  institution: {
    title: "Connect an institution",
    sub: "Select your financial institution to begin syncing",
  },
  credentials: {
    title: "Secure login",
    sub: "Securing end-to-end encrypted handshake...",
  },
  success: {
    title: "Sync complete",
    sub: "Parsing account balances and ledger data...",
  },
};

/** Compact step dots — filled as the flow advances. */
function StepDots({ step }: { step: ConnectStep }) {
  const order: ConnectStep[] = ["institution", "credentials", "success"];
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
 * The parent mounts this component per-session (with a fresh `key`) so the
 * flow state always initializes cleanly — no reset-on-open effects needed.
 */
export default function ConnectAccountModal({
  onClose,
  onComplete,
}: {
  onClose: () => void;
  onComplete: (institution: Institution) => void;
}) {
  const [step, setStep] = useState<ConnectStep>("institution");
  const [selected, setSelected] = useState<Institution | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  /* Escape dismisses (except during the success hand-off, which owns closing). */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && step !== "success") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, onClose]);

  /* Every simulated timer is tracked so unmount never leaks a callback. */
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach(clearTimeout);
  }, []);

  const submitCredentials = useCallback(() => {
    if (!selected) return;
    setSubmitting(true);
    /* Simulate a brief credential round-trip before showing success. */
    const verify = setTimeout(() => {
      setSubmitting(false);
      setStep("success");
      /* Ingest the new institution's balance + transactions immediately. */
      onComplete(selected);
      /* Then gracefully close the overlay after the 2s hydration beat. */
      timers.current.push(setTimeout(onClose, 2000));
    }, 900);
    timers.current.push(verify);
  }, [selected, onComplete, onClose]);

  const copy = STEP_COPY[step];

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Connect a financial institution"
      onClick={(e) => {
        if (e.target === e.currentTarget && step !== "success") onClose();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-6 backdrop-blur-sm"
    >
      <div className="terminal-overlay-card flex w-full max-w-md flex-col bg-card border border-border-muted rounded-xl shadow-2xl overflow-hidden">
        {/* Header — persistent across every step */}
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
          {step !== "success" && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="text-text-muted hover:text-foreground cursor-pointer text-sm leading-none p-1"
            >
              ✕
            </button>
          )}
        </div>

        {/* ============ STEP 1 · Institution selection grid ============ */}
        {step === "institution" && (
          <div className="p-5">
            <div className="grid grid-cols-2 gap-2">
              {INSTITUTIONS.map((inst) => (
                <button
                  key={inst.id}
                  type="button"
                  onClick={() => {
                    setSelected(inst);
                    setStep("credentials");
                  }}
                  className="flex items-center gap-3 rounded-lg border border-border-muted p-3 text-left transition-colors hover:border-foreground/25 hover:bg-foreground/[0.04] cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/40"
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
                    <span className="block text-[11px] tracking-tight text-text-muted">
                      {inst.kind}
                    </span>
                  </span>
                </button>
              ))}
            </div>
            <p className="mt-4 flex items-center gap-1.5 text-[11px] text-text-muted">
              <span className="inline-block h-3 w-3 rounded-full border border-border-muted text-[8px] leading-[10px] text-center">
                🔒
              </span>
              Read-only access · Credentials never stored · 256-bit encryption
            </p>
          </div>
        )}

        {/* ============ STEP 2 · Secure login interface ============ */}
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
                Securing end-to-end encrypted handshake...
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
              {submitting ? "Verifying..." : "Submit Credentials"}
            </button>
            <button
              type="button"
              onClick={() => setStep("institution")}
              className="mt-2 w-full cursor-pointer text-[11px] tracking-tight text-text-muted hover:text-foreground"
            >
              ← Choose a different institution
            </button>
          </div>
        )}

        {/* ============ STEP 3 · Sync success + live hydration ============ */}
        {step === "success" && selected && (
          <div className="p-5 flex flex-col gap-4">
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
            {/* Micro text marquee — hydration status line */}
            <div className="overflow-hidden border-t border-border-muted pt-3">
              <p className="anim-marquee whitespace-nowrap text-center text-[11px] tracking-tight text-text-muted">
                Parsing account balances and ledger data... · Parsing account
                balances and ledger data... · Parsing account balances and
                ledger data...
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}