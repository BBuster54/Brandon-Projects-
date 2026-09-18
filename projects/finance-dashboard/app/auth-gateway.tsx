"use client";

/**
 * App Authentication Screen Gateway — the ultra-minimal dark-mode
 * credentials window intercepting the entire workspace shell while
 * `authState` is "login" or "register".
 *
 * Fey aesthetic: pitch-black `grid-lines` canvas, a single `bg-card` panel
 * with razor-thin `border-border-muted` hairlines, `tracking-tight`
 * typography, and one primary white action button. The register variant
 * reveals the Name field — clicking "Create Account" resolves the gateway
 * with `isNewUser: true` so the Plaid-style bank-sync onboarding runs
 * strictly AFTER account creation, never before.
 *
 * Both modes share one form: registration swaps the copy, adds the Name
 * input and tightens password policy (min 8). A short simulated round-trip
 * ("Creating account…" / "Verifying…") keeps the hand-off deliberate, and
 * the pending timer is cleaned up on unmount.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";

export type AuthMode = "login" | "register";

/** Input styling shared by every credential field (matches the connect flow). */
const inputCls =
  "rounded-lg border border-border-muted bg-background/60 px-3 py-2 text-sm tracking-tight text-foreground placeholder:text-text-muted/60 focus:border-foreground/30 focus:outline-none";

export default function AuthGateway({
  mode,
  onModeChange,
  onAuthenticate,
}: {
  mode: AuthMode;
  onModeChange: (mode: AuthMode) => void;
  /** Resolves the gateway — `isNewUser` is true only for registrations. */
  onAuthenticate: (name: string, isNewUser: boolean, email?: string) => void;

}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const isRegister = mode === "register";

  /* Every simulated timer is tracked so unmount never leaks a callback. */
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  useEffect(() => {
    const pending = timers.current;
    return () => pending.forEach(clearTimeout);
  }, []);

  const handleSubmit = useCallback(
    (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (submitting) return;
      setSubmitting(true);
      /* Simulated credential round-trip before resolving the gateway. */
      timers.current.push(
        setTimeout(() => {
          onAuthenticate(isRegister ? name : "", isRegister, email);
        }, 600),
      );
    },
    [submitting, onAuthenticate, isRegister, name, email],
  );

  return (
    <div className="bg-background text-foreground grid-lines flex min-h-screen items-center justify-center p-6 font-sans">
      <div className="terminal-overlay-card w-full max-w-sm rounded-xl border border-border-muted bg-card p-8 tracking-tight">
        {/* Brand mark */}
        <div className="flex flex-col items-center text-center">
          <span
            aria-hidden
            className="flex h-10 w-10 items-center justify-center rounded-lg bg-foreground text-background"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4">
              <path d="M12 2 L22 12 L12 22 L2 12 Z" />
            </svg>
          </span>
          <h1 className="mt-4 text-lg font-semibold tracking-tight text-foreground">
            {isRegister ? "Create your account" : "Welcome back"}
          </h1>
          <p className="mt-1 text-xs tracking-tight text-text-muted">
            {isRegister
              ? "One account for every institution you'll ever link."
              : "Sign in to your financial command center."}
          </p>
        </div>

        {/* Credentials form — the Name field reveals on the register variant */}
        <form onSubmit={handleSubmit} className="mt-7 flex flex-col gap-2.5">
          {isRegister && (
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-text-muted">
                Name
              </span>
              <input
                type="text"
                required
                autoComplete="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Brandon Sparrow"
                className={inputCls}
              />
            </label>
          )}
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wider text-text-muted">
              Email
            </span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className={inputCls}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wider text-text-muted">
              Password
            </span>
            <input
              type="password"
              required
              minLength={isRegister ? 8 : undefined}
              autoComplete={isRegister ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={inputCls}
            />
          </label>

          <button
            type="submit"
            disabled={submitting}
            className="mt-3 w-full cursor-pointer rounded-lg bg-foreground py-2.5 text-sm font-medium tracking-tight text-background transition-opacity hover:opacity-90 disabled:cursor-wait disabled:opacity-60"
          >
            {submitting
              ? isRegister
                ? "Creating account..."
                : "Verifying..."
              : isRegister
                ? "Create Account"
                : "Sign In"}
          </button>
        </form>

        {/* Mode toggle */}
        <button
          type="button"
          onClick={() => onModeChange(isRegister ? "login" : "register")}
          className="mt-4 w-full cursor-pointer text-[11px] tracking-tight text-text-muted hover:text-foreground"
        >
          {isRegister
            ? "Already registered? Sign in"
            : "New here? Create an account"}
        </button>

        <p className="mt-6 border-t border-border-muted pt-4 text-center text-[11px] tracking-tight text-text-muted">
          Read-only bank access · 256-bit encryption · Credentials never stored
        </p>
      </div>
    </div>
  );
}