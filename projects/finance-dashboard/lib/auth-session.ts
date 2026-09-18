"use client";

/**
 * Auth session manager — persistent sign-in across window reloads.
 *
 * SECURITY POSTURE (read this before shipping to real users)
 * This stores a client-generated token in localStorage. That is correct for
 * the current architecture, where authentication is simulated and there is
 * no server issuing credentials: the token's only job is to remember that
 * this browser completed the gateway, so a refresh doesn't dump the user
 * back at the login wall.
 *
 * It is NOT a security boundary. localStorage is readable by any script on
 * the origin, so an XSS bug exposes it, and nothing here is verified by a
 * server. When real auth lands, the durable part of this module is its
 * SHAPE — `readSession` / `writeSession` / `clearSession` called from the
 * same three places — while the storage itself should move to an
 * httpOnly, Secure, SameSite cookie set by the server. Swapping the three
 * function bodies is then the entire migration.
 */

/** Storage bucket. Versioned so a shape change can invalidate old sessions. */
const SESSION_KEY = "fey-cockpit:session:v1";

/** Sessions older than this are treated as expired and cleared on read. */
const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

export type AuthSession = {
  token: string;
  name: string;
  email: string;
  /** Epoch ms the session was issued. */
  issuedAt: number;
};

/**
 * Every access is guarded by a `typeof window` check. These functions are
 * imported by a client component that Next.js still renders on the server
 * during the initial pass, and touching localStorage there is a hard crash,
 * not a warning.
 */
function hasStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

/** Opaque, non-guessable session id. crypto.randomUUID where available. */
function mintToken(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  /* Fallback for older Safari — adequate for a local session marker. */
  return `sess_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 12)}`;
}

/**
 * Reads the stored session. Returns null for: no storage, no entry, corrupt
 * JSON, a malformed shape, or an expired issue date. A corrupt entry is
 * deleted rather than left to fail on every subsequent read.
 */
export function readSession(): AuthSession | null {
  if (!hasStorage()) return null;
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw) as Partial<AuthSession> | null;
    if (
      !parsed ||
      typeof parsed.token !== "string" ||
      typeof parsed.issuedAt !== "number" ||
      !Number.isFinite(parsed.issuedAt)
    ) {
      window.localStorage.removeItem(SESSION_KEY);
      return null;
    }

    if (Date.now() - parsed.issuedAt > SESSION_TTL_MS) {
      window.localStorage.removeItem(SESSION_KEY);
      return null;
    }

    return {
      token: parsed.token,
      name: typeof parsed.name === "string" ? parsed.name : "",
      email: typeof parsed.email === "string" ? parsed.email : "",
      issuedAt: parsed.issuedAt,
    };
  } catch {
    /* Quota errors, disabled storage, private-mode Safari — all non-fatal. */
    return null;
  }
}

/** Issues and persists a new session. Returns it, or null if storage failed. */
export function writeSession(name: string, email: string): AuthSession | null {
  if (!hasStorage()) return null;
  const session: AuthSession = {
    token: mintToken(),
    name,
    email,
    issuedAt: Date.now(),
  };
  try {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    return session;
  } catch {
    /* Storage full or blocked — sign-in still succeeds for this tab, it
       just won't survive a reload. Degrading beats refusing to log in. */
    return session;
  }
}

/** Drops the session. Called on explicit Log Out only. */
export function clearSession(): void {
  if (!hasStorage()) return;
  try {
    window.localStorage.removeItem(SESSION_KEY);
  } catch {
    /* Nothing actionable — the in-memory state has already been reset. */
  }
}