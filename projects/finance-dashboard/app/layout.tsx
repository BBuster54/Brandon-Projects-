"use client";

/**
 * Root layout — global shell for the finance dashboard.
 *
 * Hosts the Stock Research Terminal as a floating command center:
 * - Cmd/Ctrl + K toggles the terminal from anywhere (browser default intercepted).
 * - Escape dismisses it instantly; clicking the blurred backdrop also closes it.
 *
 * Phase 6 — the layout also mounts <MarketDataProvider>, which owns the one
 * and only market polling loop in the application. Its position here is
 * load-bearing, not incidental: the ⌘K terminal is rendered BY THIS LAYOUT,
 * as a sibling of {children}, so a provider mounted inside the dashboard
 * could never reach it. Hoisting it to the layout puts the terminal, the
 * dashboard cards and the bottom marquee ticker on one shared price
 * dictionary — which is what makes cent-level drift between them
 * structurally impossible rather than merely unlikely.
 */

import { useEffect, useState } from "react";
import { Geist, Geist_Mono } from "next/font/google";

import { MarketDataProvider } from "@/lib/market-store";
import StockResearchTerminal from "./search/stock-research-terminal";
import { TERMINAL_VISIBILITY_EVENT } from "./terminal-events";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function RootLayout({ children }: LayoutProps<"/">) {
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);

  /* Global keyboard shortcuts for the Stock Research Terminal. */
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        /* Intercept the browser's default Cmd/Ctrl+K behavior. */
        e.preventDefault();
        setIsTerminalOpen((open) => !open);
      } else if (e.key === "Escape") {
        setIsTerminalOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  /* Broadcast the terminal's visibility so surfaces rendered deeper in the
     tree (e.g. the dashboard's bottom market marquee ticker) can react to
     the ⌘K overlay without prop drilling through the layout boundary. */
  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent<boolean>(TERMINAL_VISIBILITY_EVENT, {
        detail: isTerminalOpen,
      }),
    );
  }, [isTerminalOpen]);

  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <MarketDataProvider>
          {isTerminalOpen && (
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Stock Research Terminal"
              onClick={(e) => {
                /* Clicking the blurred backdrop dismisses the terminal. */
                if (e.target === e.currentTarget) setIsTerminalOpen(false);
              }}
              className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md flex items-center justify-center p-6"
            >
              <div className="terminal-overlay-card w-full max-w-6xl h-[85vh] bg-card border border-border-muted rounded-2xl shadow-2xl overflow-hidden">
                <StockResearchTerminal
                  variant="modal"
                  onClose={() => setIsTerminalOpen(false)}
                />
              </div>
            </div>
          )}
          {children}
        </MarketDataProvider>
      </body>
    </html>
  );
}