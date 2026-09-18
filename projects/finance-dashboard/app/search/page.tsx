import type { Metadata } from "next";

import StockResearchTerminal from "./stock-research-terminal";

export const metadata: Metadata = {
  title: "Search",
  description: "Search across markets, tickers, and news.",
};

export default function Search() {
  return <StockResearchTerminal />;
}