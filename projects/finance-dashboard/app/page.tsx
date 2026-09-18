/**
 * Dashboard route — server wrapper that owns the static metadata export and
 * renders the interactive client dashboard (modal + data-ingestion state).
 */

import type { Metadata } from "next";

import FinanceDashboard from "./finance-dashboard";

export const metadata: Metadata = {
  title: "Finance Dashboard",
  description:
    "Fey-style personal finance dashboard with a global stock research terminal.",
};

export default function Page() {
  return <FinanceDashboard />;
}