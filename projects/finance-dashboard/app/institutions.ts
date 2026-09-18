/**
 * Mock institution catalog for the Plaid-style connect flow.
 * Each institution seeds a mock balance (ingested into dashboard
 * metrics) and a batch of transactions (ingested into the ledger).
 */

export type InstitutionKind =
  | "Depository"
  | "Brokerage"
  | "Retirement"
  | "Crypto";

export type MockTransaction = {
  merchant: string;
  /** Negative = debit, positive = credit. */
  amount: number;
  category: string;
};

export type Institution = {
  id: string;
  name: string;
  /** Single-letter badge rendered in the brand-colored circle. */
  initials: string;
  /** Tailwind bg-* class for the logo badge. */
  color: string;
  kind: InstitutionKind;
  accountLabel: string;
  balance: number;
  /** Depositories roll up into the Cash metric ticker. */
  affectsCash: boolean;
  transactions: MockTransaction[];
};

export const INSTITUTIONS: Institution[] = [
  {
    id: "chase",
    name: "Chase",
    initials: "C",
    color: "bg-blue-700",
    kind: "Depository",
    accountLabel: "Total Checking ••4821",
    balance: 8240.5,
    affectsCash: true,
    transactions: [
      { merchant: "Payroll · Northwind Labs", amount: 2412.0, category: "Income" },
      { merchant: "Blue Bottle Coffee", amount: -6.75, category: "Dining" },
      { merchant: "MTA OMNY", amount: -2.9, category: "Transport" },
    ],
  },
  {
    id: "bank-of-america",
    name: "Bank of America",
    initials: "B",
    color: "bg-red-600",
    kind: "Depository",
    accountLabel: "Advantage Savings ••9302",
    balance: 15980.2,
    affectsCash: true,
    transactions: [
      { merchant: "Transfer from Chase", amount: 500.0, category: "Transfers" },
      { merchant: "Interest Earned", amount: 12.43, category: "Income" },
      { merchant: "Card Payment", amount: -240.0, category: "Shopping" },
    ],
  },
  {
    id: "wells-fargo",
    name: "Wells Fargo",
    initials: "W",
    color: "bg-amber-600",
    kind: "Depository",
    accountLabel: "Everyday Checking ••1174",
    balance: 3120.75,
    affectsCash: true,
    transactions: [
      { merchant: "Chevron", amount: -52.1, category: "Transport" },
      { merchant: "Trader Joe's", amount: -84.32, category: "Groceries" },
      { merchant: "Netflix", amount: -15.49, category: "Subscriptions" },
    ],
  },
  {
    id: "fidelity",
    name: "Fidelity",
    initials: "F",
    color: "bg-emerald-700",
    kind: "Brokerage",
    accountLabel: "Individual Brokerage ••7715",
    balance: 64210.8,
    affectsCash: false,
    transactions: [
      { merchant: "Dividend · AAPL", amount: 24.6, category: "Income" },
      { merchant: "Buy · VOO", amount: -1200.0, category: "Investments" },
      { merchant: "Buy · NVDA", amount: -850.0, category: "Investments" },
    ],
  },
  {
    id: "vanguard",
    name: "Vanguard",
    initials: "V",
    color: "bg-purple-600",
    kind: "Retirement",
    accountLabel: "401(k) Target 2055 ••2208",
    balance: 98540.0,
    affectsCash: false,
    transactions: [
      { merchant: "Contribution", amount: 650.0, category: "Retirement" },
      { merchant: "Employer Match", amount: 325.0, category: "Retirement" },
      { merchant: "Dividend Reinvest", amount: 118.24, category: "Investments" },
    ],
  },
  {
    id: "coinbase",
    name: "Coinbase",
    initials: "C",
    color: "bg-sky-500",
    kind: "Crypto",
    accountLabel: "USD Wallet ••3318",
    balance: 2890.0,
    affectsCash: false,
    transactions: [
      { merchant: "Buy · BTC", amount: -250.0, category: "Crypto" },
      { merchant: "Reward · USDC", amount: 4.12, category: "Crypto" },
      { merchant: "Sell · ETH", amount: 610.0, category: "Crypto" },
    ],
  },
];