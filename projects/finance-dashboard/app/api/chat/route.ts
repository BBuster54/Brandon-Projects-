import { NextResponse } from "next/server";

/**
 * POST /api/chat — server-side AI Advisor engine.
 *
 * GEMINI_API_KEY is read from `process.env` inside this handler and sent as
 * an `x-goog-api-key` HEADER. It is never placed in the URL (query strings
 * are captured by proxy and platform access logs) and never reaches the
 * browser: the client posts a question plus a context payload, and receives
 * only rendered markdown back.
 *
 * Request  : { question, history?, context? }
 * Response : { reply: string, source: "gemini" | "fallback", model?: string }
 *
 * The route always answers 200 with a usable `reply`. A 500 here would blank
 * the advisor column; a graceful degraded snapshot keeps the cockpit honest
 * about being offline while still telling the user something true.
 */

/* Node runtime: the provider module and this handler both touch server-only
   env, and the Edge runtime's stricter fetch semantics buy nothing here. */
export const runtime = "nodejs";
/* Advice is a pure function of a live payload — never cache it. */
export const dynamic = "force-dynamic";

const GEMINI_MODEL = process.env.GEMINI_MODEL ?? "gemini-3.6-flash";
const GEMINI_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

/* -------------------------------------------------------------------------- */
/*                           Request payload shapes                           */
/* -------------------------------------------------------------------------- */

type ChatTurn = { role: "advisor" | "user"; body: string };

type CryptoPosition = {
  token: string;
  amount: number;
  price: number;
  deltaPct: number;
};

type EquityPosition = {
  symbol: string;
  shares?: number;
  price: number;
  dayPct: number;
};

type ExpenseCategory = {
  name: string;
  spent: number;
  budget?: number;
  progress?: number;
};

type CashFlowProfile = {
  income: number;
  rent: number;
  utilities: number;
  discretionary: number;
  saved?: boolean;
};

type DashboardContext = {
  netWorth?: number;
  cash?: number;
  liabilities?: number;
  monthlyIncome?: number;
  monthlyOutlays?: number;
  cashFlowProfile?: CashFlowProfile;
  connectedInstitutions?: { name: string; balance: number }[];
  connectedCount?: number;
  expenseCategories?: ExpenseCategory[];
  crypto?: CryptoPosition[];
  equities?: EquityPosition[];
  ledgerCount?: number;
  emergencyMonths?: number;
  activeView?: string;
};

type ChatRequest = {
  question?: unknown;
  history?: unknown;
  context?: unknown;
};

/* -------------------------------------------------------------------------- */
/*                                  Helpers                                   */
/* -------------------------------------------------------------------------- */

function usd(n: number): string {
  return `$${Math.round(n).toLocaleString("en-US")}`;
}

function finite(v: unknown, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Renders the live dashboard payload as a compact, unambiguous briefing
 * block. Gemini reasons far more reliably over labelled prose than over raw
 * JSON, and an explicit "$0 means not yet linked" note is what stops the
 * model from congratulating a new user on a debt-free net worth of zero.
 */
function renderContextBrief(ctx: DashboardContext): string {
  const lines: string[] = [];

  const netWorth = finite(ctx.netWorth);
  const cash = finite(ctx.cash);
  const income = finite(ctx.monthlyIncome);
  const outlays = finite(ctx.monthlyOutlays);
  const capacity = income - outlays;

  lines.push("## Balance sheet");
  lines.push(
    `- Net worth: ${usd(netWorth)} | Liquid cash: ${usd(cash)} | Liabilities: ${usd(finite(ctx.liabilities))}`,
  );

  const institutions = Array.isArray(ctx.connectedInstitutions)
    ? ctx.connectedInstitutions
    : [];
  lines.push(
    institutions.length > 0
      ? `- Linked institutions (${institutions.length}): ${institutions
          .map((i) => `${i.name} (${usd(finite(i.balance))})`)
          .join(", ")}`
      : "- Linked institutions: NONE. The user has not connected any bank or brokerage yet.",
  );

  lines.push("");
  lines.push("## Monthly cash flow");
  const profile = ctx.cashFlowProfile;
  if (profile && (finite(profile.income) > 0 || finite(profile.rent) > 0)) {
    lines.push(
      `- Take-home income: ${usd(finite(profile.income))} | Rent/mortgage: ${usd(finite(profile.rent))} | Utilities & insurance: ${usd(finite(profile.utilities))} | Discretionary: ${usd(finite(profile.discretionary))}`,
    );
    lines.push(
      `- Source: ${profile.saved ? "manually entered and saved by the user" : "derived from ledger activity"}`,
    );
  } else {
    lines.push(
      "- Cash flow profile is UNCONFIGURED (all zeros). Income and outlays have not been entered and no ledger debits exist.",
    );
  }
  lines.push(
    `- Net monthly capacity: ${capacity >= 0 ? "+" : "-"}${usd(Math.abs(capacity))} | Savings rate: ${
      income > 0 ? `${((capacity / income) * 100).toFixed(1)}%` : "undefined (no income recorded)"
    }`,
  );
  if (ctx.emergencyMonths !== undefined) {
    lines.push(
      `- Emergency runway: ${finite(ctx.emergencyMonths).toFixed(1)} months of essentials`,
    );
  }

  const categories = Array.isArray(ctx.expenseCategories)
    ? ctx.expenseCategories.filter((c) => finite(c.spent) > 0)
    : [];
  lines.push("");
  lines.push("## Spending categories");
  if (categories.length > 0) {
    for (const c of categories.sort((a, b) => finite(b.spent) - finite(a.spent))) {
      const budget = finite(c.budget);
      lines.push(
        `- ${c.name}: ${usd(finite(c.spent))} spent${budget > 0 ? ` of ${usd(budget)} budget (${finite(c.progress)}% used)` : ""}`,
      );
    }
  } else {
    lines.push(
      "- No spending recorded this month. Category budgets exist but no debits have posted.",
    );
  }

  const equities = Array.isArray(ctx.equities) ? ctx.equities : [];
  const held = equities.filter((e) => finite(e.shares) > 0);
  lines.push("");
  lines.push("## Investments");
  if (held.length > 0) {
    for (const e of held) {
      lines.push(
        `- ${e.symbol}: ${finite(e.shares)} shares @ ${usd(finite(e.price))} (${finite(e.dayPct).toFixed(2)}% today)`,
      );
    }
  } else if (equities.length > 0) {
    lines.push(
      `- WATCHLIST ONLY (zero shares held): ${equities.map((e) => e.symbol).join(", ")}. These are quotes, not positions.`,
    );
  } else {
    lines.push("- No equities tracked.");
  }

  const crypto = Array.isArray(ctx.crypto) ? ctx.crypto : [];
  const cryptoHeld = crypto.filter((c) => finite(c.amount) > 0);
  if (cryptoHeld.length > 0) {
    for (const c of cryptoHeld) {
      lines.push(
        `- ${c.token}: ${finite(c.amount)} @ ${usd(finite(c.price))} (${finite(c.deltaPct).toFixed(2)}% 24h)`,
      );
    }
  } else if (crypto.length > 0) {
    lines.push(
      `- Crypto tracked at zero balance: ${crypto.map((c) => c.token).join(", ")}. Prices are live, holdings are not.`,
    );
  }

  if (ctx.activeView) {
    lines.push("");
    lines.push(`## UI context`);
    lines.push(`- The user is currently on the "${ctx.activeView}" view.`);
  }

  return lines.join("\n");
}

/**
 * System instruction. The hard constraints exist because a wealth advisor
 * that invents numbers is worse than no advisor at all: the model is bound
 * to the supplied figures, forbidden from inventing balances, and required
 * to treat a $0 cockpit as an onboarding state rather than a diagnosis.
 */
const SYSTEM_INSTRUCTION = `You are the in-house wealth advisor inside a premium personal finance terminal. You are warm, direct and precise — the tone of a trusted private banker who respects the client's intelligence and never condescends.

OUTPUT FORMAT (strict):
- Reply in clean markdown bullet points. Every bullet starts with "- ".
- 3 to 5 bullets maximum. No headings, no preamble, no sign-off, no emoji.
- Use **bold** for the single most important figure or action in each bullet.
- Keep each bullet to one or two sentences. Total reply under 130 words.

ACCURACY RULES (non-negotiable):
- Use ONLY the figures in the DASHBOARD CONTEXT below. Never invent, estimate or hallucinate a balance, holding, institution or transaction that is not listed.
- If a figure needed to answer is missing or zero, say so plainly and name the exact step that would supply it (link an institution, save the cash flow profile).
- A $0 net worth means the user has not linked accounts yet. Treat it as an onboarding state, never as poverty, failure, or a debt-free achievement.
- "Watchlist only" means zero shares held. Never describe watchlist tickers as holdings or compute a position value for them.
- Live market prices move constantly; never promise a future price or guarantee a return.

ADVICE RULES:
- Every reply must contain at least one concrete, actionable next step the user can take inside this app or this week.
- Prefer specific numbers over generic principles: reference the 50/30/20 baseline, their actual savings rate, their actual runway.
- You are not a licensed financial advisor. Do not recommend specific securities to buy or sell; discuss allocation, risk and cash flow structure instead.
- If the question is off-topic for personal finance, answer briefly and steer back to their cockpit.`;

/** Local degraded reply — used when the key is absent or Gemini is down. */
function fallbackReply(question: string, ctx: DashboardContext): string {
  const netWorth = finite(ctx.netWorth);
  const cash = finite(ctx.cash);
  const capacity = finite(ctx.monthlyIncome) - finite(ctx.monthlyOutlays);
  const linked = finite(ctx.connectedCount);

  if (netWorth === 0 && linked === 0) {
    return [
      `- The advisor engine is **offline**, so here is the local snapshot for "${question}".`,
      "- Your cockpit is still at a **$0 baseline** with no linked institutions, so there are no balances to reason over yet.",
      "- Next step: **link one institution** (read-only) or save a manual cash flow profile — either one activates every metric on this dashboard.",
    ].join("\n");
  }

  return [
    `- The advisor engine is **offline**, so here is the local snapshot for "${question}".`,
    `- Net worth **${usd(netWorth)}**, liquid cash **${usd(cash)}**, net monthly capacity **${capacity >= 0 ? "+" : "-"}${usd(Math.abs(capacity))}**.`,
    `- **${linked}** institution${linked === 1 ? "" : "s"} linked; live market data keeps streaming while the advisor reconnects.`,
  ].join("\n");
}

/* -------------------------------------------------------------------------- */
/*                                  Handler                                   */
/* -------------------------------------------------------------------------- */

export async function POST(request: Request) {
  let body: ChatRequest;
  try {
    body = (await request.json()) as ChatRequest;
  } catch {
    return NextResponse.json(
      { reply: "- I couldn't read that request. Try sending it again.", source: "fallback" },
      { status: 400 },
    );
  }

  const question =
    typeof body.question === "string" ? body.question.trim().slice(0, 2000) : "";
  if (!question) {
    return NextResponse.json(
      { reply: "- Ask me something about your cash flow, budgets or allocation.", source: "fallback" },
      { status: 400 },
    );
  }

  const context: DashboardContext =
    body.context && typeof body.context === "object"
      ? (body.context as DashboardContext)
      : {};

  /* Last 8 turns only — enough for pronoun resolution, cheap on tokens. */
  const history: ChatTurn[] = Array.isArray(body.history)
    ? (body.history as ChatTurn[])
        .filter(
          (t) =>
            t &&
            typeof t.body === "string" &&
            (t.role === "user" || t.role === "advisor"),
        )
        .slice(-8)
    : [];

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    /* Missing key is a deployment problem, not a user problem — degrade. */
    return NextResponse.json({
      reply: fallbackReply(question, context),
      source: "fallback",
    });
  }

  /* Gemini roles are "user" and "model"; the advisor maps to "model". */
  const contents = [
    ...history.map((turn) => ({
      role: turn.role === "user" ? "user" : "model",
      parts: [{ text: turn.body }],
    })),
    {
      role: "user",
      parts: [
        {
          text: `DASHBOARD CONTEXT (live, authoritative):\n${renderContextBrief(context)}\n\nCLIENT QUESTION:\n${question}`,
        },
      ],
    },
  ];

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20_000);

  try {
    const res = await fetch(GEMINI_ENDPOINT, {
      method: "POST",
      signal: controller.signal,
      cache: "no-store",
      headers: {
        "content-type": "application/json",
        /* Header auth — the key never touches the URL or any access log. */
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
        contents,
        generationConfig: {
          temperature: 0.55,
          topP: 0.9,
          maxOutputTokens: 600,
        },
        safetySettings: [],
      }),
    });

    if (!res.ok) {
      /* Surface the upstream status in server logs only — never to the
         client, since Gemini error bodies can echo request metadata. */
      console.error(`[api/chat] gemini ${res.status}: ${await res.text()}`);
      return NextResponse.json({
        reply: fallbackReply(question, context),
        source: "fallback",
      });
    }

    const data = (await res.json()) as {
      candidates?: {
        content?: { parts?: { text?: string }[] };
        finishReason?: string;
      }[];
    };

    const reply =
      data.candidates?.[0]?.content?.parts
        ?.map((p) => p.text ?? "")
        .join("")
        .trim() ?? "";

    if (!reply) {
      return NextResponse.json({
        reply: fallbackReply(question, context),
        source: "fallback",
      });
    }

    return NextResponse.json({ reply, source: "gemini", model: GEMINI_MODEL });
  } catch (err) {
    console.error("[api/chat] engine unreachable", err);
    return NextResponse.json({
      reply: fallbackReply(question, context),
      source: "fallback",
    });
  } finally {
    clearTimeout(timer);
  }
}