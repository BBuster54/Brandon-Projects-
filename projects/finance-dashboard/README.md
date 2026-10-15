# Finance Dashboard

A responsive personal-finance dashboard for monitoring account balances, transactions, market quotes, crypto prices, bills, and portfolio activity. It is built with Next.js App Router, React, TypeScript, and Tailwind CSS.

## What is live data vs. local data

- **Equities:** server-side Alpha Vantage adapter, with batched symbols, input validation, TTL caching, a daily request budget, and last-known-good fallback.
- **Crypto:** server-side CoinGecko adapter, with a short cache, timeout protection, and cached fallback if the upstream provider is unavailable.
- **AI assistant:** optional server-side Gemini integration. It never exposes its API key to the browser and responds with a clear fallback when unavailable.
- **Accounts, budgets, transactions, bills, and goals:** currently local dashboard data. They are intentionally separate from provider-backed market data so the app does not imply a live bank connection it does not have.

The browser refreshes crypto no more than every 10 seconds and equities no more than every 60 seconds. The server caches upstream responses as an additional protection against duplicate requests and provider rate limits.

## Local setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env.local` and add only the provider keys you plan to use. Do not commit `.env.local`.

3. Run the app:

   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000).

## Environment variables

| Variable | Purpose | Required |
| --- | --- | --- |
| `WALL_ST_API_KEY` | Alpha Vantage stock quote, search, profile, and candle data | For live equity data |
| `COINGECKO_API_KEY` | CoinGecko crypto price data | For authenticated CoinGecko access |
| `COINGECKO_TIER` | `demo` or `pro`, which selects the correct CoinGecko host and header | No; defaults to `demo` |
| `GEMINI_API_KEY` | Gemini-powered assistant responses | For live assistant responses |
| `GEMINI_MODEL` | Gemini model identifier | No; defaults to `gemini-3.6-flash` |
| `CRYPTO_CACHE_TTL_MS` | Server-side crypto cache duration in milliseconds | No; defaults to `10000` |
| `STOCK_API_DAILY_BUDGET` | Maximum Alpha Vantage upstream calls allowed per process each day | No; defaults to `25` |

Missing or unavailable providers degrade gracefully: quotes retain their last successful values where possible, and unavailable features explain their status instead of failing the page.

## Quality checks

```bash
npm run lint
npm run build
```

Run both commands before deploying. The production build also checks TypeScript and route compilation.

## Deployment

Deploy the `projects/finance-dashboard` directory as its own Vercel project. Configure the same environment variables in Vercel’s Production environment; keys belong in Vercel environment settings, never in source control. After any credentials have been shared outside their intended secret store, rotate them with the provider before production deployment.
