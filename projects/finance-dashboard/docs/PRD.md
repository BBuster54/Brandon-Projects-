# Product Requirements Document — Finance Dashboard

## Overview
A personal financial tracking application for managing accounts, transactions,
budgets, and savings goals. Built primarily for personal use, with architecture
and polish that would support real users later if desired.

## Problem
Tracking spending across multiple accounts and staying within a budget is
tedious without a single, fast overview. Existing tools (banking apps, generic
budgeting apps) are either tied to one bank, bloated with features I don't
need, or don't give a quick answer to "how am I doing this month?"

## Target User
V1: Brandon (personal use, real data, real usage).
Architecture should not preclude adding real multi-user support later, but
V1 does not need to be production-hardened for the public.

## Goals (V1 / MVP)
- Manually track multiple accounts and their balances
- Log, edit, delete, search, filter, and categorize transactions
- See a fast, scannable dashboard: net balance, income, spending, trend
- Set monthly and per-category budgets and track progress against them
- Set savings goals and track progress toward them

## Non-Goals (V1)
- Bank account syncing (Plaid or similar) — manual entry only for now
- Multi-user support / public signup flow beyond basic auth
- AI-powered insights (deferred to a later phase, only after the core app
  is solid and genuinely useful without it)
- Mobile app — responsive web only

## Core User Flow


## Feature List (MVP scope)

### Authentication
- Sign up, login, logout
- Password reset deferred unless time allows (not essential for a single
  personal user, but keep the schema/auth provider capable of it)

### Accounts
- Create, edit, delete an account (name, type, starting balance)
- View current balance per account

### Transactions
- Create, edit, delete a transaction
- Fields: amount, category, account, date, note
- Search, filter by category/date range, sort

### Dashboard
- Net balance across all accounts
- Income vs. spending this month
- Spending trend chart (last 30/90 days)
- Category breakdown
- Recent transactions list

### Budgets
- Set a monthly budget per category
- Visual progress (spent vs. budgeted)
- Remaining amount, clearly flagged if over budget

### Goals
- Create a savings goal (name, target amount, target date)
- Track progress (manual contributions or calculated from account balance
  changes — decide during architecture phase)
- Visual progress toward each goal

## Success Criteria (V1 done)
- I can log in and see my real financial state accurately reflected
- Adding/editing/deleting a transaction updates the dashboard, budgets, and
  goals correctly, immediately
- The dashboard is scannable in under 10 seconds
- Deployed, real URL, no placeholder/fake data in the shipped version
- Passes the project's own QA checklist (see docs/TESTING.md)

## Deferred to Later Phases
- Recurring transactions
- CSV import/export
- Month-over-month comparison views
- AI financial insights ("why did I spend more this month?")

