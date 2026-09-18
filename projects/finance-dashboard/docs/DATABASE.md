# Database Schema — Finance Dashboard

## Overview

PostgreSQL database with Supabase, using row-level security (RLS) to ensure users can only access their own data.

## Tables

### accounts

| Column           | Type          | Constraints                                           |
| ---------------- | ------------- | ----------------------------------------------------- |
| id               | uuid          | PRIMARY KEY, DEFAULT gen_random_uuid()                |
| user_id          | uuid          | NOT NULL, REFERENCES auth.users(id) ON DELETE CASCADE |
| name             | text          | NOT NULL                                              |
| type             | account_type  | NOT NULL                                              |
| starting_balance | numeric(12,2) | NOT NULL                                              |
| created_at       | timestamptz   | NOT NULL, DEFAULT NOW()                               |
| updated_at       | timestamptz   | NOT NULL, DEFAULT NOW()                               |

### categories

| Column     | Type        | Constraints                                           |
| ---------- | ----------- | ----------------------------------------------------- |
| id         | uuid        | PRIMARY KEY, DEFAULT gen_random_uuid()                |
| user_id    | uuid        | NOT NULL, REFERENCES auth.users(id) ON DELETE CASCADE |
| name       | text        | NOT NULL                                              |
| color      | text        | NOT NULL                                              |
| icon       | text        | NOT NULL                                              |
| created_at | timestamptz | NOT NULL, DEFAULT NOW()                               |
| updated_at | timestamptz | NOT NULL, DEFAULT NOW()                               |

### transactions

| Column      | Type          | Constraints                                           |
| ----------- | ------------- | ----------------------------------------------------- |
| id          | uuid          | PRIMARY KEY, DEFAULT gen_random_uuid()                |
| user_id     | uuid          | NOT NULL, REFERENCES auth.users(id) ON DELETE CASCADE |
| account_id  | uuid          | NOT NULL, REFERENCES accounts(id) ON DELETE CASCADE   |
| category_id | uuid          | REFERENCES categories(id) ON DELETE SET NULL          |
| amount      | numeric(12,2) | NOT NULL                                              |
| date        | date          | NOT NULL                                              |
| note        | text          |                                                       |
| created_at  | timestamptz   | NOT NULL, DEFAULT NOW()                               |
| updated_at  | timestamptz   | NOT NULL, DEFAULT NOW()                               |

### budgets

| Column      | Type          | Constraints                                           |
| ----------- | ------------- | ----------------------------------------------------- |
| id          | uuid          | PRIMARY KEY, DEFAULT gen_random_uuid()                |
| user_id     | uuid          | NOT NULL, REFERENCES auth.users(id) ON DELETE CASCADE |
| category_id | uuid          | NOT NULL, REFERENCES categories(id) ON DELETE CASCADE |
| month       | integer       | NOT NULL, CHECK (month BETWEEN 1 AND 12)              |
| year        | integer       | NOT NULL, CHECK (year BETWEEN 2000 AND 2100)          |
| amount      | numeric(12,2) | NOT NULL                                              |
| created_at  | timestamptz   | NOT NULL, DEFAULT NOW()                               |
| updated_at  | timestamptz   | NOT NULL, DEFAULT NOW()                               |

Unique constraint: (user_id, category_id, month, year)

### goals

| Column        | Type          | Constraints                                           |
| ------------- | ------------- | ----------------------------------------------------- |
| id            | uuid          | PRIMARY KEY, DEFAULT gen_random_uuid()                |
| user_id       | uuid          | NOT NULL, REFERENCES auth.users(id) ON DELETE CASCADE |
| name          | text          | NOT NULL                                              |
| target_amount | numeric(12,2) | NOT NULL                                              |
| target_date   | date          |                                                       |
| note          | text          |                                                       |
| created_at    | timestamptz   | NOT NULL, DEFAULT NOW()                               |
| updated_at    | timestamptz   | NOT NULL, DEFAULT NOW()                               |

### goal_contributions

| Column     | Type          | Constraints                                      |
| ---------- | ------------- | ------------------------------------------------ |
| id         | uuid          | PRIMARY KEY, DEFAULT gen_random_uuid()           |
| goal_id    | uuid          | NOT NULL, REFERENCES goals(id) ON DELETE CASCADE |
| amount     | numeric(12,2) | NOT NULL                                         |
| date       | date          | NOT NULL                                         |
| note       | text          |                                                  |
| created_at | timestamptz   | NOT NULL, DEFAULT NOW()                          |
| updated_at | timestamptz   | NOT NULL, DEFAULT NOW()                          |

## Enums

### account_type

```sql
CREATE TYPE account_type AS ENUM (
  'checking',
  'savings',
  'credit_card',
  'investment',
  'cash',
  'other'
);
```

## Row-Level Security Policies

### accounts

```sql
-- Enable RLS
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;

-- Select (read)
CREATE POLICY "Users can view own accounts"
  ON accounts FOR SELECT
  USING (auth.uid() = user_id);

-- Insert
CREATE POLICY "Users can insert own accounts"
  ON accounts FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Update
CREATE POLICY "Users can update own accounts"
  ON accounts FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Delete
CREATE POLICY "Users can delete own accounts"
  ON accounts FOR DELETE
  USING (auth.uid() = user_id);
```

### categories

```sql
ALTER TABLE categories ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own categories"
  ON categories FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own categories"
  ON categories FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own categories"
  ON categories FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own categories"
  ON categories FOR DELETE
  USING (auth.uid() = user_id);
```

### transactions

```sql
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own transactions"
  ON transactions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own transactions"
  ON transactions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own transactions"
  ON transactions FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own transactions"
  ON transactions FOR DELETE
  USING (auth.uid() = user_id);
```

### budgets

```sql
ALTER TABLE budgets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own budgets"
  ON budgets FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own budgets"
  ON budgets FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own budgets"
  ON budgets FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own budgets"
  ON budgets FOR DELETE
  USING (auth.uid() = user_id);
```

### goals

```sql
ALTER TABLE goals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own goals"
  ON goals FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own goals"
  ON goals FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own goals"
  ON goals FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own goals"
  ON goals FOR DELETE
  USING (auth.uid() = user_id);
```

### goal_contributions

```sql
ALTER TABLE goal_contributions ENABLE ROW LEVEL SECURITY;

-- Select (read)
CREATE POLICY "Users can view own goal contributions"
  ON goal_contributions FOR SELECT
  USING (
    EXISTS (
      SELECT 1
      FROM goals
      WHERE goals.id = goal_contributions.goal_id
        AND goals.user_id = auth.uid()
    )
  );

-- Insert
CREATE POLICY "Users can insert own goal contributions"
  ON goal_contributions FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1
      FROM goals
      WHERE goals.id = goal_contributions.goal_id
        AND goals.user_id = auth.uid()
    )
  );

-- Update
CREATE POLICY "Users can update own goal contributions"
  ON goal_contributions FOR UPDATE
  USING (
    EXISTS (
      SELECT 1
      FROM goals
      WHERE goals.id = goal_contributions.goal_id
        AND goals.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1
      FROM goals
      WHERE goals.id = goal_contributions.goal_id
        AND goals.user_id = auth.uid()
    )
  );

-- Delete
CREATE POLICY "Users can delete own goal contributions"
  ON goal_contributions FOR DELETE
  USING (
    EXISTS (
      SELECT 1
      FROM goals
      WHERE goals.id = goal_contributions.goal_id
        AND goals.user_id = auth.uid()
    )
  );
```

## Triggers

### updated_at Auto-Update

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

CREATE TRIGGER update_accounts_updated_at
  BEFORE UPDATE ON accounts
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_categories_updated_at
  BEFORE UPDATE ON categories
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transactions_updated_at
  BEFORE UPDATE ON transactions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_budgets_updated_at
  BEFORE UPDATE ON budgets
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_goals_updated_at
  BEFORE UPDATE ON goals
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_goal_contributions_updated_at
  BEFORE UPDATE ON goal_contributions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
```

## Computed Values

### Account Current Balance

```sql
SELECT
  id,
  name,
  type,
  starting_balance,
  starting_balance + COALESCE(
    (SELECT SUM(amount) FROM transactions WHERE account_id = accounts.id),
    0
  ) AS current_balance
FROM accounts
WHERE user_id = auth.uid();
```

### Goal Current Amount

```sql
SELECT
  id,
  name,
  target_amount,
  COALESCE(
    (SELECT SUM(amount) FROM goal_contributions WHERE goal_id = goals.id),
    0
  ) AS current_amount
FROM goals
WHERE user_id = auth.uid();
```

## Design Decisions

1. **No stored current_balance** - Computed as `starting_balance + SUM(transactions.amount)` to prevent drift
2. **goal_contributions table** - Explicit manual contributions, simpler than deriving from transactions
3. **NUMERIC(12,2)** for all money fields (max 999,999,999.99)
4. **Auto-updating updated_at** via trigger on all tables
5. **user_id on every table** for strict RLS isolation
6. **amount sign convention**: positive = income, negative = expense
7. **budgets scoped by month/year** for historical tracking and future month-over-month comparisons
8. **categories as user-defined** for flexibility
9. **CHECK constraint on budgets.month** ensures valid month values (1-12)
10. **CHECK constraint on budgets.year** ensures valid year values (2000-2100)
11. **goal_contributions RLS via goals table** ensures users can only access contributions for their own goals
