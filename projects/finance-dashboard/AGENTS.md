<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->


## Project Standards

Read /docs before making architectural changes.
Do not rewrite working code without a reason.
Do not introduce dependencies unnecessarily.
Follow the existing design system.
Maintain TypeScript strictness.
Do not expose secrets.
Validate inputs.
Consider accessibility.
Consider responsive behavior.
Consider loading/error/empty states.
Write tests for business logic.
Run lint/typecheck/tests after implementation.
Update documentation when architecture changes.
Do not claim success without testing.
