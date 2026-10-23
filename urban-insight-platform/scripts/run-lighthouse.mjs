import { chromium } from "@playwright/test";
import { spawnSync } from "node:child_process";
const result = spawnSync(
  process.execPath,
  ["node_modules/@lhci/cli/src/cli.js", "autorun"],
  {
    stdio: "inherit",
    env: { ...process.env, CHROME_PATH: chromium.executablePath() },
  },
);
process.exitCode = result.status ?? 1;
