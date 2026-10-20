// Canonical app -> portfolio. Never edit the copied demo directly.
import {
  readFile,
  writeFile,
  mkdir,
  readdir,
  copyFile,
} from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, join, dirname } from "node:path";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const target = resolve(
  process.env.PORTFOLIO_ROOT || resolve(root, "../../portfolio-website"),
  "public/demos/carbon-footprint-tracker",
);
const check = process.argv.includes("--check");
const files = ["index.html", "favicon.svg"];
async function gather(folder) {
  for (const e of await readdir(join(root, folder), { withFileTypes: true })) {
    const p = join(folder, e.name);
    if (e.isDirectory()) await gather(p);
    else files.push(p);
  }
}
for (const folder of ["css", "js", "data", "fonts"]) await gather(folder);
let mismatches = [];
for (const file of files) {
  const source = await readFile(join(root, file));
  if (check) {
    try {
      if (!source.equals(await readFile(join(target, file))))
        mismatches.push(file);
    } catch {
      mismatches.push(file);
    }
  } else {
    await mkdir(dirname(join(target, file)), { recursive: true });
    await copyFile(join(root, file), join(target, file));
  }
}
const bytes = await readFile(join(root, "data/factors.json"));
const manifest = {
  canonical:
    "BBuster54/Brandon-Projects-/carbon-footprint-tracker/data/factors.json",
  revision: execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  version: JSON.parse(bytes).version,
  sha256: createHash("sha256").update(bytes).digest("hex"),
};
if (check) {
  try {
    const m = JSON.parse(
      await readFile(join(target, "data/sync-manifest.json")),
    );
    if (m.version !== manifest.version || m.sha256 !== manifest.sha256)
      mismatches.push("data/sync-manifest.json");
  } catch {
    mismatches.push("data/sync-manifest.json");
  }
  if (mismatches.length)
    throw new Error("Sync mismatch: " + mismatches.join(", "));
  console.log(
    "Canonical app and portfolio copy match, including version and SHA-256.",
  );
} else {
  await writeFile(
    join(target, "data/sync-manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  console.log("Synchronized " + files.length + " files to " + target);
}
