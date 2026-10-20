import {
  readFile,
  writeFile,
  readdir,
  mkdir,
  copyFile,
} from "node:fs/promises";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
const root = resolve(dirname(fileURLToPath(import.meta.url)), ".."),
  target = resolve(
    process.env.PORTFOLIO_ROOT || resolve(root, "../../portfolio-website"),
    "public/demos/urban-insight",
  ),
  check = process.argv.includes("--check");
const files = ["index.html", "favicon.svg"];
async function walk(dir) {
  for (const e of await readdir(join(root, dir), { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p);
    else files.push(p);
  }
}
for (const dir of ["css", "js", "data", "fonts", "vendor"]) await walk(dir);
const hashes = {},
  mismatches = [];
for (const p of files) {
  const bytes = await readFile(join(root, p));
  hashes[p] = createHash("sha256").update(bytes).digest("hex");
  if (check) {
    try {
      if (!bytes.equals(await readFile(join(target, p)))) mismatches.push(p);
    } catch {
      mismatches.push(p);
    }
  } else {
    await mkdir(dirname(join(target, p)), { recursive: true });
    await copyFile(join(root, p), join(target, p));
  }
}
const version = JSON.parse(
    await readFile(join(root, "data/cities.json")),
  ).version,
  manifest = {
    version,
    revision: execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: root,
      encoding: "utf8",
    }).trim(),
    canonical: "BBuster54/Brandon-Projects-/urban-insight-platform",
    files: hashes,
  };
if (check) {
  const saved = JSON.parse(await readFile(join(target, "sync-manifest.json")));
  if (
    saved.version !== version ||
    JSON.stringify(saved.files) !== JSON.stringify(hashes)
  )
    mismatches.push("sync-manifest.json");
  if (mismatches.length)
    throw new Error("Copy mismatch: " + mismatches.join(", "));
  console.log("Urban app, version and SHA-256 checksums match.");
} else {
  await writeFile(
    join(target, "sync-manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  console.log(
    "Synced " +
      files.length +
      " public files. No environment files are included.",
  );
}
