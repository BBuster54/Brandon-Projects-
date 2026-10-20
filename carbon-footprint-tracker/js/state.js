import { defaults, validate } from "./model.js";
export function readStorage(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
export function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
export function encodeState(input, version, goal = "", baseline = null) {
  const params = new URLSearchParams({ v: version });
  for (const key of Object.keys(defaults)) params.set(key, String(input[key]));
  if (goal !== "") params.set("goal", String(goal));
  if (baseline)
    for (const key of Object.keys(defaults))
      params.set("b_" + key, String(baseline[key]));
  return params.toString();
}
export function decodeState(search, version) {
  const params = new URLSearchParams(search);
  if (!params.size)
    return { input: { ...defaults }, goal: "", baseline: null, message: "" };
  const input = { ...defaults };
  for (const key of Object.keys(defaults))
    if (params.has(key)) input[key] = params.get(key);
  if (!validate(input).valid)
    return {
      input: { ...defaults },
      goal: "",
      baseline: null,
      message: "This link contains invalid inputs. Default values are shown.",
    };
  let baseline = null;
  if (params.has("b_transport")) {
    baseline = { ...defaults };
    for (const key of Object.keys(defaults))
      if (params.has("b_" + key)) baseline[key] = params.get("b_" + key);
    if (!validate(baseline).valid) baseline = null;
  }
  const raw = params.get("goal"),
    n = Number(raw);
  const goal =
    raw !== null &&
    raw !== "" &&
    Number.isFinite(n) &&
    n > 0 &&
    n <= 10000000000
      ? raw
      : "";
  return {
    input,
    goal,
    baseline,
    message:
      params.get("v") && params.get("v") !== version
        ? "This link uses a different model version. Results were recalculated with model " +
          version +
          "."
        : "",
  };
}
export function sanitizeHistory(value) {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (s) =>
        s &&
        typeof s.month === "string" &&
        /^\d{4}-(0[1-9]|1[0-2])$/.test(s.month) &&
        typeof s.version === "string" &&
        validate(s.input ?? {}).valid &&
        Number.isFinite(s.total) &&
        s.total >= 0,
    )
    .slice(-120);
}
export function saveSnapshot(history, snapshot) {
  return [
    ...history.filter(
      (s) =>
        !(
          s.month === snapshot.month &&
          s.input.scope === snapshot.input.scope &&
          Number(s.input.household) === Number(snapshot.input.household)
        ),
    ),
    snapshot,
  ]
    .sort((a, b) => a.month.localeCompare(b.month))
    .slice(-120);
}
