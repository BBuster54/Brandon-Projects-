import { calculate, validate } from "./model.js";
import {
  readStorage,
  writeStorage,
  decodeState,
  encodeState,
  sanitizeHistory,
  saveSnapshot,
} from "./state.js";
import { el, format } from "./render.js";
export function initHistoryAndSharing({
  getInput,
  setInput,
  update,
  data,
  scenarios,
}) {
  const loaded = decodeState(location.search, data.version);
  setInput(loaded.input);
  scenarios.restore(loaded.baseline);
  const goal = document.querySelector("#goal");
  goal.value = loaded.goal;
  const status = document.querySelector("#share-status");
  status.textContent = loaded.message;
  let records = sanitizeHistory(readStorage("carbon-history-v2", [])),
    result = null,
    input = null;
  const now = new Date(),
    month = document.querySelector("#snapshot-month");
  month.value =
    now.getFullYear() + "-" + String(now.getMonth() + 1).padStart(2, "0");
  function url() {
    const value = encodeState(
      getInput(),
      data.version,
      goal.value,
      scenarios.baseline,
    );
    return location.origin + location.pathname + "?" + value;
  }
  function sync() {
    if (validate(getInput()).valid) {
      try {
        history.replaceState(null, "", url());
      } catch {
        /* Sandboxed embeds may disallow URL changes. */
      }
    }
  }
  function renderHistory() {
    const list = document.querySelector("#history-list");
    list.replaceChildren();
    const matching = records.filter(
      (s) =>
        input &&
        s.input.scope === input.scope &&
        Number(s.input.household) === Number(input.household),
    );
    document.querySelector("#history-empty").hidden = matching.length > 0;
    const svg = document.querySelector("#trend");
    svg.replaceChildren();
    svg.toggleAttribute("hidden", matching.length < 2);
    if (matching.length > 1) {
      const ns = "http://www.w3.org/2000/svg",
        path = document.createElementNS(ns, "polyline"),
        max = Math.max(1, ...matching.map((s) => s.total));
      const ordinal = (month) => Number(month.slice(0,4)) * 12 + Number(month.slice(5));
      const first = ordinal(matching[0].month), span = Math.max(1, ordinal(matching.at(-1).month) - first);
      const x = (entry) => 20 + (ordinal(entry.month) - first) / span * 560;
      path.setAttribute(
        "points",
        matching
          .map(
            (s) =>
              x(s) +
              "," +
              (115 - (s.total / max) * 95),
          )
          .join(" "),
      );
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "currentColor");
      path.setAttribute("stroke-width", "3");
      svg.append(path);
      for (let i = 0; i < matching.length; i++) {
        const c = document.createElementNS(ns, "circle");
        c.setAttribute("cx", String(x(matching[i])));
        c.setAttribute("cy", String(115 - (matching[i].total / max) * 95));
        c.setAttribute("r", "4");
        c.setAttribute("fill", "currentColor");
        svg.append(c);
      }
    }
    for (const s of matching) {
      const item = el("li"),
        label = el(
          "span",
          s.month + " · " + format(s.total) + " kg · model " + s.version,
        );
      const button = el("button", "Delete");
      button.setAttribute("aria-label", "Delete snapshot for " + s.month);
      button.addEventListener("click", () => {
        records = records.filter((r) => r !== s);
        persist("Snapshot deleted.");
        renderHistory();
        document.querySelector("#save-month").focus();
      });
      item.append(label, button);
      list.append(item);
    }
    document.querySelector("#clear-history").disabled = records.length === 0;
  }
  function persist(message) {
    const saved = writeStorage("carbon-history-v2", records);
    document.querySelector("#history-status").textContent = saved
      ? message
      : message +
        " Browser storage is blocked or full; changes last only for this session.";
  }
  document.querySelector("#save-month").addEventListener("click", () => {
    if (!result || !validate(getInput()).valid) return;
    input = getInput();
    result = calculate(input, data);
    if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(month.value)) {
      document.querySelector("#history-status").textContent =
        "Choose a valid month.";
      return;
    }
    records = saveSnapshot(records, {
      month: month.value,
      input: { ...input },
      total: result.total,
      version: data.version,
      savedAt: new Date().toISOString(),
    });
    persist(
      "Snapshot saved. Saving the same month and scope replaces its previous value.",
    );
    renderHistory();
  });
  const confirm = document.querySelector("#confirm-clear");
  document.querySelector("#clear-history").addEventListener("click", () => {
    confirm.hidden = false;
    document.querySelector("#confirm-clear-yes").focus();
  });
  document.querySelector("#confirm-clear-no").addEventListener("click", () => {
    confirm.hidden = true;
    document.querySelector("#clear-history").focus();
  });
  document.querySelector("#confirm-clear-yes").addEventListener("click", () => {
    records = [];
    persist("All snapshots cleared.");
    confirm.hidden = true;
    renderHistory();
    document.querySelector("#save-month").focus();
  });
  document.querySelector("#copy-link").addEventListener("click", async () => {
    if (!validate(getInput()).valid) return;
    const text = url();
    try {
      await navigator.clipboard.writeText(text);
      status.textContent =
        "Link copied. It includes your inputs, goal and comparison baseline.";
    } catch {
      const field = document.querySelector("#share-fallback");
      field.hidden = false;
      field.value = text;
      field.focus();
      field.select();
      status.textContent =
        "Copy the selected link manually. Clipboard access is unavailable.";
    }
  });
  goal.addEventListener("input", sync);
  for (const id of ["capture-baseline", "reset-scenario"])
    document.querySelector("#" + id).addEventListener("click", sync);
  document.addEventListener("footprint-update", (e) => {
    ({ input, result } = e.detail);
    sync();
    renderHistory();
    document.querySelector("#save-month").disabled = false;
    document.querySelector("#copy-link").disabled = false;
  });
  document.addEventListener("footprint-invalid", () => {
    document.querySelector("#save-month").disabled = true;
    document.querySelector("#copy-link").disabled = true;
  });
  window.addEventListener("popstate", () => {
    const state = decodeState(location.search, data.version);
    setInput(state.input);
    goal.value = state.goal;
    scenarios.restore(state.baseline);
    status.textContent = state.message;
    update();
  });
  return {
    get records() {
      return records;
    },
  };
}
