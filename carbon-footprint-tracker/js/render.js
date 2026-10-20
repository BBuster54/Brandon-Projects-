const formatters = new Map();
export function format(n, digits = 1) {
  if (!formatters.has(digits))
    formatters.set(
      digits,
      new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }),
    );
  return formatters.get(digits).format(n);
}
export const colors = {
  transport: "#a84328",
  electricity: "#3157c5",
  gas: "#8252ad",
  flights: "#196c83",
  waste: "#8b6514",
  diet: "#32735e",
};
export const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
export function renderResults(result, input, lever) {
  document.querySelector("#total").textContent = format(result.total);
  document.querySelector("#projection").textContent =
    format(result.annual) +
    " kg CO₂e / year · " +
    format(result.tonnes, 3) +
    " metric tons / month";
  document.querySelector("#range").textContent =
    "Assumed range: " +
    format(result.low) +
    "–" +
    format(result.high) +
    " kg CO₂e / month";
  document.querySelector("#basis").textContent =
    input.scope === "person"
      ? "Per person, per month"
      : "Whole household, per month";
  const ordered = [...result.categories].sort((a, b) => b.value - a.value);
  const bar = document.querySelector("#footprint-bar"),
    list = document.querySelector("#breakdown");
  bar.replaceChildren();
  list.replaceChildren();
  for (const c of ordered) {
    if (c.value > 0) {
      const piece = el("span");
      piece.style.flexGrow = c.value;
      piece.style.backgroundColor = colors[c.id];
      piece.title = c.label;
      bar.append(piece);
    }
    const row = el("li");
    const dot = el("span", undefined, "swatch");
    dot.style.backgroundColor = colors[c.id];
    row.append(
      dot,
      el("span", c.label),
      el("strong", format(c.value) + " kg"),
      el(
        "span",
        format(result.total ? (c.value / result.total) * 100 : 0) + "%",
        "muted",
      ),
    );
    list.append(row);
  }
  if (result.total === 0)
    bar.append(el("span", "No modeled emissions", "zero-bar"));
  document.querySelector("#insight").textContent =
    ordered[0].label +
    " is the largest category. " +
    (lever.saving > 0
      ? lever.label +
        " to model a saving of " +
        format(lever.saving) +
        " kg per month."
      : "No further savings in the tested changes.");
  document.querySelector("#announcement").textContent =
    "Monthly estimate: " +
    format(result.total) +
    " kilograms CO₂ equivalent. " +
    (input.scope === "person" ? "Per person." : "Whole household.");
}
export function renderMethodology(data) {
  document.querySelector("#factor-version").textContent =
    "Model " + data.version + " · Last reviewed " + data.lastReviewed;
  const details = document.querySelector("#methodology details");
  let rendered = false;
  function renderTable() {
    if (!details.open || rendered) return;
    rendered = true;
    const body = document.querySelector("#factor-table");
    const rows = document.createDocumentFragment();
    for (const f of data.factors) {
      const row = el("tr");
      row.append(
        el("th", f.label),
        el("td", format(f.value, 3) + " " + f.unit),
      );
      const source = el("td");
      if (f.sourceUrl) {
        const a = el("a", f.source);
        a.href = f.sourceUrl;
        a.rel = "noreferrer";
        a.target = "_blank";
        source.append(a);
      } else {
        source.append(el("span", "Source pending", "badge"));
        if (f.source === "illustrative, unsourced")
          source.append(el("small", "Illustrative, unsourced"));
      }
      row.append(
        source,
        el("td", f.year ? String(f.year) : "Unverified"),
        el("td", f.confidence),
      );
      rows.append(row);
    }
    body.replaceChildren(rows);
  }
  details.addEventListener("toggle", renderTable);
  renderTable();
}
