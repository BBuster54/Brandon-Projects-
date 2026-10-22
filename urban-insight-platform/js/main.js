import { loadCities } from "./data/cities.js";
import {
  metricDefinitions,
  missingMetric,
  filteredCities,
} from "./data/normalize.js";
import {
  defaultState,
  parseState,
  syncURL,
  safeStorage,
  storageRead,
  storageWrite,
} from "./state.js";
import { renderCity, renderCityTable } from "./render/metrics.js";
import { renderRank } from "./render/charts.js";
import { renderMethods } from "./render/provenance.js";
// Presentation only: filters start collapsed on phones and dock open on larger screens.
const phoneLayout = matchMedia("(max-width: 600px)");
const dockFilters = () => {
  document.querySelector("#filter-disclosure").open = !phoneLayout.matches;
};
dockFilters();
phoneLayout.addEventListener("change", dockFilters);
let cities = [],
  records = {},
  samples = {},
  state = { ...defaultState };
let mapController = null;
import { createAirProvider } from "./data/providers/openMeteo.js";
import { initMap } from "./render/map.js";
import { loadACS } from "./data/providers/acs.js";
let airProvider,
  trends = {};
let officialLoaded = false;
async function refreshAir() {
  const button = document.querySelector("#refresh");
  button.disabled = true;
  button.textContent = "Loading air quality…";
  const result = await airProvider.load();
  trends = result.trends;
  for (const city of cities)
    Object.assign(records[city.id], result.records[city.id]);
  document.querySelector("#data-status").textContent =
    result.status + (officialLoaded ? "" : " Official data not yet loaded.");
  render();
  button.disabled = false;
  button.textContent = "Refresh air quality";
}
document.querySelector("#refresh").addEventListener("click", refreshAir);
import { renderCompare, renderTrend } from "./render/compare.js";
import { downloadCSV } from "./export.js";
document.querySelector("#compare").addEventListener("change", (e) => {
  state.compare = e.target.value;
  render();
});
document.querySelector("#export").addEventListener("click", () => {
  downloadCSV(
    filteredCities(cities, records, state.category),
    records,
    state.view,
  );
  document.querySelector("#export-status").textContent =
    "CSV exported with sources and reference dates. Sample data excluded.";
});
document
  .querySelector("#print")
  .addEventListener("click", () => window.print());
document.querySelector("#share").addEventListener("click", async () => {
  const input = document.querySelector("#share-url");
  input.value = location.href;
  try {
    await navigator.clipboard.writeText(location.href);
    document.querySelector("#export-status").textContent =
      "View URL copied. Recipients fetch their own current air quality; this is not a frozen snapshot.";
  } catch {
    input.hidden = false;
    input.focus();
    input.select();
    document.querySelector("#export-status").textContent =
      "Copy the selected URL to share this view.";
  }
});
const themeButton = document.querySelector("#theme");
let theme = storageRead(safeStorage("localStorage"), "urban-theme", "system");
function applyTheme() {
  if (!["light", "dark"].includes(theme)) {
    theme = "system";
    delete document.documentElement.dataset.theme;
  } else document.documentElement.dataset.theme = theme;
  themeButton.textContent = "Theme: " + theme;
}
applyTheme();
themeButton.addEventListener("click", () => {
  theme = { system: "light", light: "dark", dark: "system" }[theme];
  applyTheme();
  storageWrite(safeStorage("localStorage"), "urban-theme", theme);
});
function selectCity(id) {
  state.city = id;
  render();
  document
    .querySelector("#city-title")
    .scrollIntoView({ block: "nearest", behavior: "instant" });
}
function render() {
  for (const key of ["city", "metric", "category", "sort"])
    document.querySelector("#" + key).value = state[key];
  for (const b of document.querySelectorAll("[role=tab]")) {
    const active = b.dataset.view === state.view;
    b.setAttribute("aria-selected", String(active));
    b.tabIndex = active ? 0 : -1;
  }
  document
    .querySelector("#view-panel")
    .setAttribute("aria-labelledby", "tab-" + state.view);
  const city = cities.find((c) => c.id === state.city) ?? cities[0],
    visible = filteredCities(cities, records, state.category);
  renderCity(city, records[city.id], state.view);
  renderCityTable(visible, records, selectCity);
  renderRank(visible, records, state);
  document.querySelector("#visible-count").textContent =
    visible.length + " " + (visible.length === 1 ? "city" : "cities");
  const metrics = document.querySelector("#metric-list");
  metrics.classList.remove("skeleton");
  metrics.setAttribute("aria-busy", "false");
  mapController?.update(visible, records, state);
  syncURL(state);
  document.querySelector("#announcement").textContent =
    city.name +
    ", " +
    state.view +
    ". " +
    visible.length +
    " cities match the filter.";
  renderCompare(cities, records, state);
  renderTrend(city, trends[city.id] ?? [], records[city.id].aqi);
  if (
    document.querySelector("#methods-disclosure").open ||
    document.querySelector("#samples-disclosure").open
  )
    renderMethods(cities, records, samples);
}
for (const key of ["city", "metric", "category", "sort"])
  document.querySelector("#" + key).addEventListener("change", (e) => {
    state[key] = e.target.value;
    render();
  });
document.querySelector("#reset").addEventListener("click", () => {
  state = { ...defaultState };
  render();
});
const tabs = [...document.querySelectorAll("[role=tab]")];
for (const tab of tabs) {
  tab.addEventListener("click", () => {
    state.view = tab.dataset.view;
    render();
  });
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let index = tabs.indexOf(tab);
    index =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
            tabs.length;
    tabs[index].focus();
    state.view = tabs[index].dataset.view;
    render();
  });
}
window.addEventListener("popstate", () => {
  state = parseState(location.search, cities);
  render();
});
for (const id of ["methods-disclosure", "samples-disclosure"])
  document.querySelector("#" + id).addEventListener("toggle", () => {
    if (document.querySelector("#" + id).open)
      renderMethods(cities, records, samples);
  });
try {
  const [meta, acs, sampleResponse] = await Promise.all([
    loadCities(),
    loadACS(),
    fetch("data/sample-metrics.json", { signal: AbortSignal.timeout(8000) }),
  ]);
  if (!sampleResponse.ok) throw new Error("Data unavailable");
  cities = meta;
  const sampleData = await sampleResponse.json();
  officialLoaded = Object.keys(acs.cities ?? {}).length > 0;
  samples = Object.fromEntries(sampleData.cities.map((c) => [c.id, c.metrics]));
  state = parseState(location.search, cities);
  for (const c of cities) {
    document
      .querySelector("#city")
      .append(new Option(c.name + ", " + c.state, c.id));
    document.querySelector("#compare").append(new Option(c.name, c.id));
    records[c.id] = {
      ...Object.fromEntries(
        Object.keys(metricDefinitions).map((k) => [k, missingMetric(k)]),
      ),
      ...acs.cities[c.id]?.metrics,
      aqi: samples[c.id].aqi,
    };
  }
  for (const [key, d] of Object.entries(metricDefinitions))
    document.querySelector("#metric").append(new Option(d.label, key));
  document.querySelector("#data-status").textContent =
    "Loading current modeled air quality… " +
    (officialLoaded
      ? "Official ACS five-year snapshot loaded."
      : "Official data not yet loaded.");
  render();
  mapController = initMap(cities, selectCity);
  airProvider = createAirProvider(cities, samples);
  await refreshAir();
} catch {
  document.querySelector("#data-status").textContent =
    "City data could not load. Serve this static site over HTTP and reload. Data files remain available in the source repository.";
  document
    .querySelectorAll(".controls select,.controls button")
    .forEach((c) => (c.disabled = true));
}
