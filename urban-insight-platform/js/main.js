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
let cities = [],
  records = {},
  samples = {},
  state = { ...defaultState };
let mapController = null;
// phase-two-imports
// phase-three-imports
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
  // phase-three-render
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
  const [meta, acsResponse, sampleResponse] = await Promise.all([
    loadCities(),
    fetch("data/acs.json"),
    fetch("data/sample-metrics.json"),
  ]);
  if (!acsResponse.ok || !sampleResponse.ok)
    throw new Error("Data unavailable");
  cities = meta;
  const [acs, sampleData] = await Promise.all([
    acsResponse.json(),
    sampleResponse.json(),
  ]);
  samples = Object.fromEntries(sampleData.cities.map((c) => [c.id, c.metrics]));
  state = parseState(location.search, cities);
  for (const c of cities) {
    document
      .querySelector("#city")
      .append(new Option(c.name + ", " + c.state, c.id));
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
    "No current modeled air-quality data. Official data not yet loaded. Illustrative values are confined to the sample disclosure.";
  render();
  // phase-two-init
  // phase-three-init
} catch {
  document.querySelector("#data-status").textContent =
    "City data could not load. Serve this static site over HTTP and reload. Data files remain available in the source repository.";
  document
    .querySelectorAll(".controls select,.controls button")
    .forEach((c) => (c.disabled = true));
}
