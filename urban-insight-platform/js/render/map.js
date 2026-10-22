import { node, format } from "./shared.js";
import { categories, markerCategory } from "../aqi.js";
import { comparable, metricDefinitions } from "../data/normalize.js";
import { applyEPA } from "./epa-style.js";
export function initMap(cities, onSelect) {
  const L = globalThis.L;
  if (!L) return { update() {} };
  const root = document.querySelector("#map");
  root.replaceChildren();
  const map = L.map(root, {
    gestureHandling: true,
    gestureHandlingOptions: {
      duration: 1400,
      text: {
        touch: "Use two fingers to move the map",
        scroll: "Hold Ctrl or ⌘ and scroll to zoom",
        scrollMac: "Hold ⌘ and scroll to zoom",
      },
    },
    zoomControl: true,
    scrollWheelZoom: false,
    touchZoom: false,
  });
  map.attributionControl.setPrefix(false);
  const bounds = L.latLngBounds(cities.map((c) => [c.lat, c.lon])).pad(0.2);
  map.on("resize", () => map.fitBounds(bounds, { animate: false }));
  const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution:
      '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
    referrerPolicy: "strict-origin-when-cross-origin",
    updateWhenIdle: true,
    keepBuffer: 0,
  });
  tiles.addTo(map);
  let shown = new Map(),
    hasFit = false,
    tileWarning = false;
  tiles.on("tileerror", () => {
    if (!tileWarning) {
      tileWarning = true;
      document.querySelector("#map-credit").textContent =
        "Map tiles are unavailable. City markers and the accessible directory still work. © OpenStreetMap contributors.";
    }
  });
  tiles.on("tileload", () => {
    if (tileWarning) {
      tileWarning = false;
      document.querySelector("#map-credit").textContent =
        "© OpenStreetMap contributors · City centers are geographic anchors, not monitor locations.";
    }
  });
  document.querySelector("#map-note").textContent =
    "Two fingers to move; Ctrl/⌘ + scroll to zoom. Or select a city in the directory.";
  const legend = document.querySelector("#map-legend");
  for (const cat of categories) {
    const item = node("span"),
      dot = node("i");
    applyEPA(dot, cat.color);
    item.append(dot, node("span", cat.label));
    legend.append(item);
  }
  const none = node("span"),
    dot = node("i");
  dot.className = "neutral";
  none.append(dot, node("span", "No current reading"));
  legend.append(none);
  document.querySelector("#map-credit").replaceChildren(
    node("span", "© "),
    Object.assign(node("a", "OpenStreetMap contributors"), {
      href: "https://www.openstreetmap.org/copyright",
    }),
    node(
      "span",
      " · City centers are geographic anchors, not monitor locations.",
    ),
  );
  function update(visible, records, state) {
    if (!hasFit) {
      map.fitBounds(
        L.latLngBounds(cities.map((c) => [c.lat, c.lon])).pad(0.2),
        { animate: false },
      );
      hasFit = true;
    }
    const focusedCity = document.activeElement?.dataset.cityId;
    for (const marker of shown.values()) map.removeLayer(marker);
    shown = new Map();
    for (const city of visible) {
      const aqi = records[city.id]?.aqi,
        category = markerCategory(aqi),
        selected = state.city === city.id,
        m = records[city.id]?.[state.metric],
        isSample = aqi?.kind !== "live" || aqi.value === null;
      const content = node(
        "div",
        undefined,
        "map-marker" +
          (selected ? " selected" : "") +
          (isSample ? " sample-marker" : ""),
      );
      const circle = node("span", undefined, "marker-dot");
      if (!isSample && state.metric === "aqi")
        applyEPA(circle, category?.color ?? "#fff7e9");
      else if (!isSample) circle.style.backgroundColor = "var(--accent)";
      content.append(circle, node("span", city.name, "marker-name"));
      const label =
        city.name +
        " · " +
        (state.metric === "aqi"
          ? isSample
            ? "No current reading"
            : (category?.label ?? "Unavailable") + (aqi.stale ? " · stale" : "")
          : metricDefinitions[state.metric].label +
            ": " +
            (comparable(m)
              ? format(m.value) + " " + m.unit
              : "Official data not yet loaded"));
      content.querySelector(".marker-name").textContent = label;
      const marker = L.marker([city.lat, city.lon], {
        icon: L.divIcon({
          html: content,
          className: "city-marker-wrapper",
          iconSize: [44, 44],
          iconAnchor: [16, 22],
        }),
        keyboard: true,
        title: label,
        alt: label,
        riseOnHover: true,
      });
      marker.on("click", () => onSelect(city.id));
      marker.addTo(map);
      const element = marker.getElement();
      element.setAttribute("role", "button");
      element.dataset.cityId = city.id;
      element.setAttribute("aria-label", label);
      element.setAttribute("aria-pressed", String(selected));
      element.addEventListener("keydown", (event) => {
        if (event.key === " ") {
          event.preventDefault();
          onSelect(city.id);
        }
      });
      shown.set(city.id, marker);
    }
    if (focusedCity)
      (
        shown.get(focusedCity)?.getElement() ?? document.querySelector("#city")
      ).focus({ preventScroll: true });
  }
  return { update, map };
}
