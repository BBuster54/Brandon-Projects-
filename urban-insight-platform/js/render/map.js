import { node, format } from "./shared.js";
import { categories, markerCategory } from "../aqi.js";
import { comparable, metricDefinitions } from "../data/normalize.js";
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
    "Use two fingers to move the map; Ctrl/⌘ + scroll to zoom. The city directory provides the same selection without using the map.";
  const legend = document.querySelector("#map-legend");
  for (const cat of categories) {
    const item = node("span"),
      dot = node("i");
    dot.style.backgroundColor = cat.color;
    item.append(dot, node("span", cat.label));
    legend.append(item);
  }
  const none = node("span"),
    dot = node("i");
  dot.className = "neutral";
  none.append(dot, node("span", "No current reading"));
  legend.append(none);
  document
    .querySelector("#map-credit")
    .replaceChildren(
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
        circle.style.backgroundColor = category?.color ?? "#fff";
      else if (!isSample) circle.style.backgroundColor = "var(--accent)";
      content.append(circle, node("span", city.name, "marker-name"));
      const label =
        city.name +
        " · " +
        (isSample
          ? "No current reading"
          : state.metric === "aqi"
            ? (category?.label ?? "Unavailable") + (aqi.stale ? " · stale" : "")
            : metricDefinitions[state.metric].label +
              ": " +
              (comparable(m)
                ? format(m.value) + " " + m.unit
                : "Official data not yet loaded"));
      const marker = L.marker([city.lat, city.lon], {
        icon: L.divIcon({
          html: content,
          className: "city-marker-wrapper",
          iconSize: [120, 44],
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
    if (focusedCity) (shown.get(focusedCity)?.getElement() ?? document.querySelector("#city")).focus({preventScroll:true});
    if (!hasFit) {
      map.fitBounds(
        L.latLngBounds(cities.map((c) => [c.lat, c.lon])).pad(0.2),
        { animate: false },
      );
      hasFit = true;
    }
  }
  return { update, map };
}
