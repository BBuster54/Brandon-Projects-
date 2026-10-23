import { node, format } from "./shared.js";
import { categories, markerCategory } from "../aqi.js";
import { comparable, metricDefinitions } from "../data/normalize.js";
import { applyEPA } from "./epa-style.js";
import { placeMapLabel } from "./map-label-layout.js";
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
  const bounds = L.latLngBounds(cities.map((c) => [c.lat, c.lon]));
  const fit = () =>
    map.fitBounds(bounds.pad(root.clientWidth < 600 ? 0 : 0.14), {
      animate: false,
      padding: [24, 24],
    });
  map.on("resize", fit);
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
  function positionLabels() {
    const anchors = cities.map((c) => ({
      id: c.id,
      ...map.latLngToContainerPoint([c.lat, c.lon]),
    }));
    const r = root.getBoundingClientRect(),
      dock = document
        .querySelector("#filter-disclosure")
        .getBoundingClientRect();
    const blocked =
      dock.bottom > r.top
        ? [
            {
              x: dock.left - r.left,
              y: dock.top - r.top,
              width: dock.width,
              height: dock.height,
            },
          ]
        : [];
    for (const [id, marker] of shown) {
      const pin = anchors.find((p) => p.id === id),
        label = marker.getElement()?.querySelector(".marker-name");
      if (!label) continue;
      label.hidden =
        pin.x < 0 ||
        pin.x > root.clientWidth ||
        pin.y < 0 ||
        pin.y > root.clientHeight;
      const box = placeMapLabel(
        pin,
        anchors.filter((p) => p.id !== id),
        root.clientWidth,
        root.clientHeight,
        {
          blocked,
          labelHeight: Math.max(
            52,
            Math.ceil(label.textContent.length / 24) * 20 + 12,
          ),
        },
      );
      label.style.setProperty("--label-left", box.x - pin.x + 22 + "px");
      label.style.setProperty("--label-top", box.y - pin.y + 22 + "px");
    }
  }
  map.on("moveend zoomend resize", positionLabels);
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
      fit();
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
          iconAnchor: [22, 22],
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
    positionLabels();
  }
  return { update, map };
}
