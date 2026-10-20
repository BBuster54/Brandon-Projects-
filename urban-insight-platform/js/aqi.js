export const categories = [
  { id: "good", label: "Good", max: 50, color: "#00e400" },
  { id: "moderate", label: "Moderate", max: 100, color: "#ffff00" },
  {
    id: "sensitive",
    label: "Unhealthy for sensitive groups",
    max: 150,
    color: "#ff7e00",
  },
  { id: "unhealthy", label: "Unhealthy", max: 200, color: "#ff0000" },
  { id: "very-unhealthy", label: "Very unhealthy", max: 300, color: "#8f3f97" },
  { id: "hazardous", label: "Hazardous", max: Infinity, color: "#7e0023" },
];
export function aqiCategory(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0)
    return null;
  return categories.find((c) => value <= c.max);
}
export function markerCategory(metric) {
  return metric?.kind === "live" && metric.value !== null
    ? aqiCategory(metric.value)
    : null;
}
