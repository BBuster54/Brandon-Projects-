const formats = new Map();
export function format(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value)))
    return "—";
  if (!formats.has(digits))
    formats.set(
      digits,
      new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }),
    );
  return formats.get(digits).format(Number(value));
}
export function node(tag, text, className) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}
export function localTime(timestamp, city) {
  if (!timestamp || !Number.isFinite(Date.parse(timestamp)))
    return "Observation date unknown";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: city.timeZone,
    timeZoneName: "short",
  }).format(new Date(timestamp));
}
