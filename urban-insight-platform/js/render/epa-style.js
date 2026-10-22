import { contrast } from "../contrast.js";
export function fillEdge(fill) {
  return contrast(fill, "#000000") >= contrast(fill, "#ffffff")
    ? "#000000"
    : "#ffffff";
}
export function applyEPA(element, fill) {
  element.style.setProperty("--epa-fill", fill);
  element.style.setProperty("--epa-edge", fillEdge(fill));
}
export const tileRange = {
  light: { min: "#636363", max: "#cecece" },
  dark: { min: "#282828", max: "#4b4b4b" },
};
