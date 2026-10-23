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
  light: { min: "#606060", max: "#d0d0d0" },
  dark: { min: "#262626", max: "#4d4d4d" },
};
