/**
 * Window event broadcast by the root layout whenever the ⌘K Stock Research
 * Terminal overlay opens or closes (`detail: boolean`). Surfaces rendered
 * deeper in the tree — e.g. the dashboard's bottom market marquee ticker —
 * listen for it to stay in sync without prop drilling through the layout
 * boundary.
 */
export const TERMINAL_VISIBILITY_EVENT = "fey:terminal-visibility";