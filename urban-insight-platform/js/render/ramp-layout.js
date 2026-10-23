// Presentation-only layout; source values and EPA category boundaries never change.
export function rampLayout(
  values,
  width,
  { labelWidth = 200, gap = 12, maxLanes = 3 } = {},
) {
  const lanes = [];
  const items = values.map((value, index) => ({
    value,
    index,
    x: (Math.min(500, Math.max(0, value)) / 500) * width,
    offScale: value > 500,
  }));
  let overflow = false;
  for (const item of [...items].sort(
    (a, b) => a.x - b.x || a.index - b.index,
  )) {
    item.labelX = Math.max(
      labelWidth / 2,
      Math.min(width - labelWidth / 2, item.x),
    );
    let lane = lanes.findIndex(
      (end) => item.labelX - labelWidth / 2 >= end + gap,
    );
    if (lane < 0) lane = lanes.length;
    if (lane >= maxLanes) overflow = true;
    lanes[lane] = item.labelX + labelWidth / 2;
    item.lane = lane;
  }
  return {
    mode: width < 600 || overflow ? "stacked" : "staggered",
    items,
    lanes: lanes.length,
  };
}
