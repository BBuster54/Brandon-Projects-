// Screen-space annotation placement only. Geographic coordinates are never shifted.
export function placeMapLabel(
  pin,
  others,
  width,
  height,
  { labelWidth = 185, labelHeight = 90, blocked = [] } = {},
) {
  const gap = 32,
    pad = 8,
    clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const candidates = [
    [pin.x - labelWidth, pin.y + gap],
    [pin.x - labelWidth, pin.y - gap - labelHeight],
    [pin.x + gap, pin.y + gap],
    [pin.x + gap, pin.y - gap - labelHeight],
    [pin.x - labelWidth / 2, pin.y + gap],
    [pin.x - labelWidth / 2, pin.y - gap - labelHeight],
  ];
  const obstacles = [
    ...others.map((p) => ({ x: p.x - 26, y: p.y - 26, width: 52, height: 52 })),
    ...blocked,
  ];
  const area = (a, b) =>
    Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) *
    Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return candidates
    .map(([x, y]) => ({
      x: clamp(x, pad, width - labelWidth - pad),
      y: clamp(y, pad, height - labelHeight - pad),
      width: labelWidth,
      height: labelHeight,
    }))
    .map((r) => ({
      ...r,
      overlap: obstacles.reduce((sum, o) => sum + area(r, o), 0),
    }))
    .sort((a, b) => a.overlap - b.overlap)[0];
}
