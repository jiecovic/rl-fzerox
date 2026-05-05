const RUN_CHART_SERIES_PALETTE = [
  "var(--accent)",
  "var(--run-accent)",
  "#b7791f",
  "#7c3aed",
  "#c2410c",
  "#0f766e",
] as const;

export function chartSeriesColor(index: number) {
  return RUN_CHART_SERIES_PALETTE[index % RUN_CHART_SERIES_PALETTE.length];
}

export function latestPointValue(points: { value: number }[]) {
  return points.at(-1)?.value ?? null;
}

export function formatChartValue(value: number | null, title: string) {
  if (value === null) {
    return "n/a";
  }
  if (title.includes("rate") || title.includes("fraction")) {
    return value.toFixed(3);
  }
  if (title === "Sim / wall") {
    return `${value.toFixed(2)}x`;
  }
  if (Math.abs(value) >= 1_000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}
