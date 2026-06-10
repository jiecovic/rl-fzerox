// src/rl_fzerox/apps/run_manager/web/src/widgets/runCharts/charts/run_plot_card/model.ts
import type { Dispatch, MutableRefObject, SetStateAction } from "react";

import type UPlot from "uplot";

import type {
  PlotData,
  RunPlotSeries,
  TooltipEntry,
  TooltipState,
  UPlotConstructor,
  XRange,
} from "@/widgets/runCharts/charts/run_plot_card/types";

export const PLOT_HEIGHT = 204;
const TOOLTIP_CHROME_HEIGHT = 34;
const TOOLTIP_GAP = 18;
const TOOLTIP_MARGIN = 8;
const TOOLTIP_ROW_HEIGHT = 22;
const TOOLTIP_WIDTH = 220;

export function buildPlotOptions({
  formatValueRef,
  fullXRangeRef,
  onTooltipChange,
  onZoomChange,
  styles,
  visibleSeriesRef,
  widthRef,
}: {
  formatValueRef: MutableRefObject<(value: number | null) => string>;
  fullXRangeRef: MutableRefObject<XRange | null>;
  onTooltipChange: Dispatch<SetStateAction<TooltipState | null>>;
  onZoomChange: (range: XRange | null) => void;
  styles: CSSStyleDeclaration;
  visibleSeriesRef: MutableRefObject<readonly RunPlotSeries[]>;
  widthRef: MutableRefObject<number>;
}): UPlot.Options {
  return {
    axes: [
      {
        grid: { show: false },
        size: 28,
        stroke: cssColor(styles, "--muted", "#6b7280"),
        values: (_self, splits) => splits.map((value) => formatStepValue(value)),
      },
      {
        grid: {
          show: true,
          stroke: cssColor(styles, "--border", "#2f3a4f"),
          width: 1,
        },
        size: 56,
        stroke: cssColor(styles, "--muted", "#6b7280"),
        values: (_self, splits) =>
          splits.map((value) => formatValueRef.current(typeof value === "number" ? value : null)),
      },
    ],
    class: "run-uplot",
    cursor: {
      drag: { x: true, y: false },
      focus: { prox: 20 },
      points: {
        fill: cssColor(styles, "--surface", "#0f172a"),
        size: 6,
        stroke: cssColor(styles, "--text", "#e5e7eb"),
        width: 1.5,
      },
      x: true,
      y: true,
    },
    height: PLOT_HEIGHT,
    hooks: {
      setCursor: [
        (plot) => {
          onTooltipChange(currentTooltipState(plot, visibleSeriesRef.current, widthRef.current));
        },
      ],
      setScale: [
        (plot, scaleKey) => {
          if (scaleKey !== "x") {
            return;
          }
          const min = plot.scales.x.min;
          const max = plot.scales.x.max;
          const fullXRange = fullXRangeRef.current;
          if (typeof min !== "number" || typeof max !== "number" || fullXRange === null) {
            return;
          }
          if (sameXRange({ min, max }, fullXRange)) {
            onZoomChange(null);
            return;
          }
          onZoomChange({ min, max });
        },
      ],
    },
    legend: { show: false },
    padding: [10, 12, 8, 10],
    scales: {
      x: { time: false },
      y: { auto: true },
    },
    series: [
      {},
      ...visibleSeriesRef.current.map((entry) => ({
        label: entry.name,
        points: { show: false },
        spanGaps: true,
        stroke: resolveCanvasColor(entry.color, styles),
        width: 2,
      })),
    ],
    width: widthRef.current,
  };
}

export function currentTooltipState(
  plot: UPlot,
  series: readonly RunPlotSeries[],
  width: number,
): TooltipState | null {
  const idx = plot.cursor.idx;
  const left = plot.cursor.left;
  const top = plot.cursor.top;
  if (typeof idx !== "number" || left === undefined || top === undefined) {
    return null;
  }

  const xValues = plot.data[0];
  const step = xValues[idx];
  if (typeof step !== "number") {
    return null;
  }

  const values: TooltipEntry[] = [];
  for (const [seriesIndex, entry] of series.entries()) {
    const seriesValues = plot.data[seriesIndex + 1];
    const rawValue = seriesValues?.[idx];
    if (typeof rawValue === "number") {
      values.push({ color: entry.color, name: entry.name, value: rawValue });
    }
  }
  if (values.length === 0) {
    return null;
  }

  const tooltipHeight = TOOLTIP_CHROME_HEIGHT + values.length * TOOLTIP_ROW_HEIGHT;
  const showLeftOfCursor = left > width - TOOLTIP_WIDTH - TOOLTIP_GAP - TOOLTIP_MARGIN;
  const showAboveCursor = top > plot.height - tooltipHeight - TOOLTIP_GAP - TOOLTIP_MARGIN;
  const rawLeft = showLeftOfCursor ? left - TOOLTIP_WIDTH - TOOLTIP_GAP : left + TOOLTIP_GAP;
  const rawTop = showAboveCursor ? top - tooltipHeight - TOOLTIP_GAP : top + TOOLTIP_GAP;

  return {
    left: clamp(rawLeft, TOOLTIP_MARGIN, Math.max(TOOLTIP_MARGIN, width - TOOLTIP_WIDTH)),
    step,
    top: clamp(rawTop, TOOLTIP_MARGIN, Math.max(TOOLTIP_MARGIN, plot.height - tooltipHeight)),
    values,
  };
}

export async function ensureUPlot(): Promise<UPlotConstructor> {
  const module: unknown = await import("uplot");
  if (typeof module === "function") {
    return module as UPlotConstructor;
  }
  if (
    typeof module === "object" &&
    module !== null &&
    "default" in module &&
    typeof module.default === "function"
  ) {
    return module.default as UPlotConstructor;
  }
  throw new Error("failed to load uplot constructor");
}

export function alignedDataFromSeries(series: readonly RunPlotSeries[]): PlotData | null {
  if (series.length === 0) {
    return null;
  }

  const stepSet = new Set<number>();
  for (const entry of series) {
    for (const point of entry.points) {
      stepSet.add(point.step);
    }
  }

  const xValues = [...stepSet].sort((left, right) => left - right);
  if (xValues.length === 0) {
    return null;
  }

  const xIndexByStep = new Map<number, number>();
  xValues.forEach((step, index) => {
    xIndexByStep.set(step, index);
  });

  const ySeries = series.map<(number | null)[]>((entry) => {
    const values = Array<number | null>(xValues.length).fill(null);
    for (const point of entry.points) {
      const pointIndex = xIndexByStep.get(point.step);
      if (pointIndex !== undefined) {
        values[pointIndex] = point.value;
      }
    }
    return values;
  });

  return [xValues, ...ySeries];
}

export function plotSeriesKey(series: readonly RunPlotSeries[]) {
  return series
    .map((entry) => [entry.runId, entry.name, entry.color].join("\u001f"))
    .join("\u001e");
}

function cssColor(styles: CSSStyleDeclaration, variable: string, fallback: string) {
  const value = styles.getPropertyValue(variable).trim();
  return value === "" ? fallback : value;
}

function resolveCanvasColor(color: string, styles: CSSStyleDeclaration) {
  if (!color.startsWith("var(")) {
    return color;
  }
  const token = color.slice(4, -1).trim();
  const value = styles.getPropertyValue(token).trim();
  return value === "" ? color : value;
}

function formatStepValue(value: number) {
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(value % 1_000_000 === 0 ? 0 : 1)}M`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(value % 1_000 === 0 ? 0 : 1)}k`;
  }
  return value.toFixed(0);
}

export function xRange(xValues: readonly number[]): XRange {
  return { max: xValues[xValues.length - 1] ?? 0, min: xValues[0] ?? 0 };
}

export function clampXRange(range: XRange | null, fullRange: XRange | null) {
  if (range === null || fullRange === null) {
    return null;
  }
  const min = Math.max(range.min, fullRange.min);
  const max = Math.min(range.max, fullRange.max);
  return max > min ? { max, min } : null;
}

function sameXRange(left: XRange, right: XRange) {
  return Math.abs(left.min - right.min) < 1e-9 && Math.abs(left.max - right.max) < 1e-9;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}
