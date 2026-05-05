import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import type UPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { ResetIcon } from "@/shared/ui/icons";

export type RunPlotPoint = { step: number; value: number };

export type RunPlotSeries = {
  color: string;
  latest: number | null;
  name: string;
  points: readonly RunPlotPoint[];
  runId: string;
};

interface RunPlotCardProps {
  emptyText: string;
  formatValue: (value: number | null) => string;
  series: readonly RunPlotSeries[];
  title: string;
}

type PlotData = [number[], ...(number | null)[][]];
type TooltipEntry = { color: string; name: string; value: number };
type TooltipState = { left: number; step: number; top: number; values: readonly TooltipEntry[] };
type XRange = { max: number; min: number };
type UPlotConstructor = new (
  opts: UPlot.Options,
  data?: UPlot.AlignedData,
  targ?: HTMLElement | ((self: UPlot, init: () => void) => void),
) => UPlot;

const PLOT_HEIGHT = 204;
const TOOLTIP_CHROME_HEIGHT = 34;
const TOOLTIP_GAP = 18;
const TOOLTIP_MARGIN = 8;
const TOOLTIP_ROW_HEIGHT = 22;
const TOOLTIP_WIDTH = 220;

export function RunPlotCard({ emptyText, formatValue, series, title }: RunPlotCardProps) {
  const plotRef = useRef<HTMLDivElement | null>(null);
  const plotInstanceRef = useRef<UPlot | null>(null);
  const plotConstructorRef = useRef<UPlotConstructor | null>(null);
  const plotSeriesKeyRef = useRef("");
  const formatValueRef = useRef(formatValue);
  const visibleSeriesRef = useRef<readonly RunPlotSeries[]>([]);
  const fullXRangeRef = useRef<XRange | null>(null);
  const widthRef = useRef(0);
  const xZoomRef = useRef<XRange | null>(null);

  const [shellNode, setShellNode] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const [hasZoom, setHasZoom] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const visibleSeries = useMemo(() => series.filter((entry) => entry.points.length > 0), [series]);
  const seriesKey = useMemo(
    () => visibleSeries.map((entry) => entry.runId).join("|"),
    [visibleSeries],
  );
  const plotData = useMemo(() => alignedDataFromSeries(visibleSeries), [visibleSeries]);

  formatValueRef.current = formatValue;
  visibleSeriesRef.current = visibleSeries;
  fullXRangeRef.current = plotData === null ? null : xRange(plotData[0]);

  useEffect(() => {
    widthRef.current = width;
  }, [width]);

  useEffect(() => {
    if (shellNode === null) {
      return undefined;
    }
    setWidth(Math.floor(shellNode.getBoundingClientRect().width));
    if (typeof ResizeObserver === "undefined") {
      return undefined;
    }
    const resizeObserver = new ResizeObserver((entries) => {
      const nextWidth = Math.floor(entries[0]?.contentRect.width ?? 0);
      setWidth((current) => (current === nextWidth ? current : nextWidth));
    });
    resizeObserver.observe(shellNode);
    return () => {
      resizeObserver.disconnect();
    };
  }, [shellNode]);

  useEffect(() => {
    return () => {
      plotInstanceRef.current?.destroy();
      plotInstanceRef.current = null;
    };
  }, []);

  useEffect(() => {
    const target = plotRef.current;
    if (target === null || width <= 0) {
      return undefined;
    }
    if (plotData === null) {
      setTooltip(null);
      plotInstanceRef.current?.destroy();
      plotInstanceRef.current = null;
      plotSeriesKeyRef.current = "";
      return undefined;
    }

    const styles = getComputedStyle(target);
    let cancelled = false;

    void ensureUPlot().then((PlotRuntime) => {
      if (cancelled) {
        return;
      }
      plotConstructorRef.current = PlotRuntime;

      const existingPlot = plotInstanceRef.current;
      const needsRebuild = existingPlot === null || plotSeriesKeyRef.current !== seriesKey;
      if (needsRebuild) {
        existingPlot?.destroy();
        const plot = new PlotRuntime(
          buildPlotOptions({
            formatValueRef,
            fullXRangeRef,
            onTooltipChange: setTooltip,
            onZoomChange: (range) => {
              xZoomRef.current = range;
              setHasZoom(range !== null);
            },
            visibleSeriesRef,
            widthRef,
            styles,
          }),
          plotData,
          target,
        );
        plotInstanceRef.current = plot;
        plotSeriesKeyRef.current = seriesKey;
        const initialZoom = clampXRange(xZoomRef.current, fullXRangeRef.current);
        if (initialZoom !== null) {
          plot.setScale("x", initialZoom);
        } else {
          setHasZoom(false);
        }
        return;
      }

      const plot = existingPlot;
      plot.setSize({ height: PLOT_HEIGHT, width });
      plot.setData(plotData, xZoomRef.current === null);

      const clampedZoom = clampXRange(xZoomRef.current, fullXRangeRef.current);
      if (clampedZoom !== null) {
        plot.setScale("x", clampedZoom);
      } else {
        xZoomRef.current = null;
        setHasZoom(false);
      }
      setTooltip(currentTooltipState(plot, visibleSeriesRef.current, width));
    });

    return () => {
      cancelled = true;
    };
  }, [plotData, seriesKey, width]);

  return (
    <div className="run-chart-card">
      <div className="run-chart-card-header">
        <strong>{title}</strong>
        <div className="run-chart-card-actions">
          {hasZoom ? (
            <button
              aria-label={`Reset ${title} zoom`}
              className="icon-button compact-icon-button tooltip-anchor"
              data-tooltip="Reset zoom"
              type="button"
              onClick={resetZoom}
            >
              <ResetIcon />
            </button>
          ) : null}
          <span>{visibleSeries.length === 0 ? "n/a" : `${visibleSeries.length} runs`}</span>
        </div>
      </div>
      {plotData === null ? (
        <div className="run-chart-empty">{emptyText}</div>
      ) : (
        <div aria-label={title} className="run-chart-canvas-shell" ref={setShellNode} role="img">
          <div className="run-chart-canvas" ref={plotRef} />
          {tooltip !== null ? (
            <div
              className="run-chart-tooltip"
              style={{ left: `${tooltip.left}px`, top: `${tooltip.top}px` }}
            >
              <div className="run-chart-tooltip-step">{tooltip.step.toLocaleString()} steps</div>
              <div className="run-chart-tooltip-values">
                {tooltip.values.map((entry) => (
                  <div className="run-chart-tooltip-row" key={entry.name}>
                    <span
                      aria-hidden="true"
                      className="run-chart-legend-swatch"
                      style={{ background: entry.color }}
                    />
                    <span className="run-chart-tooltip-name">{entry.name}</span>
                    <span className="run-chart-tooltip-value">{formatValue(entry.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );

  function resetZoom() {
    const plot = plotInstanceRef.current;
    const fullXRange = fullXRangeRef.current;
    if (plot === null || fullXRange === null) {
      return;
    }
    xZoomRef.current = null;
    setHasZoom(false);
    plot.setScale("x", fullXRange);
  }
}

function buildPlotOptions({
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

function currentTooltipState(
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

async function ensureUPlot(): Promise<UPlotConstructor> {
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

function alignedDataFromSeries(series: readonly RunPlotSeries[]): PlotData | null {
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

function xRange(xValues: readonly number[]): XRange {
  return { max: xValues[xValues.length - 1] ?? 0, min: xValues[0] ?? 0 };
}

function clampXRange(range: XRange | null, fullRange: XRange | null) {
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
