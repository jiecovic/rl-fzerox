// web/run-manager/src/entities/runChart/ui/runPlotCard/usePlot.ts
import { useEffect, useMemo, useRef, useState } from "react";

import type UPlot from "uplot";

import {
  alignedDataFromSeries,
  buildPlotOptions,
  clampXRange,
  currentTooltipState,
  ensureUPlot,
  PLOT_HEIGHT,
  plotSeriesKey,
  xRange,
} from "@/entities/runChart/ui/runPlotCard/model";
import type {
  RunPlotSeries,
  TooltipState,
  UPlotConstructor,
  XRange,
} from "@/entities/runChart/ui/runPlotCard/types";

export function useRunPlot({
  formatValue,
  series,
}: {
  formatValue: (value: number | null) => string;
  series: readonly RunPlotSeries[];
}) {
  const plotRef = useRef<HTMLDivElement | null>(null);
  const plotInstanceRef = useRef<UPlot | null>(null);
  const plotSeriesKeyRef = useRef("");
  const formatValueRef = useRef(formatValue);
  const visibleSeriesRef = useRef<readonly RunPlotSeries[]>([]);
  const fullXRangeRef = useRef<XRange | null>(null);
  const widthRef = useRef(0);
  const xZoomRef = useRef<XRange | null>(null);
  const plotConstructorRef = useRef<UPlotConstructor | null>(null);

  const [shellNode, setShellNode] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const [hasZoom, setHasZoom] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  const visibleSeries = useMemo(() => series.filter((entry) => entry.points.length > 0), [series]);
  const seriesKey = useMemo(() => plotSeriesKey(visibleSeries), [visibleSeries]);
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
            styles,
            visibleSeriesRef,
            widthRef,
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

  return {
    hasZoom,
    plotData,
    plotRef,
    resetZoom,
    setShellNode,
    tooltip,
    visibleSeries,
  };
}
