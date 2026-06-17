// web/run-manager/src/entities/runChart/ui/RunPlotCard.tsx
import "uplot/dist/uPlot.min.css";

import type {
  RunPlotCardProps,
  RunPlotPoint,
  RunPlotSeries,
} from "@/entities/runChart/ui/runPlotCard/types";
import { useRunPlot } from "@/entities/runChart/ui/runPlotCard/usePlot";
import { ResetIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export type { RunPlotPoint, RunPlotSeries };

export function RunPlotCard({
  emptyText,
  formatValue,
  series,
  seriesUnit = "runs",
  title,
}: RunPlotCardProps) {
  const { hasZoom, plotData, plotRef, resetZoom, setShellNode, tooltip, visibleSeries } =
    useRunPlot({
      formatValue,
      series,
    });

  return (
    <div className="grid gap-3 border border-app-border bg-app-surface-muted p-3.5">
      <div className="flex items-baseline justify-between gap-3">
        <strong>{title}</strong>
        <div className="inline-flex items-center gap-2">
          {hasZoom ? (
            <TooltipIconButton
              aria-label={`Reset ${title} zoom`}
              size="compact"
              tooltip="Reset zoom"
              onClick={resetZoom}
            >
              <ResetIcon />
            </TooltipIconButton>
          ) : null}
          <span className="text-xs text-app-muted">
            {visibleSeries.length === 0
              ? "n/a"
              : seriesCountLabel(visibleLogicalSeriesCount(visibleSeries), seriesUnit)}
          </span>
        </div>
      </div>
      {plotData === null ? (
        <div className="border border-dashed border-app-border bg-app-surface px-3.5 py-5.5 text-xs text-app-muted">
          {emptyText}
        </div>
      ) : (
        <div
          aria-label={title}
          className="relative min-h-[204px] pb-1.5"
          ref={setShellNode}
          role="img"
        >
          <div className="min-h-[204px] w-full" ref={plotRef} />
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
}

function seriesCountLabel(count: number, unit: string) {
  return `${count} ${count === 1 ? singularSeriesUnit(unit) : unit}`;
}

function visibleLogicalSeriesCount(series: readonly RunPlotSeries[]) {
  return new Set(series.map((entry) => entry.groupId ?? entry.runId)).size;
}

function singularSeriesUnit(unit: string) {
  if (unit === "branches") {
    return "branch";
  }
  if (unit === "lineages") {
    return "lineage";
  }
  if (unit === "runs") {
    return "run";
  }
  return unit.replace(/s$/, "");
}
