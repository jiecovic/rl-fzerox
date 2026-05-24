// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts/RunPlotCard.tsx
import "uplot/dist/uPlot.min.css";

import type {
  RunPlotCardProps,
  RunPlotPoint,
  RunPlotSeries,
} from "@/features/runs/charts/run_plot_card/types";

import { useRunPlot } from "@/features/runs/charts/run_plot_card/usePlot";
import { ResetIcon } from "@/shared/ui/icons";

export type { RunPlotPoint, RunPlotSeries };

export function RunPlotCard({ emptyText, formatValue, series, title }: RunPlotCardProps) {
  const { hasZoom, plotData, plotRef, resetZoom, setShellNode, tooltip, visibleSeries } =
    useRunPlot({
      formatValue,
      series,
    });

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
}
