// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/RunChartLegend.tsx
import type { ManagedRun } from "@/shared/api/contract";

import { chartSeriesColor, type LineageRunGroup } from "./model";

interface RunChartLegendProps {
  colorByRunId: ReadonlyMap<string, string>;
  groups: readonly LineageRunGroup[];
  onOpenRun?: (run: ManagedRun) => void;
}

export function RunChartLegend({ colorByRunId, groups, onOpenRun }: RunChartLegendProps) {
  return (
    <section className="run-chart-global-legend" aria-label="Selected run colors">
      {groups.map((group) => (
        <div
          className="run-chart-lineage-group run-chart-legend-lineage-group"
          key={group.lineageId}
        >
          <div className="run-chart-lineage-header">
            <strong>{group.label}</strong>
            <span>{group.runs.length} selected</span>
          </div>
          <ul className="run-chart-lineage-legend-list">
            {group.runs.map((run, index) => (
              <li className="run-chart-global-legend-row" key={run.id}>
                <button
                  className="run-chart-global-legend-button"
                  type="button"
                  onClick={() => onOpenRun?.(run)}
                >
                  <span
                    aria-hidden="true"
                    className="run-chart-legend-swatch"
                    style={{
                      background: colorByRunId.get(run.id) ?? chartSeriesColor(index),
                    }}
                  />
                  <span className="run-chart-global-legend-name">{run.name}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}
