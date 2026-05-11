// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/RunChartLegend.tsx
import type { ManagedRun } from "@/shared/api/contract";

import { chartSeriesColor, type LineageRunGroup } from "./model";

interface RunChartLegendProps {
  colorMode: "lineage" | "run";
  colorByRunId: ReadonlyMap<string, string>;
  groups: readonly LineageRunGroup[];
  onOpenRun?: (run: ManagedRun) => void;
}

export function RunChartLegend({
  colorByRunId,
  colorMode,
  groups,
  onOpenRun,
}: RunChartLegendProps) {
  if (colorMode === "lineage") {
    return (
      <section className="run-chart-global-legend" aria-label="Selected run colors">
        <div className="run-chart-lineage-legend-list">
          {groups.map((group) => (
            <LineageLegendEntry colorByRunId={colorByRunId} group={group} key={group.lineageId} />
          ))}
        </div>
      </section>
    );
  }

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
                  <LegendSwatch color={colorByRunId.get(run.id) ?? chartSeriesColor(index)} />
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

function LineageLegendEntry({
  colorByRunId,
  group,
}: {
  colorByRunId: ReadonlyMap<string, string>;
  group: LineageRunGroup;
}) {
  const fallbackColor = chartSeriesColor(0);
  const color = colorByRunId.get(group.runs[0]?.id ?? "") ?? fallbackColor;
  return (
    <div className="run-chart-global-legend-row">
      <span className="run-chart-global-legend-button is-static">
        <LegendSwatch color={color} />
        <span className="run-chart-global-legend-name">{group.label}</span>
      </span>
    </div>
  );
}

function LegendSwatch({ color }: { color: string }) {
  return (
    <span aria-hidden="true" className="run-chart-legend-swatch" style={{ background: color }} />
  );
}
