// web/run-manager/src/widgets/runCharts/chartsPanel/RunChartLegend.tsx

import { chartSeriesColor, type LineageRunGroup } from "@/entities/runChart/model";
import type { ManagedRun } from "@/shared/api/contract";

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
      <section className={globalLegendClass} aria-label="Selected run colors">
        <div className="flex flex-wrap justify-end gap-2 max-[1100px]:justify-start">
          {groups.map((group) => (
            <LineageLegendEntry colorByRunId={colorByRunId} group={group} key={group.lineageId} />
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className={globalLegendClass} aria-label="Selected run colors">
      {groups.map((group) => (
        <div className="grid gap-2 py-1" key={group.lineageId}>
          <div className="flex items-center justify-between gap-3 border-b border-[color-mix(in_srgb,var(--border)_88%,transparent)] pb-1.5">
            <strong>{group.label}</strong>
            <span className="text-xs text-app-muted">{group.runs.length} selected</span>
          </div>
          <ul className="flex flex-wrap justify-end gap-2 max-[1100px]:justify-start">
            {group.runs.map((run, index) => (
              <li className="inline-flex" key={run.id}>
                <button
                  className={`${legendPillClass} run-chart-global-legend-button`}
                  type="button"
                  onClick={() => onOpenRun?.(run)}
                >
                  <LegendSwatch color={colorByRunId.get(run.id) ?? chartSeriesColor(index)} />
                  <span className={`${legendNameClass} run-chart-global-legend-name`}>
                    {run.name}
                  </span>
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
    <div className="inline-flex">
      <span className={`${legendPillClass} run-chart-global-legend-button`}>
        <LegendSwatch color={color} />
        <span className={`${legendNameClass} run-chart-global-legend-name`}>{group.label}</span>
      </span>
    </div>
  );
}

function LegendSwatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      className="run-chart-legend-swatch mt-1 h-2.5 w-2.5"
      style={{ background: color }}
    />
  );
}

const globalLegendClass =
  "sticky top-2.5 z-[3] ml-auto grid w-[min(100%,820px)] max-w-full gap-2 border border-[color-mix(in_srgb,var(--border-strong)_72%,transparent)] bg-[color-mix(in_srgb,var(--surface)_94%,transparent)] p-2 shadow-[0_8px_20px_rgba(15,23,42,0.1)] backdrop-blur-[10px] max-[1100px]:static max-[1100px]:mb-3 max-[1100px]:w-full";

const legendPillClass =
  "inline-grid max-w-full grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border border-[color-mix(in_srgb,var(--border-strong)_50%,transparent)] bg-[color-mix(in_srgb,var(--surface)_84%,transparent)] px-2.5 py-1.5 text-left text-inherit hover:text-app-text focus-visible:text-app-text focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-app-accent";

const legendNameClass =
  "min-w-0 max-w-[220px] overflow-hidden text-ellipsis whitespace-nowrap text-xs";
