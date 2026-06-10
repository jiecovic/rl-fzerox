// src/rl_fzerox/apps/run_manager/web/src/features/runs/chartsPanel/model/colors.ts
import type { ManagedRun } from "@/shared/api/contract";

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

export function buildChartColorByRunId(
  allRuns: readonly ManagedRun[],
  selectedRuns: readonly ManagedRun[],
) {
  const selectedLineageIds = orderedUniqueLineageIds(selectedRuns);
  if (selectedLineageIds.length > 1) {
    return lineageColorByRunId(allRuns);
  }
  return selectedRunColorByRunId(allRuns, selectedRuns);
}

function selectedRunColorByRunId(
  allRuns: readonly ManagedRun[],
  selectedRuns: readonly ManagedRun[],
) {
  const colorByRunId = new Map<string, string>();
  for (const [index, run] of allRuns.entries()) {
    colorByRunId.set(run.id, chartSeriesColor(index));
  }
  for (const [index, run] of selectedRuns.entries()) {
    colorByRunId.set(run.id, chartSeriesColor(index));
  }
  return colorByRunId;
}

function lineageColorByRunId(runs: readonly ManagedRun[]) {
  const colorByLineageId = new Map<string, string>();
  for (const [index, lineageId] of orderedUniqueLineageIds(runs).entries()) {
    colorByLineageId.set(lineageId, chartSeriesColor(index));
  }
  return new Map(
    runs.map((run) => [run.id, colorByLineageId.get(run.lineage_id) ?? chartSeriesColor(0)]),
  );
}

function orderedUniqueLineageIds(runs: readonly ManagedRun[]) {
  return [...new Set(runs.map((run) => run.lineage_id))].sort((left, right) =>
    left.localeCompare(right),
  );
}
