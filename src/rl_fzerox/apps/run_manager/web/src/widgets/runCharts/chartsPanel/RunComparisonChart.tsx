// src/rl_fzerox/apps/run_manager/web/src/widgets/runCharts/chartsPanel/RunComparisonChart.tsx
import { useMemo } from "react";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";
import { RunPlotCard, type RunPlotPoint } from "@/widgets/runCharts/charts/RunPlotCard";
import {
  chartSeriesColor,
  formatChartValue,
  latestPointValue,
} from "@/widgets/runCharts/chartsPanel/model";

interface RunComparisonChartProps {
  buildPoints: (run: ManagedRun, samples: ManagedRunMetricSample[]) => RunPlotPoint[];
  colorByRunId: ReadonlyMap<string, string>;
  emptyText: string;
  metricsByRun: Record<string, ManagedRunMetricSample[]>;
  runs: ManagedRun[];
  title: string;
}

export function RunComparisonChart({
  buildPoints,
  colorByRunId,
  emptyText,
  metricsByRun,
  runs,
  title,
}: RunComparisonChartProps) {
  const series = useMemo(
    () =>
      runs.map((run, index) => {
        const points = buildPoints(run, metricsByRun[run.id] ?? []);
        return {
          color: colorByRunId.get(run.id) ?? chartSeriesColor(index),
          latest: latestPointValue(points),
          name: run.name,
          points,
          runId: run.id,
        };
      }),
    [buildPoints, colorByRunId, metricsByRun, runs],
  );
  const formatValue = useMemo(
    () => (value: number | null) => formatChartValue(value, title),
    [title],
  );

  return (
    <RunPlotCard emptyText={emptyText} formatValue={formatValue} series={series} title={title} />
  );
}
