// web/run-manager/src/widgets/runCharts/chartsPanel/RunComparisonChart.tsx
import { useMemo } from "react";
import { chartSeriesColor, formatChartValue, latestPointValue } from "@/entities/runChart/model";
import { RunPlotCard, type RunPlotPoint } from "@/entities/runChart/ui/RunPlotCard";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";

export interface RunComparisonSeriesGroup {
  color: string;
  id: string;
  label: string;
  runIds: readonly string[];
}

interface RunComparisonChartProps {
  buildPoints: (run: ManagedRun, samples: ManagedRunMetricSample[]) => RunPlotPoint[];
  emptyText: string;
  metricsByRun: Record<string, ManagedRunMetricSample[]>;
  runs: ManagedRun[];
  seriesGroups: readonly RunComparisonSeriesGroup[];
  seriesUnit: string;
  title: string;
}

export function RunComparisonChart({
  buildPoints,
  emptyText,
  metricsByRun,
  runs,
  seriesGroups,
  seriesUnit,
  title,
}: RunComparisonChartProps) {
  const runById = useMemo(() => new Map(runs.map((run) => [run.id, run])), [runs]);
  const series = useMemo(
    () =>
      seriesGroups.map((group, index) => {
        const points = group.runIds
          .flatMap((runId) => {
            const run = runById.get(runId);
            return run === undefined ? [] : buildPoints(run, metricsByRun[run.id] ?? []);
          })
          .sort((left, right) => left.step - right.step);
        return {
          color: group.color || chartSeriesColor(index),
          latest: latestPointValue(points),
          name: group.label,
          points,
          runId: group.id,
        };
      }),
    [buildPoints, metricsByRun, runById, seriesGroups],
  );
  const formatValue = useMemo(
    () => (value: number | null) => formatChartValue(value, title),
    [title],
  );

  return (
    <RunPlotCard
      emptyText={emptyText}
      formatValue={formatValue}
      series={series}
      seriesUnit={seriesUnit}
      title={title}
    />
  );
}
