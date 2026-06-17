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
  const groupByRunId = useMemo(() => {
    const next = new Map<string, RunComparisonSeriesGroup>();
    for (const group of seriesGroups) {
      for (const runId of group.runIds) {
        next.set(runId, group);
      }
    }
    return next;
  }, [seriesGroups]);
  const series = useMemo(
    () =>
      runs.map((run, index) => {
        const group = groupByRunId.get(run.id);
        const points = buildPoints(run, metricsByRun[run.id] ?? []);
        return {
          color: group?.color || chartSeriesColor(index),
          ...(group === undefined ? {} : { groupId: group.id }),
          latest: latestPointValue(points),
          name: group?.label ?? run.name,
          points,
          runId: run.id,
        };
      }),
    [buildPoints, groupByRunId, metricsByRun, runs],
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
