// src/rl_fzerox/apps/run_manager/web/src/widgets/runCharts/chartsPanel/model/types.ts

import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";
import type { RunPlotPoint } from "@/widgets/runCharts/charts/RunPlotCard";
import type { RunChartGroupId } from "@/widgets/runCharts/chartsPanel/model/catalog";

export type RunChartDescriptor = {
  id: string;
  emptyText: string;
  group: RunChartGroupId;
  metricKeys?: readonly string[];
  title: string;
  buildPoints: (run: ManagedRun, samples: ManagedRunMetricSample[]) => RunPlotPoint[];
};

export type RunChartGroup = {
  id: RunChartGroupId;
  title: string;
  charts: RunChartDescriptor[];
};
