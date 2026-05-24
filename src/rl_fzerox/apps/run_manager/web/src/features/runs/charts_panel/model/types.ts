// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/model/types.ts
import type { RunPlotPoint } from "@/features/runs/charts/RunPlotCard";
import type { RunChartGroupId } from "@/features/runs/charts_panel/model/catalog";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";

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
