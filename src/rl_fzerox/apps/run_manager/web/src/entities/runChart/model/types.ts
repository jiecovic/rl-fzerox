// src/rl_fzerox/apps/run_manager/web/src/entities/runChart/model/types.ts

import type { RunChartGroupId } from "@/entities/runChart/model/catalog";
import type { RunPlotPoint } from "@/entities/runChart/ui/RunPlotCard";
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
