// src/rl_fzerox/apps/run_manager/web/src/widgets/runCharts/chartsPanel/storage.ts

import type { RunMetricRangeMode } from "@/shared/api/client";
import { getCachedRunMetrics } from "@/shared/api/client";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";
import { defaultSelectedRunIds } from "@/widgets/runCharts/chartsPanel/model";

export const CHART_SELECTION_STORAGE_KEY = "run-chart-selected-runs";

export function cachedMetricsByRun(runIds: readonly string[], rangeMode: RunMetricRangeMode) {
  return Object.fromEntries(
    runIds
      .map((runId) => [runId, getCachedRunMetrics(runId, rangeMode)] as const)
      .filter((entry): entry is readonly [string, ManagedRunMetricSample[]] => entry[1] !== null),
  );
}

export function readStoredSelectedRunIds(runs: ManagedRun[], focusedRunId: string | null) {
  const defaults = defaultSelectedRunIds(runs, focusedRunId);
  if (typeof window === "undefined") {
    return defaults;
  }
  try {
    const raw = window.localStorage.getItem(CHART_SELECTION_STORAGE_KEY);
    if (raw === null) {
      return defaults;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return defaults;
    }
    const allowed = new Set(runs.map((run) => run.id));
    const filtered = parsed.filter(
      (value): value is string => typeof value === "string" && allowed.has(value),
    );
    return filtered.length === 0 ? defaults : filtered;
  } catch {
    return defaults;
  }
}

export function writeStoredSelectedRunIds(runIds: readonly string[]) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(CHART_SELECTION_STORAGE_KEY, JSON.stringify(runIds));
}
