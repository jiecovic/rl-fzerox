// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/useRunChartMetrics.ts
import { useEffect, useState } from "react";
import type { RunMetricRangeMode } from "@/shared/api/client";
import { fetchFreshRunMetrics } from "@/shared/api/client";
import type { ManagedRunMetricSample } from "@/shared/api/contract";

import { cachedMetricsByRun } from "./storage";

export function useRunChartMetrics(
  selectedRunIds: readonly string[],
  rangeMode: RunMetricRangeMode,
) {
  const [metricsByRun, setMetricsByRun] = useState<Record<string, ManagedRunMetricSample[]>>(() =>
    cachedMetricsByRun(selectedRunIds, rangeMode),
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const allowed = new Set(selectedRunIds);
    setMetricsByRun((current) => ({
      ...Object.fromEntries(Object.entries(current).filter(([runId]) => allowed.has(runId))),
      ...cachedMetricsByRun(selectedRunIds, rangeMode),
    }));
  }, [rangeMode, selectedRunIds]);

  useEffect(() => {
    if (selectedRunIds.length === 0) {
      setLoadError(null);
      return undefined;
    }

    let ignore = false;

    async function loadMetrics() {
      try {
        setMetricsByRun((current) => ({
          ...current,
          ...cachedMetricsByRun(selectedRunIds, rangeMode),
        }));
        const samples = await Promise.all(
          selectedRunIds.map(
            async (runId) => [runId, await fetchFreshRunMetrics(runId, rangeMode)] as const,
          ),
        );
        if (!ignore) {
          setMetricsByRun((current) => ({ ...current, ...Object.fromEntries(samples) }));
          setLoadError(null);
        }
      } catch (caught) {
        if (!ignore) {
          setLoadError(caught instanceof Error ? caught.message : "failed to load run metrics");
        }
      }
    }

    void loadMetrics();
    const intervalId = window.setInterval(() => {
      void loadMetrics();
    }, 5_000);
    return () => {
      ignore = true;
      window.clearInterval(intervalId);
    };
  }, [rangeMode, selectedRunIds]);

  return { loadError, metricsByRun };
}
