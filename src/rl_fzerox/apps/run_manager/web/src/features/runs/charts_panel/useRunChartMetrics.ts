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
    let inFlight = false;
    let activeController: AbortController | null = null;

    async function loadMetrics() {
      if (inFlight) {
        return;
      }
      inFlight = true;
      const controller = new AbortController();
      activeController = controller;
      try {
        setMetricsByRun((current) => ({
          ...current,
          ...cachedMetricsByRun(selectedRunIds, rangeMode),
        }));
        const samples = await Promise.all(
          selectedRunIds.map(
            async (runId) =>
              [
                runId,
                await fetchFreshRunMetrics(runId, rangeMode, { signal: controller.signal }),
              ] as const,
          ),
        );
        if (!ignore) {
          setMetricsByRun((current) => ({ ...current, ...Object.fromEntries(samples) }));
          setLoadError(null);
        }
      } catch (caught) {
        if (isAbortError(caught)) {
          return;
        }
        if (!ignore) {
          setLoadError(caught instanceof Error ? caught.message : "failed to load run metrics");
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        inFlight = false;
      }
    }

    void loadMetrics();
    const intervalId = window.setInterval(() => {
      void loadMetrics();
    }, 5_000);
    return () => {
      ignore = true;
      activeController?.abort();
      window.clearInterval(intervalId);
    };
  }, [rangeMode, selectedRunIds]);

  return { loadError, metricsByRun };
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
