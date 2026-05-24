// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/useRunChartMetrics.ts
import { useEffect, useState } from "react";
import { cachedMetricsByRun } from "@/features/runs/charts_panel/storage";
import type { RunMetricRangeMode } from "@/shared/api/client";
import { fetchFreshRunMetrics } from "@/shared/api/client";
import type { ManagedRunMetricSample } from "@/shared/api/contract";
import { useDocumentVisible } from "@/shared/browser/useDocumentVisible";

const MAX_METRIC_REQUEST_CONCURRENCY = 4;

export function useRunChartMetrics(
  selectedRunIds: readonly string[],
  rangeMode: RunMetricRangeMode,
  refreshingRunIds: readonly string[],
) {
  const [metricsByRun, setMetricsByRun] = useState<Record<string, ManagedRunMetricSample[]>>(() =>
    cachedMetricsByRun(selectedRunIds, rangeMode),
  );
  const [loadError, setLoadError] = useState<string | null>(null);
  const documentVisible = useDocumentVisible();

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
    if (!documentVisible) {
      return undefined;
    }

    let ignore = false;
    let inFlight = false;
    let activeController: AbortController | null = null;

    async function loadMetrics(runIds: readonly string[]) {
      if (inFlight) {
        return;
      }
      if (runIds.length === 0) {
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
        const samples = await loadRunMetricsWithLimit(runIds, rangeMode, controller.signal);
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

    void loadMetrics(selectedRunIds);
    const intervalId =
      refreshingRunIds.length > 0
        ? window.setInterval(() => {
            void loadMetrics(refreshingRunIds);
          }, 5_000)
        : null;
    return () => {
      ignore = true;
      activeController?.abort();
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [documentVisible, rangeMode, refreshingRunIds, selectedRunIds]);

  return { loadError, metricsByRun };
}

async function loadRunMetricsWithLimit(
  runIds: readonly string[],
  rangeMode: RunMetricRangeMode,
  signal: AbortSignal,
) {
  const samples: Array<readonly [string, ManagedRunMetricSample[]]> = [];
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < runIds.length) {
      const runId = runIds[nextIndex];
      nextIndex += 1;
      if (runId === undefined) {
        return;
      }
      samples.push([runId, await fetchFreshRunMetrics(runId, rangeMode, { signal })]);
    }
  }

  const workerCount = Math.min(MAX_METRIC_REQUEST_CONCURRENCY, runIds.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return samples;
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
