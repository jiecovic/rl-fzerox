// src/rl_fzerox/apps/run_manager/web/src/app/workspace/useManagerData.ts
import { startTransition, useCallback, useEffect, useRef, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { compareRuns, runSummaryFromDetail } from "@/app/workspace/model";
import { fetchRun, fetchRuns, subscribeRunLiveUpdates } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
} from "@/shared/api/contract";

const MAX_CACHED_RUN_DETAILS = 16;
const RUN_LIVE_FALLBACK_POLL_MS = 5_000;

export function useManagerData() {
  const [drafts, setDrafts] = useState<ManagedDraft[]>([]);
  const [runs, setRuns] = useState<ManagedRun[]>([]);
  const [metadata, setMetadata] = useState<ConfigMetadata | null>(null);
  const [defaultConfig, setDefaultConfig] = useState<ManagedRunConfig | null>(null);
  const [runDetailsById, setRunDetailsById] = useState<Record<string, ManagedRunDetail>>({});
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const runDetailsRef = useRef(runDetailsById);
  const runDetailAccessOrderRef = useRef<string[]>([]);
  const runDetailRequestsRef = useRef(new Map<string, Promise<ManagedRunDetail>>());
  runDetailsRef.current = runDetailsById;

  const upsertRunDetail = useCallback((run: ManagedRunDetail) => {
    rememberRunDetailAccess(runDetailAccessOrderRef.current, run.id);
    setRunDetailsById((current) =>
      trimRunDetailCache({ ...current, [run.id]: run }, null, runDetailAccessOrderRef.current),
    );
  }, []);

  const loadRunDetail = useCallback(async (runId: string) => {
    const cached = runDetailsRef.current[runId];
    if (cached !== undefined) {
      rememberRunDetailAccess(runDetailAccessOrderRef.current, runId);
      return cached;
    }
    const inflight = runDetailRequestsRef.current.get(runId);
    if (inflight !== undefined) {
      return inflight;
    }
    const request = fetchRun(runId)
      .then((run) => {
        rememberRunDetailAccess(runDetailAccessOrderRef.current, run.id);
        setRunDetailsById((current) =>
          trimRunDetailCache({ ...current, [run.id]: run }, null, runDetailAccessOrderRef.current),
        );
        return run;
      })
      .finally(() => {
        runDetailRequestsRef.current.delete(runId);
      });
    runDetailRequestsRef.current.set(runId, request);
    return request;
  }, []);

  const reloadManagerData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const managerData = await loadManagerData();
      const runDetails = managerData.runs.filter(hasRunDetail);
      for (const run of runDetails) {
        rememberRunDetailAccess(runDetailAccessOrderRef.current, run.id);
      }
      setDrafts(managerData.drafts);
      setRuns(managerData.runs.map((run) => (hasRunDetail(run) ? runSummaryFromDetail(run) : run)));
      setRunDetailsById((current) =>
        trimRunDetailCache(
          {
            ...current,
            ...Object.fromEntries(runDetails.map((run) => [run.id, run])),
          },
          null,
          runDetailAccessOrderRef.current,
        ),
      );
      setMetadata(managerData.metadata);
      setDefaultConfig((current) => current ?? managerData.templates[0]?.config ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load run manager data");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void reloadManagerData();
  }, [reloadManagerData]);

  useEffect(() => {
    let unsubscribe: (() => void) | null = null;
    let fallbackPollInterval: number | null = null;
    let fallbackPollInFlight = false;
    let liveConnected = false;

    function applyRunSnapshot(nextRuns: ManagedRun[]) {
      const sortedRuns = [...nextRuns].sort(compareRuns);
      const visibleRunIds = new Set(sortedRuns.map((run) => run.id));
      startTransition(() => {
        setError(null);
        setRuns((current) => (sameRunPayload(current, sortedRuns) ? current : sortedRuns));
        setRunDetailsById((current) => {
          const next = trimRunDetailCache(current, visibleRunIds, runDetailAccessOrderRef.current);
          return next === current || sameRunDetailsById(current, next) ? current : next;
        });
      });
    }

    async function pollRunsFallback() {
      if (fallbackPollInFlight || liveConnected || document.visibilityState === "hidden") {
        return;
      }
      fallbackPollInFlight = true;
      try {
        applyRunSnapshot(await fetchRuns());
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "failed to refresh run list");
      } finally {
        fallbackPollInFlight = false;
      }
    }

    function startFallbackPolling() {
      if (fallbackPollInterval !== null) {
        return;
      }
      void pollRunsFallback();
      fallbackPollInterval = window.setInterval(() => {
        void pollRunsFallback();
      }, RUN_LIVE_FALLBACK_POLL_MS);
    }

    function stopFallbackPolling() {
      if (fallbackPollInterval === null) {
        return;
      }
      window.clearInterval(fallbackPollInterval);
      fallbackPollInterval = null;
    }

    function connect() {
      if (unsubscribe !== null || document.visibilityState === "hidden") {
        return;
      }
      unsubscribe = subscribeRunLiveUpdates({
        onConnectionChange: (connected) => {
          liveConnected = connected;
          if (connected) {
            stopFallbackPolling();
            setError(null);
            return;
          }
          startFallbackPolling();
        },
        onError: (caught) => {
          setError(caught.message);
        },
        onRuns: applyRunSnapshot,
      });
    }

    function disconnect() {
      unsubscribe?.();
      unsubscribe = null;
      liveConnected = false;
      stopFallbackPolling();
    }

    function syncVisibility() {
      if (document.visibilityState === "hidden") {
        disconnect();
        return;
      }
      connect();
    }

    connect();
    document.addEventListener("visibilitychange", syncVisibility);
    return () => {
      document.removeEventListener("visibilitychange", syncVisibility);
      disconnect();
    };
  }, []);

  return {
    defaultConfig,
    drafts,
    error,
    isLoading,
    loadRunDetail,
    metadata,
    reloadManagerData,
    runs,
    runDetailsById,
    setDefaultConfig,
    setDrafts,
    setRuns,
    upsertRunDetail,
  };
}

function hasRunDetail(run: ManagedRun): run is ManagedRunDetail {
  return "config" in run;
}

function sameRunPayload(left: readonly ManagedRun[], right: readonly ManagedRun[]) {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (!sameRunSummary(left[index], right[index])) {
      return false;
    }
  }
  return true;
}

function sameRunSummary(left: ManagedRun, right: ManagedRun) {
  if (
    left.id !== right.id ||
    left.name !== right.name ||
    left.status !== right.status ||
    left.config_hash !== right.config_hash ||
    left.action_repeat !== right.action_repeat ||
    left.created_at !== right.created_at ||
    left.lineage_id !== right.lineage_id ||
    left.lineage_step_offset !== right.lineage_step_offset ||
    left.parent_run_id !== right.parent_run_id ||
    left.source_run_id !== right.source_run_id ||
    left.source_artifact !== right.source_artifact ||
    left.source_num_timesteps !== right.source_num_timesteps ||
    left.started_at !== right.started_at ||
    left.stopped_at !== right.stopped_at ||
    left.pending_command !== right.pending_command ||
    left.worker_heartbeat_at !== right.worker_heartbeat_at
  ) {
    return false;
  }
  if (left.lineage_groups.join("\0") !== right.lineage_groups.join("\0")) {
    return false;
  }
  return sameRuntime(left.runtime, right.runtime) && sameRecentEvents(left, right);
}

function sameRuntime(left: ManagedRun["runtime"], right: ManagedRun["runtime"]) {
  if (left === null || right === null) {
    return left === right;
  }
  return (
    left.total_timesteps === right.total_timesteps &&
    left.num_timesteps === right.num_timesteps &&
    left.progress_fraction === right.progress_fraction &&
    left.updated_at === right.updated_at &&
    left.fps === right.fps &&
    left.episode_reward_mean === right.episode_reward_mean &&
    left.episode_length_mean === right.episode_length_mean &&
    left.approx_kl === right.approx_kl &&
    left.entropy_loss === right.entropy_loss &&
    left.value_loss === right.value_loss &&
    left.policy_gradient_loss === right.policy_gradient_loss
  );
}

function sameRecentEvents(left: ManagedRun, right: ManagedRun) {
  if (left.recent_events.length !== right.recent_events.length) {
    return false;
  }
  return left.recent_events.every((event, index) => {
    const other = right.recent_events[index];
    return (
      other !== undefined &&
      event.created_at === other.created_at &&
      event.kind === other.kind &&
      event.message === other.message
    );
  });
}

function rememberRunDetailAccess(order: string[], runId: string) {
  const existingIndex = order.indexOf(runId);
  if (existingIndex !== -1) {
    order.splice(existingIndex, 1);
  }
  order.push(runId);
}

function trimRunDetailCache(
  current: Record<string, ManagedRunDetail>,
  visibleRunIds: Set<string> | null,
  accessOrder: string[],
) {
  let entries = Object.entries(current);
  if (visibleRunIds !== null) {
    entries = entries.filter(([runId]) => visibleRunIds.has(runId));
  }
  const entryIds = new Set(entries.map(([runId]) => runId));
  for (const [runId] of entries) {
    if (!accessOrder.includes(runId)) {
      accessOrder.push(runId);
    }
  }
  const prunedOrder = accessOrder.filter((runId) => entryIds.has(runId));
  accessOrder.splice(0, accessOrder.length, ...prunedOrder);
  if (entries.length <= MAX_CACHED_RUN_DETAILS) {
    return entries.length === Object.keys(current).length ? current : Object.fromEntries(entries);
  }
  const keepIds = new Set(prunedOrder.slice(-MAX_CACHED_RUN_DETAILS));
  accessOrder.splice(0, accessOrder.length, ...prunedOrder.filter((runId) => keepIds.has(runId)));
  return Object.fromEntries(entries.filter(([runId]) => keepIds.has(runId)));
}

function sameRunDetailsById(
  left: Record<string, ManagedRunDetail>,
  right: Record<string, ManagedRunDetail>,
) {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  return leftKeys.every((key) => left[key] === right[key]);
}
