// src/rl_fzerox/apps/run_manager/web/src/app/workspace/useManagerData.ts
import { startTransition, useCallback, useEffect, useRef, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { compareRuns, runSummaryFromDetail } from "@/app/workspace/model";
import { fetchRun, fetchRuns } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
} from "@/shared/api/contract";

export function useManagerData() {
  const [drafts, setDrafts] = useState<ManagedDraft[]>([]);
  const [runs, setRuns] = useState<ManagedRun[]>([]);
  const [metadata, setMetadata] = useState<ConfigMetadata | null>(null);
  const [defaultConfig, setDefaultConfig] = useState<ManagedRunConfig | null>(null);
  const [runDetailsById, setRunDetailsById] = useState<Record<string, ManagedRunDetail>>({});
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const runDetailsRef = useRef(runDetailsById);
  const pollInFlightRef = useRef(false);
  const runDetailRequestsRef = useRef(new Map<string, Promise<ManagedRunDetail>>());
  runDetailsRef.current = runDetailsById;

  const upsertRunDetail = useCallback((run: ManagedRunDetail) => {
    setRunDetailsById((current) => ({ ...current, [run.id]: run }));
  }, []);

  const loadRunDetail = useCallback(async (runId: string) => {
    const cached = runDetailsRef.current[runId];
    if (cached !== undefined) {
      return cached;
    }
    const inflight = runDetailRequestsRef.current.get(runId);
    if (inflight !== undefined) {
      return inflight;
    }
    const request = fetchRun(runId)
      .then((run) => {
        setRunDetailsById((current) => ({ ...current, [run.id]: run }));
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
      setDrafts(managerData.drafts);
      setRuns(managerData.runs.map((run) => (hasRunDetail(run) ? runSummaryFromDetail(run) : run)));
      setRunDetailsById((current) => ({
        ...current,
        ...Object.fromEntries(runDetails.map((run) => [run.id, run])),
      }));
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
    async function reloadRuns() {
      if (pollInFlightRef.current) {
        return;
      }
      pollInFlightRef.current = true;
      try {
        const nextRuns = await fetchRuns();
        const sortedRuns = [...nextRuns].sort(compareRuns);
        const visibleRunIds = new Set(sortedRuns.map((run) => run.id));
        startTransition(() => {
          setRuns((current) => (sameRunPayload(current, sortedRuns) ? current : sortedRuns));
          setRunDetailsById((current) => {
            const entries = Object.entries(current).filter(([runId]) => visibleRunIds.has(runId));
            return entries.length === Object.keys(current).length
              ? current
              : Object.fromEntries(entries);
          });
        });
      } catch {
        // Keep the current snapshot when transient polling fails.
      } finally {
        pollInFlightRef.current = false;
      }
    }

    const intervalId = window.setInterval(() => {
      void reloadRuns();
    }, 2_000);
    return () => window.clearInterval(intervalId);
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
