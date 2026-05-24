// src/rl_fzerox/apps/run_manager/web/src/app/workspace/useManagerData.ts
import { useCallback, useEffect, useRef, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { useRunLiveSync } from "@/app/workspace/liveSync";
import { runSummaryFromDetail } from "@/app/workspace/model";
import {
  hasRunDetail,
  rememberRunDetailAccess,
  trimRunDetailCache,
} from "@/app/workspace/runDetails";
import { fetchRun } from "@/shared/api/client";
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

  useRunLiveSync({
    runDetailAccessOrderRef,
    setError,
    setRunDetailsById,
    setRuns,
  });

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
