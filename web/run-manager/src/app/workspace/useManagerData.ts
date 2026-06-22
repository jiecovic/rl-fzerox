// web/run-manager/src/app/workspace/useManagerData.ts
import { useCallback, useEffect, useRef, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { useRunLiveSync } from "@/app/workspace/liveSync";
import {
  compareSaveGames,
  runSummaryFromDetail,
  upsertSaveGameStatus,
} from "@/app/workspace/model";
import {
  hasRunDetail,
  rememberRunDetailAccess,
  trimRunDetailCache,
} from "@/app/workspace/runDetails";
import { fetchRun, fetchSaveGameStatus, fetchSaveGames } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedEvaluation,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
  ManagedSaveGame,
} from "@/shared/api/contract";

interface ReloadManagerDataOptions {
  showLoading?: boolean;
}

export function useManagerData() {
  const [drafts, setDrafts] = useState<ManagedDraft[]>([]);
  const [evaluations, setEvaluations] = useState<ManagedEvaluation[]>([]);
  const [runs, setRuns] = useState<ManagedRun[]>([]);
  const [saveGames, setSaveGames] = useState<ManagedSaveGame[]>([]);
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

  const reloadManagerData = useCallback(async (options: ReloadManagerDataOptions = {}) => {
    const showLoading = options.showLoading ?? true;
    if (showLoading) {
      setIsLoading(true);
    }
    setError(null);
    try {
      const managerData = await loadManagerData();
      const runDetails = managerData.runs.filter(hasRunDetail);
      for (const run of runDetails) {
        rememberRunDetailAccess(runDetailAccessOrderRef.current, run.id);
      }
      setDrafts(managerData.drafts);
      setEvaluations(managerData.evaluations);
      setRuns(managerData.runs.map((run) => (hasRunDetail(run) ? runSummaryFromDetail(run) : run)));
      setSaveGames(managerData.saveGames);
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
      if (showLoading) {
        setIsLoading(false);
      }
    }
  }, []);

  const refreshSaveGames = useCallback(async () => {
    try {
      const nextSaveGames = [...(await fetchSaveGames())].sort(compareSaveGames);
      setSaveGames((current) =>
        saveGamePayloadKey(current) === saveGamePayloadKey(nextSaveGames) ? current : nextSaveGames,
      );
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to refresh career saves");
    }
  }, []);

  const refreshSaveGameStatus = useCallback(async (saveGameId: string) => {
    try {
      const status = await fetchSaveGameStatus(saveGameId);
      setSaveGames((current) => upsertSaveGameStatus(current, status));
      setError(null);
    } catch (caught) {
      try {
        const nextSaveGames = [...(await fetchSaveGames())].sort(compareSaveGames);
        setSaveGames((current) =>
          saveGamePayloadKey(current) === saveGamePayloadKey(nextSaveGames)
            ? current
            : nextSaveGames,
        );
        setError(null);
      } catch {
        setError(caught instanceof Error ? caught.message : "failed to refresh career save status");
      }
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
    evaluations,
    isLoading,
    loadRunDetail,
    metadata,
    reloadManagerData,
    refreshSaveGameStatus,
    refreshSaveGames,
    runs,
    runDetailsById,
    saveGames,
    setDefaultConfig,
    setDrafts,
    setEvaluations,
    setRuns,
    setSaveGames,
    upsertRunDetail,
  };
}

function saveGamePayloadKey(saveGames: readonly ManagedSaveGame[]) {
  return JSON.stringify(saveGames);
}
