// web/run-manager/src/app/workspace/liveSync.ts
import {
  type Dispatch,
  type RefObject,
  type SetStateAction,
  startTransition,
  useEffect,
} from "react";

import { compareRuns, updateCheckpointCatalogRuns } from "@/app/workspace/model";
import { sameRunDetailsById, trimRunDetailCache } from "@/app/workspace/runDetails";
import { sameRunPayload } from "@/app/workspace/runEquality";
import { fetchRuns, subscribeRunLiveUpdates } from "@/shared/api/client";
import type {
  CheckpointCatalogResponse,
  ManagedRun,
  ManagedRunDetail,
} from "@/shared/api/contract";

const RUN_LIVE_FALLBACK_POLL_MS = 5_000;

interface RunLiveSyncOptions {
  runDetailAccessOrderRef: RefObject<string[]>;
  setError: Dispatch<SetStateAction<string | null>>;
  setCheckpointCatalog: Dispatch<SetStateAction<CheckpointCatalogResponse | null>>;
  setRunDetailsById: Dispatch<SetStateAction<Record<string, ManagedRunDetail>>>;
  setRuns: Dispatch<SetStateAction<ManagedRun[]>>;
}

export function useRunLiveSync({
  runDetailAccessOrderRef,
  setError,
  setCheckpointCatalog,
  setRunDetailsById,
  setRuns,
}: RunLiveSyncOptions) {
  useEffect(() => {
    let unsubscribe: (() => void) | null = null;
    let fallbackPollInterval: number | null = null;
    let fallbackPollInFlight = false;
    let liveConnected = false;

    function applyRunSnapshot(nextRuns: ManagedRun[]) {
      const sortedRuns = [...nextRuns].filter(isVisibleRun).sort(compareRuns);
      const snapshotRunsById = new Map(nextRuns.map((run) => [run.id, run]));
      const snapshotRunIds = new Set(nextRuns.map((run) => run.id));
      startTransition(() => {
        setError(null);
        setCheckpointCatalog((current) => updateCheckpointCatalogRuns(current, snapshotRunsById));
        setRuns((current) => (sameRunPayload(current, sortedRuns) ? current : sortedRuns));
        setRunDetailsById((current) => {
          const next = trimRunDetailCache(current, snapshotRunIds, runDetailAccessOrderRef.current);
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
  }, [runDetailAccessOrderRef, setCheckpointCatalog, setError, setRunDetailsById, setRuns]);
}

function isVisibleRun(run: ManagedRun): boolean {
  return run.status !== "created" && run.status !== "archived";
}
