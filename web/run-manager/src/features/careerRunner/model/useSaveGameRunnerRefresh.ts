// web/run-manager/src/features/careerRunner/model/useSaveGameRunnerRefresh.ts
import { useEffect } from "react";

import type { ManagedSaveGame } from "@/shared/api/contract";

export function useSaveGameRunnerRefresh({
  onRefreshStatus,
  saveGame,
}: {
  onRefreshStatus: (saveGameId: string) => Promise<void>;
  saveGame: ManagedSaveGame | null;
}) {
  const saveGameId = saveGame?.id ?? null;
  const shouldRefresh =
    saveGame !== null && (saveGame.runner_active || saveGame.status === "running");

  useEffect(() => {
    if (saveGameId === null || !shouldRefresh) {
      return undefined;
    }
    const activeSaveGameId = saveGameId;
    let inFlight = false;
    async function refresh() {
      if (inFlight || document.visibilityState === "hidden") {
        return;
      }
      inFlight = true;
      try {
        await onRefreshStatus(activeSaveGameId);
      } finally {
        inFlight = false;
      }
    }
    const intervalId = window.setInterval(() => {
      void refresh();
    }, 1500);
    return () => window.clearInterval(intervalId);
  }, [onRefreshStatus, saveGameId, shouldRefresh]);
}
