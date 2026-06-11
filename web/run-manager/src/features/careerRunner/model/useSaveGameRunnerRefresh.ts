// web/run-manager/src/features/careerRunner/model/useSaveGameRunnerRefresh.ts
import { useEffect } from "react";

import type { ManagedSaveGame } from "@/shared/api/contract";

export function useSaveGameRunnerRefresh({
  onRefresh,
  saveGame,
}: {
  onRefresh: () => Promise<void>;
  saveGame: ManagedSaveGame | null;
}) {
  useEffect(() => {
    if (saveGame === null || (!saveGame.runner_active && saveGame.status !== "running")) {
      return undefined;
    }
    let inFlight = false;
    async function refresh() {
      if (inFlight || document.visibilityState === "hidden") {
        return;
      }
      inFlight = true;
      try {
        await onRefresh();
      } finally {
        inFlight = false;
      }
    }
    const intervalId = window.setInterval(() => {
      void refresh();
    }, 1500);
    return () => window.clearInterval(intervalId);
  }, [onRefresh, saveGame]);
}
