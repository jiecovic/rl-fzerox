// src/rl_fzerox/apps/run_manager/web/src/features/save_games/useSaveGameRunnerRefresh.ts
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
    const intervalId = window.setInterval(() => {
      void onRefresh();
    }, 1500);
    return () => window.clearInterval(intervalId);
  }, [onRefresh, saveGame]);
}
