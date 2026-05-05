import { useCallback, useEffect, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { compareRuns } from "@/app/workspace/model";
import { fetchRuns } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
} from "@/shared/api/contract";

export function useManagerData() {
  const [drafts, setDrafts] = useState<ManagedDraft[]>([]);
  const [runs, setRuns] = useState<ManagedRun[]>([]);
  const [metadata, setMetadata] = useState<ConfigMetadata | null>(null);
  const [defaultConfig, setDefaultConfig] = useState<ManagedRunConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const reloadManagerData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const managerData = await loadManagerData();
      setDrafts(managerData.drafts);
      setRuns(managerData.runs);
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
    const intervalId = window.setInterval(() => {
      void fetchRuns()
        .then((nextRuns) => setRuns([...nextRuns].sort(compareRuns)))
        .catch(() => undefined);
    }, 2_000);
    return () => window.clearInterval(intervalId);
  }, []);

  return {
    defaultConfig,
    drafts,
    error,
    isLoading,
    metadata,
    reloadManagerData,
    runs,
    setDefaultConfig,
    setDrafts,
    setRuns,
  };
}
