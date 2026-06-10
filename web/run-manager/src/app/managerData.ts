// web/run-manager/src/app/managerData.ts
import {
  fetchConfigMetadata,
  fetchDrafts,
  fetchRuns,
  fetchSaveGames,
  fetchTemplates,
} from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedSaveGame,
  ManagedTemplate,
} from "@/shared/api/contract";

export interface ManagerData {
  drafts: ManagedDraft[];
  metadata: ConfigMetadata;
  runs: ManagedRun[];
  saveGames: ManagedSaveGame[];
  templates: ManagedTemplate[];
}

export async function loadManagerData(): Promise<ManagerData> {
  const [templates, drafts, runs, saveGames, metadata] = await Promise.all([
    fetchTemplates(),
    fetchDrafts(),
    fetchRuns(),
    fetchSaveGames(),
    fetchConfigMetadata(),
  ]);
  return { drafts, metadata, runs, saveGames, templates };
}
