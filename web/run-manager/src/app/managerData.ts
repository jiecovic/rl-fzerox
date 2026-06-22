// web/run-manager/src/app/managerData.ts
import {
  fetchConfigMetadata,
  fetchDrafts,
  fetchEvaluations,
  fetchRuns,
  fetchSaveGames,
  fetchTemplates,
} from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedEvaluation,
  ManagedRun,
  ManagedSaveGame,
  ManagedTemplate,
} from "@/shared/api/contract";

export interface ManagerData {
  drafts: ManagedDraft[];
  evaluations: ManagedEvaluation[];
  metadata: ConfigMetadata;
  runs: ManagedRun[];
  saveGames: ManagedSaveGame[];
  templates: ManagedTemplate[];
}

export async function loadManagerData(): Promise<ManagerData> {
  const [templates, drafts, runs, saveGames, evaluations, metadata] = await Promise.all([
    fetchTemplates(),
    fetchDrafts(),
    fetchRuns(),
    fetchSaveGames(),
    fetchEvaluations(),
    fetchConfigMetadata(),
  ]);
  return { drafts, evaluations, metadata, runs, saveGames, templates };
}
