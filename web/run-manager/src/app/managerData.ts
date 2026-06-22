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
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  metadata: ConfigMetadata;
  runs: ManagedRun[];
  saveGames: ManagedSaveGame[];
  templates: ManagedTemplate[];
}

export async function loadManagerData(): Promise<ManagerData> {
  const [templates, drafts, runs, saveGames, evaluationData, metadata] = await Promise.all([
    fetchTemplates(),
    fetchDrafts(),
    fetchRuns(),
    fetchSaveGames(),
    fetchEvaluationData(),
    fetchConfigMetadata(),
  ]);
  return {
    drafts,
    evaluationError: evaluationData.error,
    evaluations: evaluationData.evaluations,
    metadata,
    runs,
    saveGames,
    templates,
  };
}

async function fetchEvaluationData(): Promise<{
  error: string | null;
  evaluations: ManagedEvaluation[];
}> {
  try {
    return { error: null, evaluations: await fetchEvaluations() };
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : "failed to load evaluations";
    return { error: message, evaluations: [] };
  }
}
