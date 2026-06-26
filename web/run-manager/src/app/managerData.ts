// web/run-manager/src/app/managerData.ts
import {
  fetchCheckpointCatalog,
  fetchConfigMetadata,
  fetchDrafts,
  fetchEvaluationData as fetchEvaluationPayload,
  fetchRuns,
  fetchSaveGames,
  fetchTemplates,
} from "@/shared/api/client";
import type {
  CheckpointCatalogResponse,
  ConfigMetadata,
  EvaluationBaselineSuite,
  ManagedDraft,
  ManagedEvaluation,
  ManagedEvaluationPreset,
  ManagedRun,
  ManagedSaveGame,
  ManagedTemplate,
} from "@/shared/api/contract";

export interface ManagerData {
  checkpointCatalog: CheckpointCatalogResponse | null;
  checkpointCatalogError: string | null;
  drafts: ManagedDraft[];
  evaluationBaselineSuites: EvaluationBaselineSuite[];
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  evaluationPresets: ManagedEvaluationPreset[];
  metadata: ConfigMetadata;
  runs: ManagedRun[];
  saveGames: ManagedSaveGame[];
  templates: ManagedTemplate[];
}

export async function loadManagerData(): Promise<ManagerData> {
  const [templates, drafts, runs, saveGames, evaluationData, checkpointCatalogData, metadata] =
    await Promise.all([
      fetchTemplates(),
      fetchDrafts(),
      fetchRuns(),
      fetchSaveGames(),
      fetchEvaluationData(),
      fetchCheckpointCatalogData(),
      fetchConfigMetadata(),
    ]);
  return {
    checkpointCatalog: checkpointCatalogData.catalog,
    checkpointCatalogError: checkpointCatalogData.error,
    drafts,
    evaluationBaselineSuites: evaluationData.baselineSuites,
    evaluationError: evaluationData.error,
    evaluations: evaluationData.evaluations,
    evaluationPresets: evaluationData.presets,
    metadata,
    runs,
    saveGames,
    templates,
  };
}

async function fetchCheckpointCatalogData(): Promise<{
  catalog: CheckpointCatalogResponse | null;
  error: string | null;
}> {
  try {
    return { catalog: await fetchCheckpointCatalog(), error: null };
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : "failed to load checkpoint catalog";
    return { catalog: null, error: message };
  }
}

async function fetchEvaluationData(): Promise<{
  error: string | null;
  baselineSuites: EvaluationBaselineSuite[];
  evaluations: ManagedEvaluation[];
  presets: ManagedEvaluationPreset[];
}> {
  try {
    const payload = await fetchEvaluationPayload();
    return {
      baselineSuites: payload.baseline_suites,
      error: null,
      evaluations: payload.evaluations,
      presets: payload.presets,
    };
  } catch (caught) {
    const message = caught instanceof Error ? caught.message : "failed to load evaluations";
    return { baselineSuites: [], error: message, evaluations: [], presets: [] };
  }
}
