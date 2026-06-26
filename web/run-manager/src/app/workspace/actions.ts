// web/run-manager/src/app/workspace/actions.ts
import type { Dispatch, SetStateAction } from "react";
import { workspaceEvaluationActions } from "@/app/workspace/actions/evaluations";
import { workspaceSaveGameActions } from "@/app/workspace/actions/saveGames";
import {
  draftForkSource,
  forkInitialConfig,
  nextAvailableDraftName,
  nextForkDraftName,
  runSummaryFromDetail,
  upsertDraft,
  upsertRun,
} from "@/app/workspace/model";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import type { DraftEditorSession, ForkSourceEngineTuning } from "@/app/workspace/types";
import {
  clearRunAltBaselines,
  clearRunCourseAltBaselines,
  createDraftWithSource,
  deleteDraft,
  deleteLineage,
  deleteRun,
  exportRunBundle,
  fetchRunEngineTuningState,
  importRunBundle,
  launchRun,
  openRunDirectory,
  renameRun,
  resetRunEngineTuningState,
  resetRunTrackSamplingState,
  resumeRun,
  stopRun,
  updateDraftWithSource,
  updateLineageGroups,
  watchRun,
} from "@/shared/api/client";
import type {
  CareerModeRunnerLaunchRequest,
  CreateEvaluationPresetRequest,
  CreateEvaluationRequest,
  EngineTuningSourceAction,
  ManagedDraft,
  ManagedEvaluation,
  ManagedEvaluationPreset,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
  ManagedSaveGame,
  PolicyPlaybackMode,
  SaveEngineTuningCourseSetupRecommendation,
  SaveGameRunnerSettingsUpdateRequest,
  SavePolicyArtifact,
  SavePolicySourceKind,
  StartEvaluationRequest,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";

interface UseWorkspaceActionsOptions {
  drafts: ManagedDraft[];
  loadRunDetail: (runId: string) => Promise<ManagedRunDetail>;
  reloadManagerData: (options?: { showLoading?: boolean }) => Promise<void>;
  runs: ManagedRun[];
  sessions: WorkspaceSessions;
  setGlobalError: Dispatch<SetStateAction<string | null>>;
  setDrafts: Dispatch<SetStateAction<ManagedDraft[]>>;
  setEvaluations: Dispatch<SetStateAction<ManagedEvaluation[]>>;
  setRuns: Dispatch<SetStateAction<ManagedRun[]>>;
  setSaveGames: Dispatch<SetStateAction<ManagedSaveGame[]>>;
  upsertRunDetail: (run: ManagedRunDetail) => void;
}

export interface WorkspaceActions {
  createDraftFromManagedRun: (runId: string) => Promise<void>;
  createManagedEvaluation: (request: CreateEvaluationRequest) => Promise<ManagedEvaluation>;
  createManagedEvaluationPreset: (
    request: CreateEvaluationPresetRequest,
  ) => Promise<ManagedEvaluationPreset>;
  removeManagedEvaluationPreset: (preset: ManagedEvaluationPreset) => Promise<void>;
  cancelManagedEvaluation: (evaluation: ManagedEvaluation) => Promise<ManagedEvaluation>;
  startManagedEvaluation: (
    evaluation: ManagedEvaluation,
    request: StartEvaluationRequest,
  ) => Promise<ManagedEvaluation>;
  renameManagedEvaluation: (evaluationId: string, name: string) => Promise<void>;
  createManagedSaveGame: (name: string) => Promise<ManagedSaveGame>;
  forkManagedRun: (
    runId: string,
    artifact: "latest" | "best",
    copyAltBaselines: boolean,
  ) => Promise<void>;
  launchTrainingRun: (
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
    draftId: string | null,
    engineTuningSourceAction?: EngineTuningSourceAction,
  ) => Promise<ManagedRunDetail>;
  openManagedRunDirectory: (runId: string) => Promise<void>;
  openManagedSaveGameDirectory: (saveGameId: string) => Promise<void>;
  upsertManagedSaveCourseSetup: (request: {
    engineSettingRawValue: number;
    policyArtifact: SavePolicyArtifact;
    policySourceId: string;
    policySourceKind: SavePolicySourceKind;
    saveGameId: string;
    courseId?: string | null;
    cupId?: string | null;
    difficulty?: string | null;
  }) => Promise<ManagedSaveGame>;
  upsertManagedSaveCupSetup: (request: {
    cupId: string;
    saveGameId: string;
    vehicleId: string;
    difficulty?: string | null;
  }) => Promise<ManagedSaveGame>;
  importManagedSaveEngineTuning: (request: {
    courseSetups: readonly {
      courseId: string;
      cupId: string;
      difficulty?: string | null;
      vehicleId: string;
    }[];
    policyArtifact: SavePolicyArtifact;
    policyRunId: string;
    saveGameId: string;
  }) => Promise<readonly SaveEngineTuningCourseSetupRecommendation[]>;
  exportManagedRun: (run: ManagedRun) => Promise<void>;
  importManagedRunBundle: (file: File) => Promise<void>;
  removeDraft: (id: string) => Promise<void>;
  removeManagedEvaluation: (evaluation: ManagedEvaluation) => Promise<void>;
  removeLineage: (lineageId: string) => Promise<void>;
  removeRun: (run: ManagedRun) => Promise<void>;
  removeSaveGame: (saveGame: ManagedSaveGame) => Promise<void>;
  renameManagedRun: (runId: string, name: string) => Promise<void>;
  renameManagedSaveGame: (saveGameId: string, name: string) => Promise<void>;
  updateManagedSaveRunnerSettings: (
    request: SaveGameRunnerSettingsUpdateRequest,
  ) => Promise<ManagedSaveGame>;
  clearManagedRunAltBaselines: (runId: string) => Promise<void>;
  clearManagedRunCourseAltBaselines: (runId: string, courseKey: string) => Promise<void>;
  resetManagedRunEngineTuning: (runId: string) => Promise<void>;
  resetManagedRunTrackPool: (runId: string) => Promise<void>;
  resumeManagedRun: (runId: string) => Promise<void>;
  saveDraft: (
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
  ) => Promise<ManagedDraft>;
  stopManagedRun: (runId: string) => Promise<void>;
  updateManagedLineageGroups: (lineageId: string, groupNames: readonly string[]) => Promise<void>;
  updateExistingDraft: (
    sessionId: DraftEditorSession["sessionId"],
    id: string,
    name: string,
    config: ManagedRunConfig,
  ) => Promise<ManagedDraft>;
  watchManagedRun: (
    runId: string,
    artifact: "latest" | "best",
    device: WatchDevice,
    renderer: WatchRenderer,
    policyMode: PolicyPlaybackMode,
  ) => Promise<"started" | "already_running">;
  startManagedCareerMode: (
    request: CareerModeRunnerLaunchRequest,
  ) => Promise<"started" | "already_running">;
  setGlobalError: Dispatch<SetStateAction<string | null>>;
}

export function useWorkspaceActions({
  drafts: _drafts,
  loadRunDetail,
  reloadManagerData,
  runs,
  sessions,
  setGlobalError,
  setDrafts,
  setEvaluations,
  setRuns,
  setSaveGames,
  upsertRunDetail,
}: UseWorkspaceActionsOptions): WorkspaceActions {
  const evaluationActions = workspaceEvaluationActions({
    reloadManagerData,
    sessions,
    setEvaluations,
    setGlobalError,
  });
  const saveGameActions = workspaceSaveGameActions({
    reloadManagerData,
    sessions,
    setSaveGames,
  });

  async function saveDraft(
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
  ) {
    const session =
      sessions.draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const draft = await createDraftWithSource(
      name,
      config,
      session?.forkSource?.runId ?? null,
      session?.forkSource?.artifact ?? null,
    );
    setDrafts((current) => upsertDraft(current, draft));
    sessions.patchDraftEditor(sessionId, {
      currentConfig: draft.config,
      currentDraftName: draft.name,
      draftId: draft.id,
      forkSource: mergeForkSourceCopyChoice(draftForkSource(draft), session?.forkSource),
      initialDraftName: draft.name,
      initialConfig: null,
      loadedDraft: draft,
      title: draft.name,
    });
    return draft;
  }

  async function updateExistingDraft(
    sessionId: DraftEditorSession["sessionId"],
    id: string,
    name: string,
    config: ManagedRunConfig,
  ) {
    const session =
      sessions.draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const draft = await updateDraftWithSource(
      id,
      name,
      config,
      session?.forkSource?.runId ?? null,
      session?.forkSource?.artifact ?? null,
    );
    setDrafts((current) => upsertDraft(current, draft));
    sessions.patchDraftEditor(sessionId, {
      currentConfig: draft.config,
      currentDraftName: draft.name,
      draftId: draft.id,
      forkSource: mergeForkSourceCopyChoice(draftForkSource(draft), session?.forkSource),
      initialDraftName: draft.name,
      initialConfig: null,
      loadedDraft: draft,
      title: draft.name,
    });
    return draft;
  }

  async function removeDraft(id: string) {
    await deleteDraft(id);
    setDrafts((current) => current.filter((draft) => draft.id !== id));
    sessions.closeEditorsForDraft(id);
  }

  async function removeRun(run: ManagedRun) {
    await deleteRun(run.id);
    setRuns((current) => current.filter((candidate) => candidate.id !== run.id));
    sessions.closeRunTabsForRun(run.id);
    sessions.setChartsFocusRunId((current) => (current === run.id ? null : current));
  }

  async function removeLineage(lineageId: string) {
    const lineageRunIds = runs
      .filter((candidate) => candidate.lineage_id === lineageId)
      .map((candidate) => candidate.id);
    await deleteLineage(lineageId);
    sessions.closeRunTabsForRuns(lineageRunIds);
    sessions.closeEditorsForSourceRuns(lineageRunIds);
    sessions.setChartsFocusRunId((current) =>
      current !== null && lineageRunIds.includes(current) ? null : current,
    );
    await reloadManagerData();
  }

  async function launchTrainingRun(
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
    draftId: string | null,
    engineTuningSourceAction?: EngineTuningSourceAction,
  ) {
    const session =
      sessions.draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const sourceRunId = session?.forkSource?.runId ?? null;
    const sourceArtifact = session?.forkSource?.artifact ?? null;
    const run = await launchRun(
      name,
      config,
      draftId,
      sourceRunId,
      sourceArtifact,
      sourceRunId === null || sourceArtifact === null
        ? true
        : (session?.forkSource?.copyAltBaselines ?? true),
      engineTuningSourceAction,
    );
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    upsertRunDetail(run);
    sessions.openRun(run);
    return run;
  }

  async function forkManagedRun(
    runId: string,
    artifact: "latest" | "best",
    copyAltBaselines: boolean,
  ) {
    const sourceRun = runs.find((candidate) => candidate.id === runId) ?? null;
    const [sourceDetail, sourceTuning] = await Promise.all([
      loadRunDetail(runId),
      fetchRunEngineTuningState(runId, artifact),
    ]);
    const initialDraftName =
      sourceRun === null
        ? nextAvailableDraftName("fork", allKnownNames())
        : nextForkDraftName(sourceRun, runs, allKnownNames());
    const initialConfig = forkInitialConfig(sourceDetail.config);
    sessions.createForkDraft({
      artifact,
      copyAltBaselines,
      initialConfig,
      initialDraftName,
      runId,
      sourceEngineTunerBackend: sourceDetail.config.vehicle.adaptive_engine_tuner_backend,
      sourceEngineTuning:
        sourceTuning.state === null ? null : forkSourceEngineTuning(sourceDetail.config),
      sourceEngineTuningKnown: true,
    });
  }

  async function createDraftFromManagedRun(runId: string) {
    const sourceRun = runs.find((candidate) => candidate.id === runId) ?? null;
    const sourceDetail = await loadRunDetail(runId);
    const sourceName = sourceRun?.name ?? sourceDetail.name;
    const initialDraftName = nextAvailableDraftName(`${sourceName} draft`, allKnownNames());
    const draft = await createDraftWithSource(initialDraftName, sourceDetail.config, null, null);
    setDrafts((current) => upsertDraft(current, draft));
    sessions.openDraft(draft);
  }

  async function stopManagedRun(runId: string) {
    const run = await stopRun(runId);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    upsertRunDetail(run);
  }

  async function resumeManagedRun(runId: string) {
    const run = await resumeRun(runId);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    upsertRunDetail(run);
  }

  async function renameManagedRun(runId: string, name: string) {
    const run = await renameRun(runId, name);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    upsertRunDetail(run);
  }

  async function updateManagedLineageGroups(lineageId: string, groupNames: readonly string[]) {
    const lineageGroups = await updateLineageGroups(lineageId, groupNames);
    setRuns((current) =>
      current.map((run) =>
        run.lineage_id === lineageId ? { ...run, lineage_groups: lineageGroups } : run,
      ),
    );
  }

  async function openManagedRunDirectory(runId: string) {
    await openRunDirectory(runId);
  }

  async function exportManagedRun(run: ManagedRun) {
    await exportRunBundle(run);
  }

  async function importManagedRunBundle(file: File) {
    const run = await importRunBundle(file);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    upsertRunDetail(run);
    sessions.openRun(run);
    await reloadManagerData();
  }

  async function watchManagedRun(
    runId: string,
    artifact: "latest" | "best",
    device: WatchDevice,
    renderer: WatchRenderer,
    policyMode: PolicyPlaybackMode,
  ): Promise<"started" | "already_running"> {
    return await watchRun(runId, artifact, device, renderer, policyMode);
  }

  async function resetManagedRunTrackPool(runId: string) {
    await resetRunTrackSamplingState(runId);
  }

  async function resetManagedRunEngineTuning(runId: string) {
    const run = await resetRunEngineTuningState(runId);
    upsertRunDetail(run);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    await reloadManagerData({ showLoading: false });
  }

  async function clearManagedRunAltBaselines(runId: string) {
    const run = await clearRunAltBaselines(runId);
    upsertRunDetail(run);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    await reloadManagerData({ showLoading: false });
  }

  async function clearManagedRunCourseAltBaselines(runId: string, courseKey: string) {
    const run = await clearRunCourseAltBaselines(runId, courseKey);
    upsertRunDetail(run);
    setRuns((current) => upsertRun(current, runSummaryFromDetail(run)));
    await reloadManagerData({ showLoading: false });
  }

  function allKnownNames() {
    const names = new Set<string>();
    for (const draft of _drafts) {
      names.add(draft.name);
    }
    for (const run of runs) {
      names.add(run.name);
    }
    for (const session of sessions.draftEditors) {
      names.add(session.title);
    }
    return names;
  }

  return {
    ...evaluationActions,
    ...saveGameActions,
    createDraftFromManagedRun,
    forkManagedRun,
    launchTrainingRun,
    openManagedRunDirectory,
    exportManagedRun,
    importManagedRunBundle,
    removeDraft,
    removeLineage,
    removeRun,
    renameManagedRun,
    clearManagedRunAltBaselines,
    clearManagedRunCourseAltBaselines,
    resetManagedRunEngineTuning,
    resetManagedRunTrackPool,
    resumeManagedRun,
    saveDraft,
    stopManagedRun,
    updateManagedLineageGroups,
    updateExistingDraft,
    watchManagedRun,
    setGlobalError,
  };
}

function mergeForkSourceCopyChoice(
  nextSource: DraftEditorSession["forkSource"],
  previousSource: DraftEditorSession["forkSource"] | undefined,
): DraftEditorSession["forkSource"] {
  if (nextSource === null) {
    return null;
  }
  if (
    previousSource !== undefined &&
    previousSource !== null &&
    previousSource.runId === nextSource.runId &&
    previousSource.artifact === nextSource.artifact
  ) {
    return {
      ...nextSource,
      copyAltBaselines: previousSource.copyAltBaselines,
      sourceEngineTunerBackend: previousSource.sourceEngineTunerBackend,
      sourceEngineTuning: previousSource.sourceEngineTuning,
      sourceEngineTuningKnown: previousSource.sourceEngineTuningKnown,
    };
  }
  return nextSource;
}

function forkSourceEngineTuning(config: ManagedRunConfig): ForkSourceEngineTuning | null {
  if (config.vehicle.engine_mode !== "adaptive_tuner") {
    return null;
  }
  return {
    backend: config.vehicle.adaptive_engine_tuner_backend,
    banditBucketRawValues:
      config.vehicle.adaptive_engine_tuner_backend === "bandit"
        ? config.vehicle.adaptive_engine_bandit_bucket_raw_values
        : null,
    objective:
      config.vehicle.adaptive_engine_tuner_backend === "bandit"
        ? config.vehicle.adaptive_engine_tuner_objective
        : null,
    rewardFingerprint:
      config.vehicle.adaptive_engine_tuner_backend === "bandit" ? stableJson(config.reward) : null,
    maxRawValue: config.vehicle.engine_setting_max_raw_value,
    minRawValue: config.vehicle.engine_setting_min_raw_value,
  };
}

function stableJson(value: unknown): string {
  return JSON.stringify(sortObjectKeys(value));
}

function sortObjectKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortObjectKeys);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nestedValue]) => [key, sortObjectKeys(nestedValue)]),
  );
}
