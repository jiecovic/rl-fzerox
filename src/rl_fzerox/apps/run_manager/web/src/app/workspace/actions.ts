import type { Dispatch, SetStateAction } from "react";
import {
  draftForkSource,
  nextAvailableDraftName,
  upsertDraft,
  upsertRun,
} from "@/app/workspace/model";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import type { DraftEditorSession } from "@/app/workspace/types";
import {
  createDraftWithSource,
  deleteDraft,
  deleteLineage,
  deleteRun,
  launchRun,
  openRunDirectory,
  renameRun,
  resetRunTrackSamplingState,
  resumeRun,
  stopRun,
  updateDraftWithSource,
  watchRun,
} from "@/shared/api/client";
import type { ManagedDraft, ManagedRun, ManagedRunConfig } from "@/shared/api/contract";

interface UseWorkspaceActionsOptions {
  defaultConfig: ManagedRunConfig | null;
  drafts: ManagedDraft[];
  reloadManagerData: () => Promise<void>;
  runs: ManagedRun[];
  sessions: WorkspaceSessions;
  setDrafts: Dispatch<SetStateAction<ManagedDraft[]>>;
  setRuns: Dispatch<SetStateAction<ManagedRun[]>>;
}

export interface WorkspaceActions {
  createDraftFromManagedRun: (runId: string) => Promise<void>;
  forkManagedRun: (runId: string, artifact: "latest" | "best") => Promise<void>;
  launchTrainingRun: (
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
    draftId: string | null,
  ) => Promise<ManagedRun>;
  openManagedRunDirectory: (runId: string) => Promise<void>;
  removeDraft: (id: string) => Promise<void>;
  removeLineage: (lineageId: string) => Promise<void>;
  removeRun: (run: ManagedRun) => Promise<void>;
  renameManagedRun: (runId: string, name: string) => Promise<void>;
  resetManagedRunTrackPool: (runId: string) => Promise<void>;
  resumeManagedRun: (runId: string) => Promise<void>;
  saveDraft: (
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
  ) => Promise<ManagedDraft>;
  stopManagedRun: (runId: string) => Promise<void>;
  updateExistingDraft: (
    sessionId: DraftEditorSession["sessionId"],
    id: string,
    name: string,
    config: ManagedRunConfig,
  ) => Promise<ManagedDraft>;
  watchManagedRun: (runId: string, artifact: "latest" | "best") => Promise<void>;
}

export function useWorkspaceActions({
  defaultConfig,
  drafts: _drafts,
  reloadManagerData,
  runs,
  sessions,
  setDrafts,
  setRuns,
}: UseWorkspaceActionsOptions): WorkspaceActions {
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
      draftId: draft.id,
      forkSource: draftForkSource(draft),
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
      draftId: draft.id,
      forkSource: draftForkSource(draft),
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
  ) {
    const session =
      sessions.draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const run = await launchRun(
      name,
      config,
      draftId,
      session?.forkSource?.runId ?? null,
      session?.forkSource?.artifact ?? null,
    );
    setRuns((current) => upsertRun(current, run));
    sessions.openRun(run);
    return run;
  }

  async function forkManagedRun(runId: string, artifact: "latest" | "best") {
    const sourceRun = runs.find((candidate) => candidate.id === runId) ?? null;
    const sourceConfig = sourceRun?.config ?? defaultConfig;
    if (sourceConfig === null) {
      throw new Error("fork source config is unavailable");
    }
    const baseName =
      sourceRun === null
        ? artifact === "best"
          ? "fork best"
          : "fork"
        : artifact === "best"
          ? `${sourceRun.name} best fork`
          : `${sourceRun.name} fork`;
    const initialDraftName = nextAvailableDraftName(baseName, allKnownNames());
    const draft = await createDraftWithSource(initialDraftName, sourceConfig, runId, artifact);
    setDrafts((current) => upsertDraft(current, draft));
    sessions.openDraft(draft);
  }

  async function createDraftFromManagedRun(runId: string) {
    const sourceRun = runs.find((candidate) => candidate.id === runId) ?? null;
    if (sourceRun === null) {
      throw new Error("run config is unavailable");
    }
    const initialDraftName = nextAvailableDraftName(`${sourceRun.name} draft`, allKnownNames());
    const draft = await createDraftWithSource(initialDraftName, sourceRun.config, null, null);
    setDrafts((current) => upsertDraft(current, draft));
    sessions.openDraft(draft);
  }

  async function stopManagedRun(runId: string) {
    const run = await stopRun(runId);
    setRuns((current) => upsertRun(current, run));
  }

  async function resumeManagedRun(runId: string) {
    const run = await resumeRun(runId);
    setRuns((current) => upsertRun(current, run));
  }

  async function renameManagedRun(runId: string, name: string) {
    const run = await renameRun(runId, name);
    setRuns((current) => upsertRun(current, run));
  }

  async function openManagedRunDirectory(runId: string) {
    await openRunDirectory(runId);
  }

  async function watchManagedRun(runId: string, artifact: "latest" | "best") {
    await watchRun(runId, artifact);
  }

  async function resetManagedRunTrackPool(runId: string) {
    await resetRunTrackSamplingState(runId);
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
    createDraftFromManagedRun,
    forkManagedRun,
    launchTrainingRun,
    openManagedRunDirectory,
    removeDraft,
    removeLineage,
    removeRun,
    renameManagedRun,
    resetManagedRunTrackPool,
    resumeManagedRun,
    saveDraft,
    stopManagedRun,
    updateExistingDraft,
    watchManagedRun,
  };
}
