// web/run-manager/src/app/workspace/sessions/hook.ts
import { useMemo, useState } from "react";

import {
  activePrimaryWorkspaceTabId,
  buildWorkspaceSessionTabs,
  defaultDraftName,
  editorSessionId,
  evaluationSessionId,
  nextAvailableDraftName,
  nextAvailableSaveGameName,
  runSessionId,
  saveGameSessionId,
} from "@/app/workspace/model";
import {
  closeDraftSession,
  closeDraftSessionsForDraft,
  closeDraftSessionsForSourceRuns,
  createDraftSession,
  normalizeDraftSessionTitle,
  openDraftSession,
  patchDraftSessions,
} from "@/app/workspace/sessions/drafts";
import {
  closeEvaluationSession,
  closeEvaluationSessionsForEvaluation,
  openEvaluationSession,
  renameEvaluationSession,
} from "@/app/workspace/sessions/evaluations";
import {
  allKnownWorkspaceNames,
  reservedWorkspaceNamesForSession,
  resolveForkSourceRunLabel,
} from "@/app/workspace/sessions/names";
import {
  closeRunSession,
  closeRunSessionsForRuns,
  openRunSession,
} from "@/app/workspace/sessions/runs";
import type {
  UseWorkspaceSessionsOptions,
  WorkspaceSessions,
} from "@/app/workspace/sessions/types";
import type {
  DraftEditorSession,
  EvaluationSession,
  ForkSource,
  RunSession,
  SaveGameSession,
  WorkspaceTabId,
} from "@/app/workspace/types";
import { randomAttemptSeedText } from "@/features/careerRunner/model/runnerSeed";

export function useWorkspaceSessions({
  checkpointRunIds,
  drafts,
  evaluations,
  runs,
  saveGames,
}: UseWorkspaceSessionsOptions): WorkspaceSessions {
  const [activeTabId, setActiveTabId] = useState<WorkspaceTabId>("drafts");
  const [draftEditors, setDraftEditors] = useState<DraftEditorSession[]>([]);
  const [evaluationSessions, setEvaluationSessions] = useState<EvaluationSession[]>([]);
  const [runTabs, setRunTabs] = useState<RunSession[]>([]);
  const [saveGameSessions, setSaveGameSessions] = useState<SaveGameSession[]>([]);
  const [chartsFocusRunId, setChartsFocusRunId] = useState<string | null>(null);

  const activeDraftEditor = useMemo(
    () =>
      activeTabId.startsWith("editor:")
        ? (draftEditors.find((session) => session.sessionId === activeTabId) ?? null)
        : null,
    [activeTabId, draftEditors],
  );
  const activeRunTab = useMemo(
    () =>
      activeTabId.startsWith("run:")
        ? (runTabs.find((session) => session.sessionId === activeTabId) ?? null)
        : null,
    [activeTabId, runTabs],
  );
  const activeEvaluationSession = useMemo(
    () =>
      activeTabId.startsWith("evaluation:")
        ? (evaluationSessions.find((session) => session.sessionId === activeTabId) ?? null)
        : null,
    [activeTabId, evaluationSessions],
  );
  const activeSaveGameSession = useMemo(
    () =>
      activeTabId.startsWith("save-game:")
        ? (saveGameSessions.find((session) => session.sessionId === activeTabId) ?? null)
        : null,
    [activeTabId, saveGameSessions],
  );
  const checkpointRunIdSet = useMemo(() => new Set(checkpointRunIds), [checkpointRunIds]);
  const activePrimaryTabId =
    activeRunTab !== null && checkpointRunIdSet.has(activeRunTab.runId)
      ? "checkpoints"
      : activePrimaryWorkspaceTabId(activeTabId);
  const sessionTabs = useMemo(
    () =>
      buildWorkspaceSessionTabs(
        checkpointRunIdSet,
        draftEditors,
        evaluationSessions,
        evaluations,
        runTabs,
        runs,
        saveGameSessions,
      ),
    [
      checkpointRunIdSet,
      draftEditors,
      evaluationSessions,
      evaluations,
      runTabs,
      runs,
      saveGameSessions,
    ],
  );

  function openDraft(draft: UseWorkspaceSessionsOptions["drafts"][number]) {
    const sessionId = editorSessionId(draft.id);
    setDraftEditors((current) => openDraftSession(current, draft));
    setActiveTabId(sessionId);
  }

  function openRun(run: UseWorkspaceSessionsOptions["runs"][number]) {
    const sessionId = runSessionId(run.id);
    setRunTabs((current) => openRunSession(current, run));
    setActiveTabId(sessionId);
  }

  function openEvaluation(evaluation: UseWorkspaceSessionsOptions["evaluations"][number]) {
    const sessionId = evaluationSessionId(evaluation.id);
    setEvaluationSessions((current) => openEvaluationSession(current, evaluation));
    setActiveTabId(sessionId);
  }

  function showRunCharts(runId: string) {
    setChartsFocusRunId(runId);
    setActiveTabId("charts");
  }

  function createNewDraft() {
    const sessionId = editorSessionId(crypto.randomUUID());
    const initialDraftName = nextAvailableDraftName(
      defaultDraftName(),
      allKnownWorkspaceNames({ drafts, draftEditors, runs }),
    );
    setDraftEditors((current) =>
      createDraftSession(current, {
        initialDraftName,
        sessionId,
      }),
    );
    setActiveTabId(sessionId);
  }

  function createNewSaveGame() {
    const sessionId = saveGameSessionId(crypto.randomUUID());
    const nameText = nextAvailableSaveGameName(saveGames.map((saveGame) => saveGame.name));
    setSaveGameSessions((current) =>
      upsertSaveGameSession(current, {
        nameText,
        attemptSeedText: randomAttemptSeedText(),
        keepFailedPerfectRunVideos: false,
        policyMode: "deterministic",
        perfectRun: false,
        recordingEnabled: false,
        recordingInputHudEnabled: false,
        recordingUpscaleFactor: 2,
        reloadPolicyBetweenAttempts: true,
        runnerDevice: "cuda",
        runnerRenderer: "gliden64",
        saveGameId: null,
        sessionId,
        targetClearGoalText: "1",
        title: nameText,
      }),
    );
    setActiveTabId(sessionId);
  }

  function openSaveGame(saveGame: UseWorkspaceSessionsOptions["saveGames"][number]) {
    const sessionId = saveGameSessionId(saveGame.id);
    setSaveGameSessions((current) =>
      upsertSaveGameSession(current, saveGameSessionForManagedSave(current, saveGame, sessionId)),
    );
    setActiveTabId(sessionId);
  }

  function createForkDraft({
    artifact,
    copyAltBaselines,
    initialConfig,
    initialDraftName,
    runId,
    sourceEngineTunerBackend,
    sourceEngineTuning,
    sourceEngineTuningKnown,
  }: {
    artifact: ForkSource["artifact"];
    copyAltBaselines: boolean;
    initialConfig: DraftEditorSession["initialConfig"];
    initialDraftName: string;
    runId: string;
    sourceEngineTunerBackend: ForkSource["sourceEngineTunerBackend"];
    sourceEngineTuning: ForkSource["sourceEngineTuning"];
    sourceEngineTuningKnown: ForkSource["sourceEngineTuningKnown"];
  }) {
    const sessionId = editorSessionId(crypto.randomUUID());
    setDraftEditors((current) =>
      createDraftSession(current, {
        forkSource: {
          artifact,
          copyAltBaselines,
          runId,
          sourceEngineTunerBackend,
          sourceEngineTuning,
          sourceEngineTuningKnown,
        },
        initialConfig,
        initialDraftName,
        sessionId,
      }),
    );
    setActiveTabId(sessionId);
  }

  function closeDraftEditor(sessionId: DraftEditorSession["sessionId"]) {
    const nextState = closeDraftSession(draftEditors, sessionId);
    if (nextState === null) {
      return;
    }
    setDraftEditors(nextState.remaining);
    if (activeTabId === sessionId) {
      setActiveTabId(nextState.fallbackTabId);
    }
  }

  function closeRunTab(sessionId: RunSession["sessionId"]) {
    const nextState = closeRunSession(runTabs, sessionId);
    if (nextState === null) {
      return;
    }
    setRunTabs(nextState.remaining);
    if (activeTabId === sessionId) {
      setActiveTabId(nextState.fallbackTabId);
    }
  }

  function closeEvaluationTab(sessionId: EvaluationSession["sessionId"]) {
    const nextState = closeEvaluationSession(evaluationSessions, sessionId);
    if (nextState === null) {
      return;
    }
    setEvaluationSessions(nextState.remaining);
    if (activeTabId === sessionId) {
      setActiveTabId(nextState.fallbackTabId);
    }
  }

  function closeSaveGameTab(sessionId: SaveGameSession["sessionId"]) {
    const remaining = saveGameSessions.filter((session) => session.sessionId !== sessionId);
    if (remaining.length === saveGameSessions.length) {
      return;
    }
    setSaveGameSessions(remaining);
    if (activeTabId === sessionId) {
      setActiveTabId("save-games");
    }
  }

  function closeRunTabsForRun(runId: string) {
    closeRunTabsForRuns([runId]);
  }

  function closeRunTabsForRuns(runIds: readonly string[]) {
    const nextState = closeRunSessionsForRuns(runTabs, runIds);
    if (nextState === null) {
      return;
    }
    setRunTabs(nextState.remaining);
    if (nextState.removedSessionIds.has(activeTabId as RunSession["sessionId"])) {
      setActiveTabId("runs");
    }
  }

  function closeWorkspaceTab(id: WorkspaceTabId) {
    if (id === "charts") {
      setActiveTabId("runs");
      return;
    }
    if (id.startsWith("editor:")) {
      closeDraftEditor(id as DraftEditorSession["sessionId"]);
      return;
    }
    if (id.startsWith("run:")) {
      closeRunTab(id as RunSession["sessionId"]);
      return;
    }
    if (id.startsWith("evaluation:")) {
      closeEvaluationTab(id as EvaluationSession["sessionId"]);
      return;
    }
    if (id.startsWith("save-game:")) {
      closeSaveGameTab(id as SaveGameSession["sessionId"]);
    }
  }

  function setDraftEditorTitle(sessionId: DraftEditorSession["sessionId"], title: string) {
    patchDraftEditor(sessionId, { title: normalizeDraftSessionTitle(title) });
  }

  function patchDraftEditor(
    sessionId: DraftEditorSession["sessionId"],
    patch: Partial<Omit<DraftEditorSession, "sessionId">>,
  ) {
    setDraftEditors((current) => patchDraftSessions(current, sessionId, patch));
  }

  function patchSaveGameSession(
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) {
    setSaveGameSessions((current) =>
      current.map((session) =>
        session.sessionId === sessionId ? { ...session, ...patch } : session,
      ),
    );
  }

  function closeEditorsForDraft(id: string) {
    const nextState = closeDraftSessionsForDraft(draftEditors, id);
    if (nextState === null) {
      return;
    }
    setDraftEditors(nextState.remaining);
    if (nextState.removedSessionIds.has(activeTabId as DraftEditorSession["sessionId"])) {
      setActiveTabId("drafts");
    }
  }

  function closeEditorsForSourceRuns(runIds: readonly string[]) {
    const nextState = closeDraftSessionsForSourceRuns(draftEditors, runIds);
    if (nextState === null) {
      return;
    }
    setDraftEditors(nextState.remaining);
    if (nextState.removedSessionIds.has(activeTabId as DraftEditorSession["sessionId"])) {
      setActiveTabId("drafts");
    }
  }

  function closeEvaluationTabsForEvaluation(evaluationId: string) {
    const nextState = closeEvaluationSessionsForEvaluation(evaluationSessions, evaluationId);
    if (nextState === null) {
      return;
    }
    setEvaluationSessions(nextState.remaining);
    if (nextState.removedSessionIds.has(activeTabId as EvaluationSession["sessionId"])) {
      setActiveTabId("evaluations");
    }
  }

  function renameEvaluationTab(evaluationId: string, title: string) {
    setEvaluationSessions((current) => renameEvaluationSession(current, evaluationId, title));
  }

  function forkSourceRunLabel(source: DraftEditorSession["forkSource"]) {
    return resolveForkSourceRunLabel(source, runs);
  }

  function reservedNamesForSession(sessionId: DraftEditorSession["sessionId"]) {
    return reservedWorkspaceNamesForSession({
      drafts,
      draftEditors,
      sessionId,
    });
  }

  return {
    activeDraftEditor,
    activeEvaluationSession,
    activePrimaryTabId,
    activeRunTab,
    activeSaveGameSession,
    activeTabId,
    chartsFocusRunId,
    closeEditorsForDraft,
    closeEditorsForSourceRuns,
    closeEvaluationTabsForEvaluation,
    closeRunTabsForRun,
    closeRunTabsForRuns,
    closeWorkspaceTab,
    createForkDraft,
    createNewDraft,
    createNewSaveGame,
    draftEditors,
    evaluationSessions,
    forkSourceRunLabel,
    openDraft,
    openEvaluation,
    openRun,
    openSaveGame,
    patchDraftEditor,
    patchSaveGameSession,
    renameEvaluationTab,
    reservedNamesForSession,
    runTabs,
    saveGameSessions,
    setActiveTabId,
    setChartsFocusRunId,
    setDraftEditorTitle,
    showRunCharts,
    sessionTabs,
  };
}

function upsertSaveGameSession(
  current: readonly SaveGameSession[],
  next: SaveGameSession,
): SaveGameSession[] {
  const existingIndex = current.findIndex((session) => session.sessionId === next.sessionId);
  if (existingIndex < 0) {
    return [...current, next];
  }
  return current.map((session, index) => (index === existingIndex ? next : session));
}

function saveGameSessionForManagedSave(
  current: readonly SaveGameSession[],
  saveGame: UseWorkspaceSessionsOptions["saveGames"][number],
  sessionId: SaveGameSession["sessionId"],
): SaveGameSession {
  const existing = current.find((session) => session.sessionId === sessionId);
  const settings = saveGame.runner_settings;
  return {
    attemptSeedText:
      existing?.attemptSeedText ??
      (settings.attempt_seed === null ? "" : String(settings.attempt_seed)),
    keepFailedPerfectRunVideos:
      existing?.keepFailedPerfectRunVideos ?? settings.keep_failed_recordings,
    nameText: saveGame.name,
    policyMode: existing?.policyMode ?? settings.policy_mode,
    perfectRun: existing?.perfectRun ?? settings.target_restart_on_retire,
    recordingEnabled: existing?.recordingEnabled ?? settings.recording_enabled,
    recordingInputHudEnabled:
      existing?.recordingInputHudEnabled ?? settings.recording_input_hud_enabled,
    recordingUpscaleFactor: existing?.recordingUpscaleFactor ?? settings.recording_upscale_factor,
    reloadPolicyBetweenAttempts:
      existing?.reloadPolicyBetweenAttempts ?? settings.reload_policy_between_attempts,
    runnerDevice: existing?.runnerDevice ?? settings.device,
    runnerRenderer: existing?.runnerRenderer ?? settings.renderer,
    saveGameId: saveGame.id,
    sessionId,
    targetClearGoalText: existing?.targetClearGoalText ?? String(settings.target_clear_goal),
    title: saveGame.name,
  };
}
