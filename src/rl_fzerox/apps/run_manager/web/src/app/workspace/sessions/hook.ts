// src/rl_fzerox/apps/run_manager/web/src/app/workspace/sessions/hook.ts
import { useMemo, useState } from "react";

import {
  buildWorkspaceTabs,
  defaultDraftName,
  editorSessionId,
  nextAvailableDraftName,
  runSessionId,
} from "@/app/workspace/model";
import type { DraftEditorSession, RunSession, WorkspaceTabId } from "@/app/workspace/types";

import {
  closeDraftSession,
  closeDraftSessionsForDraft,
  closeDraftSessionsForSourceRuns,
  createDraftSession,
  normalizeDraftSessionTitle,
  openDraftSession,
  patchDraftSessions,
} from "./drafts";
import {
  allKnownWorkspaceNames,
  reservedWorkspaceNamesForSession,
  resolveForkSourceRunLabel,
} from "./names";
import { closeRunSession, closeRunSessionsForRuns, openRunSession } from "./runs";
import type { UseWorkspaceSessionsOptions, WorkspaceSessions } from "./types";

export function useWorkspaceSessions({
  drafts,
  runs,
}: UseWorkspaceSessionsOptions): WorkspaceSessions {
  const [activeTabId, setActiveTabId] = useState<WorkspaceTabId>("drafts");
  const [draftEditors, setDraftEditors] = useState<DraftEditorSession[]>([]);
  const [runTabs, setRunTabs] = useState<RunSession[]>([]);
  const [chartsFocusRunId, setChartsFocusRunId] = useState<string | null>(null);

  const activeDraftEditor = useMemo(
    () =>
      activeTabId === "drafts" || activeTabId === "runs" || activeTabId === "charts"
        ? null
        : (draftEditors.find((session) => session.sessionId === activeTabId) ?? null),
    [activeTabId, draftEditors],
  );
  const activeRunTab = useMemo(
    () =>
      activeTabId === "drafts" || activeTabId === "runs" || activeTabId === "charts"
        ? null
        : (runTabs.find((session) => session.sessionId === activeTabId) ?? null),
    [activeTabId, runTabs],
  );
  const workspaceTabs = useMemo(
    () => buildWorkspaceTabs(draftEditors, runTabs, runs),
    [draftEditors, runTabs, runs],
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
    activeRunTab,
    activeTabId,
    chartsFocusRunId,
    closeEditorsForDraft,
    closeEditorsForSourceRuns,
    closeRunTabsForRun,
    closeRunTabsForRuns,
    closeWorkspaceTab,
    createNewDraft,
    draftEditors,
    forkSourceRunLabel,
    openDraft,
    openRun,
    patchDraftEditor,
    reservedNamesForSession,
    runTabs,
    setActiveTabId,
    setChartsFocusRunId,
    setDraftEditorTitle,
    showRunCharts,
    workspaceTabs,
  };
}
