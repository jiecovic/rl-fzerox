import type { Dispatch, SetStateAction } from "react";
import { useMemo, useState } from "react";
import {
  buildWorkspaceTabs,
  defaultDraftName,
  draftForkSource,
  editorSessionId,
  nextAvailableDraftName,
  normalizeDraftTabTitle,
  runSessionId,
} from "@/app/workspace/model";
import type {
  DraftEditorSession,
  ForkSource,
  RunSession,
  WorkspaceTab,
  WorkspaceTabId,
} from "@/app/workspace/types";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";

interface UseWorkspaceSessionsOptions {
  drafts: ManagedDraft[];
  runs: ManagedRun[];
}

export interface WorkspaceSessions {
  activeDraftEditor: DraftEditorSession | null;
  activeRunTab: RunSession | null;
  activeTabId: WorkspaceTabId;
  chartsFocusRunId: string | null;
  closeEditorsForDraft: (draftId: string) => void;
  closeEditorsForSourceRuns: (runIds: readonly string[]) => void;
  closeRunTabsForRun: (runId: string) => void;
  closeRunTabsForRuns: (runIds: readonly string[]) => void;
  closeWorkspaceTab: (id: WorkspaceTabId) => void;
  createNewDraft: () => void;
  draftEditors: DraftEditorSession[];
  forkSourceRunLabel: (source: ForkSource | null) => string | null;
  openDraft: (draft: ManagedDraft) => void;
  openRun: (run: ManagedRun) => void;
  patchDraftEditor: (
    sessionId: DraftEditorSession["sessionId"],
    patch: Partial<Omit<DraftEditorSession, "sessionId">>,
  ) => void;
  reservedNamesForSession: (sessionId: DraftEditorSession["sessionId"]) => string[];
  runTabs: RunSession[];
  setActiveTabId: (id: WorkspaceTabId) => void;
  setChartsFocusRunId: Dispatch<SetStateAction<string | null>>;
  setDraftEditorTitle: (sessionId: DraftEditorSession["sessionId"], title: string) => void;
  showRunCharts: (runId: string) => void;
  workspaceTabs: WorkspaceTab[];
}

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

  function openDraft(draft: ManagedDraft) {
    const sessionId = editorSessionId(draft.id);
    setDraftEditors((current) =>
      current.some((session) => session.sessionId === sessionId)
        ? current
        : [
            ...current,
            {
              draftId: draft.id,
              forkSource: draftForkSource(draft),
              initialDraftName: draft.name,
              initialConfig: null,
              loadedDraft: draft,
              sessionId,
              title: draft.name,
            },
          ],
    );
    setActiveTabId(sessionId);
  }

  function openRun(run: ManagedRun) {
    const sessionId = runSessionId(run.id);
    setRunTabs((current) =>
      current.some((session) => session.sessionId === sessionId)
        ? current
        : [...current, { runId: run.id, sessionId, title: run.name }],
    );
    setActiveTabId(sessionId);
  }

  function showRunCharts(runId: string) {
    setChartsFocusRunId(runId);
    setActiveTabId("charts");
  }

  function createNewDraft() {
    const sessionId = editorSessionId(crypto.randomUUID());
    const initialDraftName = nextAvailableDraftName(defaultDraftName(), allKnownNames());
    setDraftEditors((current) => [
      ...current,
      {
        draftId: null,
        forkSource: null,
        initialDraftName,
        initialConfig: null,
        loadedDraft: null,
        sessionId,
        title: initialDraftName,
      },
    ]);
    setActiveTabId(sessionId);
  }

  function closeDraftEditor(sessionId: DraftEditorSession["sessionId"]) {
    const closingIndex = draftEditors.findIndex((session) => session.sessionId === sessionId);
    if (closingIndex === -1) {
      return;
    }
    const remaining = draftEditors.filter((session) => session.sessionId !== sessionId);
    setDraftEditors(remaining);
    if (activeTabId === sessionId) {
      const fallbackSession =
        remaining[closingIndex - 1] ?? remaining[closingIndex] ?? remaining.at(-1) ?? null;
      setActiveTabId(fallbackSession?.sessionId ?? "drafts");
    }
  }

  function closeRunTab(sessionId: RunSession["sessionId"]) {
    const closingIndex = runTabs.findIndex((session) => session.sessionId === sessionId);
    if (closingIndex === -1) {
      return;
    }
    const remaining = runTabs.filter((session) => session.sessionId !== sessionId);
    setRunTabs(remaining);
    if (activeTabId === sessionId) {
      const fallbackSession =
        remaining[closingIndex - 1] ?? remaining[closingIndex] ?? remaining.at(-1) ?? null;
      setActiveTabId(fallbackSession?.sessionId ?? "runs");
    }
  }

  function closeRunTabsForRun(runId: string) {
    closeRunTabsForRuns([runId]);
  }

  function closeRunTabsForRuns(runIds: readonly string[]) {
    if (runIds.length === 0) {
      return;
    }
    const runIdSet = new Set(runIds);
    const removedSessions = runTabs.filter((session) => runIdSet.has(session.runId));
    if (removedSessions.length === 0) {
      return;
    }
    const removedIds = new Set(removedSessions.map((session) => session.sessionId));
    const remaining = runTabs.filter((session) => !removedIds.has(session.sessionId));
    setRunTabs(remaining);
    if (removedIds.has(activeTabId as RunSession["sessionId"])) {
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
    patchDraftEditor(sessionId, { title: normalizeDraftTabTitle(title) });
  }

  function patchDraftEditor(
    sessionId: DraftEditorSession["sessionId"],
    patch: Partial<Omit<DraftEditorSession, "sessionId">>,
  ) {
    setDraftEditors((current) => {
      let changed = false;
      const next = current.map((session) => {
        if (session.sessionId !== sessionId) {
          return session;
        }
        const updated = { ...session, ...patch };
        changed =
          updated.title !== session.title ||
          updated.initialDraftName !== session.initialDraftName ||
          updated.initialConfig !== session.initialConfig ||
          updated.forkSource !== session.forkSource ||
          updated.draftId !== session.draftId ||
          updated.loadedDraft !== session.loadedDraft;
        return changed ? updated : session;
      });
      return changed ? next : current;
    });
  }

  function closeEditorsForDraft(id: string) {
    const removedSessions = draftEditors.filter((session) => session.draftId === id);
    if (removedSessions.length === 0) {
      return;
    }
    const removedSessionIds = new Set(removedSessions.map((session) => session.sessionId));
    const remaining = draftEditors.filter((session) => !removedSessionIds.has(session.sessionId));
    setDraftEditors(remaining);
    if (removedSessionIds.has(activeTabId as DraftEditorSession["sessionId"])) {
      setActiveTabId("drafts");
    }
  }

  function closeEditorsForSourceRuns(runIds: readonly string[]) {
    if (runIds.length === 0) {
      return;
    }
    const sourceRunIds = new Set(runIds);
    const removedSessionIds = new Set(
      draftEditors
        .filter((session) => {
          const sourceRunId =
            session.loadedDraft?.source_run_id ?? session.forkSource?.runId ?? null;
          return sourceRunId !== null && sourceRunIds.has(sourceRunId);
        })
        .map((session) => session.sessionId),
    );
    if (removedSessionIds.size === 0) {
      return;
    }
    setDraftEditors((current) =>
      current.filter((session) => !removedSessionIds.has(session.sessionId)),
    );
    if (removedSessionIds.has(activeTabId as DraftEditorSession["sessionId"])) {
      setActiveTabId("drafts");
    }
  }

  function forkSourceRunLabel(source: ForkSource | null) {
    if (source === null) {
      return null;
    }
    const run = runs.find((candidate) => candidate.id === source.runId) ?? null;
    return run?.name ?? source.runId;
  }

  function reservedNamesForSession(sessionId: DraftEditorSession["sessionId"]) {
    const names = new Set<string>();
    for (const draft of drafts) {
      names.add(draft.name);
    }
    for (const session of draftEditors) {
      if (session.sessionId !== sessionId) {
        names.add(session.title);
      }
    }
    return [...names];
  }

  function allKnownNames() {
    const names = new Set<string>();
    for (const draft of drafts) {
      names.add(draft.name);
    }
    for (const run of runs) {
      names.add(run.name);
    }
    for (const session of draftEditors) {
      names.add(session.title);
    }
    return names;
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
