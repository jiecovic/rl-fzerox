// src/rl_fzerox/apps/run_manager/web/src/app/workspace/sessions/types.ts
import type { Dispatch, SetStateAction } from "react";

import type {
  DraftEditorSession,
  ForkSource,
  RunSession,
  SaveGameSession,
  WorkspaceTab,
  WorkspaceTabId,
} from "@/app/workspace/types";
import type { ManagedDraft, ManagedRun, ManagedSaveGame } from "@/shared/api/contract";

export interface UseWorkspaceSessionsOptions {
  drafts: ManagedDraft[];
  runs: ManagedRun[];
  saveGames: ManagedSaveGame[];
}

export interface WorkspaceSessions {
  activeDraftEditor: DraftEditorSession | null;
  activeRunTab: RunSession | null;
  activeSaveGameSession: SaveGameSession | null;
  activeTabId: WorkspaceTabId;
  chartsFocusRunId: string | null;
  closeEditorsForDraft: (draftId: string) => void;
  closeEditorsForSourceRuns: (runIds: readonly string[]) => void;
  closeRunTabsForRun: (runId: string) => void;
  closeRunTabsForRuns: (runIds: readonly string[]) => void;
  closeWorkspaceTab: (id: WorkspaceTabId) => void;
  createNewDraft: () => void;
  createNewSaveGame: () => void;
  createForkDraft: (options: {
    artifact: ForkSource["artifact"];
    initialConfig: DraftEditorSession["initialConfig"];
    initialDraftName: string;
    runId: string;
  }) => void;
  draftEditors: DraftEditorSession[];
  forkSourceRunLabel: (source: ForkSource | null) => string | null;
  openDraft: (draft: ManagedDraft) => void;
  openRun: (run: ManagedRun) => void;
  openSaveGame: (saveGame: ManagedSaveGame) => void;
  patchDraftEditor: (
    sessionId: DraftEditorSession["sessionId"],
    patch: Partial<Omit<DraftEditorSession, "sessionId">>,
  ) => void;
  reservedNamesForSession: (sessionId: DraftEditorSession["sessionId"]) => string[];
  runTabs: RunSession[];
  saveGameSessions: SaveGameSession[];
  setActiveTabId: (id: WorkspaceTabId) => void;
  setChartsFocusRunId: Dispatch<SetStateAction<string | null>>;
  setDraftEditorTitle: (sessionId: DraftEditorSession["sessionId"], title: string) => void;
  patchSaveGameSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  showRunCharts: (runId: string) => void;
  workspaceTabs: WorkspaceTab[];
}
