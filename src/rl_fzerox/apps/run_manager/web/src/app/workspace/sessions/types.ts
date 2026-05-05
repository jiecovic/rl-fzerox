import type { Dispatch, SetStateAction } from "react";

import type {
  DraftEditorSession,
  ForkSource,
  RunSession,
  WorkspaceTab,
  WorkspaceTabId,
} from "@/app/workspace/types";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";

export interface UseWorkspaceSessionsOptions {
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
