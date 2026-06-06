// src/rl_fzerox/apps/run_manager/web/src/app/workspace/types.ts
import type {
  ManagedDraft,
  ManagedRunConfig,
  PolicyPlaybackMode,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";

export type WorkspaceTabId =
  | "drafts"
  | "runs"
  | "charts"
  | "save-games"
  | `editor:${string}`
  | `run:${string}`
  | `save-game:${string}`;

export interface ForkSource {
  artifact: "latest" | "best";
  runId: string;
}

export interface DraftEditorSession {
  currentConfig: ManagedRunConfig | null;
  currentDraftName: string;
  draftId: string | null;
  forkSource: ForkSource | null;
  initialDraftName: string;
  initialConfig: ManagedRunConfig | null;
  loadedDraft: ManagedDraft | null;
  sessionId: `editor:${string}`;
  title: string;
}

export interface RunSession {
  runId: string;
  sessionId: `run:${string}`;
  title: string;
}

export interface SaveGameSession {
  nameText: string;
  attemptSeedText: string;
  policyMode: PolicyPlaybackMode;
  runnerDevice: WatchDevice;
  runnerRenderer: WatchRenderer;
  saveGameId: string | null;
  sessionId: `save-game:${string}`;
  title: string;
}

export interface WorkspaceTab {
  closable?: boolean;
  id: WorkspaceTabId;
  icon?: "career" | "charts" | "draft" | "run";
  label: string;
  tone?: "draft" | "run";
}
