// src/rl_fzerox/apps/run_manager/web/src/app/workspace/types.ts
import type { ManagedDraft, ManagedRunConfig } from "@/shared/api/contract";

export type WorkspaceTabId = "drafts" | "runs" | "charts" | `editor:${string}` | `run:${string}`;

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

export interface WorkspaceTab {
  closable?: boolean;
  id: WorkspaceTabId;
  label: string;
  tone?: "draft" | "run";
}
