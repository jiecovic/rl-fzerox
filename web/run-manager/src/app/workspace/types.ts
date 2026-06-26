// web/run-manager/src/app/workspace/types.ts
import type {
  EngineTunerBackend,
  EngineTunerObjective,
  ManagedDraft,
  ManagedRunConfig,
  PolicyPlaybackMode,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";

export type WorkspaceTabId =
  | "checkpoints"
  | "drafts"
  | "evaluations"
  | "runs"
  | "charts"
  | "save-games"
  | `editor:${string}`
  | `evaluation:${string}`
  | `run:${string}`
  | `save-game:${string}`;

export type PrimaryWorkspaceTabId = Extract<
  WorkspaceTabId,
  "checkpoints" | "drafts" | "evaluations" | "runs" | "charts" | "save-games"
>;

export interface ForkSource {
  artifact: "latest" | "best";
  copyAltBaselines: boolean;
  runId: string;
  sourceEngineTunerBackend: EngineTunerBackend | null;
  sourceEngineTuning: ForkSourceEngineTuning | null;
  sourceEngineTuningKnown: boolean;
}

export interface ForkSourceEngineTuning {
  backend: EngineTunerBackend;
  banditBucketRawValues: readonly number[] | null;
  objective: EngineTunerObjective | null;
  rewardFingerprint: string | null;
  maxRawValue: number;
  minRawValue: number;
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

export interface EvaluationSession {
  evaluationId: string;
  sessionId: `evaluation:${string}`;
  title: string;
}

export interface SaveGameSession {
  nameText: string;
  attemptSeedText: string;
  keepFailedPerfectRunVideos: boolean;
  policyMode: PolicyPlaybackMode;
  perfectRun: boolean;
  recordingEnabled: boolean;
  recordingInputHudEnabled: boolean;
  recordingUpscaleFactor: number;
  reloadPolicyBetweenAttempts: boolean;
  runnerDevice: WatchDevice;
  runnerRenderer: WatchRenderer;
  saveGameId: string | null;
  sessionId: `save-game:${string}`;
  targetClearGoalText: string;
  title: string;
}

export interface WorkspaceTab {
  activity?: "running";
  closable?: boolean;
  id: WorkspaceTabId;
  icon?: "career" | "charts" | "checkpoint" | "draft" | "evaluation" | "run";
  label: string;
  shortLabel?: string;
  tone?: "draft" | "run";
}
