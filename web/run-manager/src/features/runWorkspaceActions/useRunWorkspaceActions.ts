// web/run-manager/src/features/runWorkspaceActions/useRunWorkspaceActions.ts
import { useEffect, useState } from "react";
import type {
  ManagedRunDetail,
  PolicyPlaybackMode,
  TrackSamplingRuntimeState,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";

export type CheckpointArtifact = "latest" | "best";
export type WatchToastTone = "info" | "error";

export interface WatchToastState {
  message: string;
  tone: WatchToastTone;
}

export interface RunWorkspaceActionsProps {
  clearTrackSamplingState: (state: TrackSamplingRuntimeState | null) => void;
  onCreateDraftFromRun: (runId: string) => Promise<void>;
  onFork: (runId: string, artifact: CheckpointArtifact) => Promise<void>;
  onOpenDirectory: (runId: string) => Promise<void>;
  onRename: (runId: string, name: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  onResetTrackPool: (runId: string) => Promise<void>;
  onStop: (runId: string) => Promise<void>;
  onWatch: (
    runId: string,
    artifact: CheckpointArtifact,
    device: WatchDevice,
    renderer: WatchRenderer,
    policyMode: PolicyPlaybackMode,
  ) => Promise<"started" | "already_running">;
  run: ManagedRunDetail;
  runName: string;
}

export interface RunWorkspaceActionState {
  canRename: boolean;
  canResume: boolean;
  canStop: boolean;
  controlError: string | null;
  copiedRunId: boolean;
  copyRunId: () => Promise<void>;
  createDraftFromRun: () => Promise<void>;
  forkRunArtifact: (artifact: CheckpointArtifact) => Promise<void>;
  isCreatingDraftFromRun: boolean;
  isForking: boolean;
  isOpeningDirectory: boolean;
  isRenaming: boolean;
  isResettingTrackPool: boolean;
  isResuming: boolean;
  isStopping: boolean;
  openRunDirectoryInBrowser: () => Promise<void>;
  renameRunLabel: (name?: string) => Promise<boolean>;
  resetTrackPoolState: () => Promise<void>;
  resumeRun: () => Promise<void>;
  selectedForkArtifact: CheckpointArtifact;
  setSelectedForkArtifact: (artifact: CheckpointArtifact) => void;
  selectedWatchArtifact: CheckpointArtifact;
  setSelectedWatchArtifact: (artifact: CheckpointArtifact) => void;
  selectedWatchDevice: WatchDevice;
  setSelectedWatchDevice: (device: WatchDevice) => void;
  selectedWatchPolicyMode: PolicyPlaybackMode;
  setSelectedWatchPolicyMode: (mode: PolicyPlaybackMode) => void;
  selectedWatchRenderer: WatchRenderer;
  setSelectedWatchRenderer: (renderer: WatchRenderer) => void;
  stopRun: () => Promise<void>;
  watchToast: WatchToastState | null;
  watchRunArtifact: (artifact: CheckpointArtifact) => Promise<void>;
  watchingArtifact: CheckpointArtifact | null;
}

export function useRunWorkspaceActions({
  clearTrackSamplingState,
  onCreateDraftFromRun,
  onFork,
  onOpenDirectory,
  onRename,
  onResume,
  onResetTrackPool,
  onStop,
  onWatch,
  run,
  runName,
}: RunWorkspaceActionsProps): RunWorkspaceActionState {
  const [controlError, setControlError] = useState<string | null>(null);
  const [watchToast, setWatchToast] = useState<WatchToastState | null>(null);
  const [copiedRunId, setCopiedRunId] = useState(false);
  const [isOpeningDirectory, setIsOpeningDirectory] = useState(false);
  const [isCreatingDraftFromRun, setIsCreatingDraftFromRun] = useState(false);
  const [isForking, setIsForking] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [selectedForkArtifact, setSelectedForkArtifact] = useState<CheckpointArtifact>("latest");
  const [selectedWatchArtifact, setSelectedWatchArtifact] = useState<CheckpointArtifact>("latest");
  const [selectedWatchDevice, setSelectedWatchDevice] = useState<WatchDevice>("cuda");
  const [selectedWatchPolicyMode, setSelectedWatchPolicyMode] =
    useState<PolicyPlaybackMode>("deterministic");
  const [watchRendererSelection, setWatchRendererSelection] = useState<{
    renderer: WatchRenderer;
    runId: string;
  }>({
    renderer: run.config.environment.renderer,
    runId: run.id,
  });
  const [watchingArtifact, setWatchingArtifact] = useState<CheckpointArtifact | null>(null);
  const [isResettingTrackPool, setIsResettingTrackPool] = useState(false);
  const selectedWatchRenderer =
    watchRendererSelection.runId === run.id
      ? watchRendererSelection.renderer
      : run.config.environment.renderer;
  const setSelectedWatchRenderer = (renderer: WatchRenderer) => {
    setWatchRendererSelection({ renderer, runId: run.id });
  };

  useEffect(() => {
    if (!copiedRunId) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => {
      setCopiedRunId(false);
    }, 1_200);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [copiedRunId]);

  useEffect(() => {
    if (watchToast === null) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => {
      setWatchToast(null);
    }, 3_200);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [watchToast]);

  async function resumeRun() {
    setIsResuming(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onResume(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to resume run");
    } finally {
      setIsResuming(false);
    }
  }

  async function stopRun() {
    setIsStopping(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onStop(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to stop run");
    } finally {
      setIsStopping(false);
    }
  }

  async function renameRunLabel(name?: string) {
    const nextName = (name ?? runName).trim();
    if (nextName.length === 0) {
      setControlError("run name is required");
      return false;
    }
    setIsRenaming(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onRename(run.id, nextName);
      return true;
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to rename run");
      return false;
    } finally {
      setIsRenaming(false);
    }
  }

  async function openRunDirectoryInBrowser() {
    setIsOpeningDirectory(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onOpenDirectory(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to open run directory");
    } finally {
      setIsOpeningDirectory(false);
    }
  }

  async function forkRunArtifact(artifact: CheckpointArtifact) {
    setIsForking(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onFork(run.id, artifact);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : `failed to fork ${artifact}`);
    } finally {
      setIsForking(false);
    }
  }

  async function createDraftFromRun() {
    setIsCreatingDraftFromRun(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onCreateDraftFromRun(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to create draft from run");
    } finally {
      setIsCreatingDraftFromRun(false);
    }
  }

  async function watchRunArtifact(artifact: CheckpointArtifact) {
    setWatchingArtifact(artifact);
    setControlError(null);
    setWatchToast(null);
    try {
      const status = await onWatch(
        run.id,
        artifact,
        selectedWatchDevice,
        selectedWatchRenderer,
        selectedWatchPolicyMode,
      );
      if (status === "already_running") {
        setWatchToast({
          message: `${artifact} watch is already running`,
          tone: "info",
        });
      }
    } catch (caught) {
      setWatchToast({
        message: caught instanceof Error ? caught.message : `failed to watch ${artifact}`,
        tone: "error",
      });
    } finally {
      setWatchingArtifact((current) => (current === artifact ? null : current));
    }
  }

  async function resetTrackPoolState() {
    setIsResettingTrackPool(true);
    setControlError(null);
    setWatchToast(null);
    try {
      await onResetTrackPool(run.id);
      clearTrackSamplingState(null);
    } catch (caught) {
      setControlError(
        caught instanceof Error ? caught.message : "failed to reset track-pool stats",
      );
    } finally {
      setIsResettingTrackPool(false);
    }
  }

  async function copyRunId() {
    try {
      await navigator.clipboard.writeText(run.id);
      setCopiedRunId(true);
      setControlError(null);
      setWatchToast(null);
    } catch {
      setControlError("failed to copy run id");
    }
  }

  const pendingCommand = run.pending_command;
  const normalizedRunName = runName.trim();
  const canRename =
    normalizedRunName.length > 0 &&
    normalizedRunName !== run.name &&
    !isRenaming &&
    !isOpeningDirectory;
  const canStop = run.status === "running" && pendingCommand === null && !isResuming;
  const canResume =
    (run.status === "paused" || run.status === "stopped" || run.status === "failed") && !isStopping;

  return {
    canRename,
    canResume,
    canStop,
    controlError,
    copiedRunId,
    copyRunId,
    createDraftFromRun,
    forkRunArtifact,
    isCreatingDraftFromRun,
    isForking,
    isOpeningDirectory,
    isRenaming,
    isResettingTrackPool,
    isResuming,
    isStopping,
    openRunDirectoryInBrowser,
    renameRunLabel,
    resetTrackPoolState,
    resumeRun,
    selectedForkArtifact,
    setSelectedForkArtifact,
    selectedWatchArtifact,
    setSelectedWatchArtifact,
    selectedWatchDevice,
    setSelectedWatchDevice,
    selectedWatchPolicyMode,
    setSelectedWatchPolicyMode,
    selectedWatchRenderer,
    setSelectedWatchRenderer,
    stopRun,
    watchToast,
    watchRunArtifact,
    watchingArtifact,
  };
}
