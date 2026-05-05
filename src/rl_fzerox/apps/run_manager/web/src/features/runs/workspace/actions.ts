import { useEffect, useState } from "react";
import type { ManagedRun, TrackSamplingRuntimeState } from "@/shared/api/contract";

export type CheckpointArtifact = "latest" | "best";

export interface RunWorkspaceActionsProps {
  clearTrackSamplingState: (state: TrackSamplingRuntimeState | null) => void;
  onCreateDraftFromRun: (runId: string) => Promise<void>;
  onFork: (runId: string, artifact: CheckpointArtifact) => Promise<void>;
  onOpenDirectory: (runId: string) => Promise<void>;
  onRename: (runId: string, name: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  onResetTrackPool: (runId: string) => Promise<void>;
  onStop: (runId: string) => Promise<void>;
  onWatch: (runId: string, artifact: CheckpointArtifact) => Promise<void>;
  run: ManagedRun;
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
  renameRunLabel: () => Promise<void>;
  resetTrackPoolState: () => Promise<void>;
  resumeRun: () => Promise<void>;
  selectedArtifact: CheckpointArtifact;
  setSelectedArtifact: (artifact: CheckpointArtifact) => void;
  stopRun: () => Promise<void>;
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
  const [copiedRunId, setCopiedRunId] = useState(false);
  const [isOpeningDirectory, setIsOpeningDirectory] = useState(false);
  const [isCreatingDraftFromRun, setIsCreatingDraftFromRun] = useState(false);
  const [isForking, setIsForking] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [selectedArtifact, setSelectedArtifact] = useState<CheckpointArtifact>("latest");
  const [watchingArtifact, setWatchingArtifact] = useState<CheckpointArtifact | null>(null);
  const [isResettingTrackPool, setIsResettingTrackPool] = useState(false);

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

  async function resumeRun() {
    setIsResuming(true);
    setControlError(null);
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
    try {
      await onStop(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to stop run");
    } finally {
      setIsStopping(false);
    }
  }

  async function renameRunLabel() {
    setIsRenaming(true);
    setControlError(null);
    try {
      await onRename(run.id, runName.trim());
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to rename run");
    } finally {
      setIsRenaming(false);
    }
  }

  async function openRunDirectoryInBrowser() {
    setIsOpeningDirectory(true);
    setControlError(null);
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
    try {
      await onWatch(run.id, artifact);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : `failed to watch ${artifact}`);
    } finally {
      setWatchingArtifact((current) => (current === artifact ? null : current));
    }
  }

  async function resetTrackPoolState() {
    setIsResettingTrackPool(true);
    setControlError(null);
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
    selectedArtifact,
    setSelectedArtifact,
    stopRun,
    watchRunArtifact,
    watchingArtifact,
  };
}
