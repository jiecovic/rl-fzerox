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

export interface RunWorkspaceActionsProps {
  clearTrackSamplingState: (state: TrackSamplingRuntimeState | null) => void;
  onClearAltBaselines: (runId: string) => Promise<void>;
  onClearCourseAltBaselines: (runId: string, courseKey: string) => Promise<void>;
  onCreateDraftFromRun: (runId: string) => Promise<void>;
  onFork: (runId: string, artifact: CheckpointArtifact, copyAltBaselines: boolean) => Promise<void>;
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
  clearAltBaselines: () => Promise<void>;
  clearCourseAltBaselines: (courseKey: string) => Promise<void>;
  cancelForkAltBaselineChoice: () => void;
  confirmForkAltBaselineChoice: (copyAltBaselines: boolean) => Promise<void>;
  forkRunArtifact: (artifact: CheckpointArtifact) => Promise<void>;
  isCreatingDraftFromRun: boolean;
  isClearingAltBaselines: boolean;
  clearingAltBaselineCourseKey: string | null;
  isForking: boolean;
  isOpeningDirectory: boolean;
  isRenaming: boolean;
  isResettingTrackPool: boolean;
  isResuming: boolean;
  isStopping: boolean;
  openRunDirectoryInBrowser: () => Promise<void>;
  pendingForkAltBaselineChoice: PendingForkAltBaselineChoice | null;
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
  saveWatchLaunchSettings: () => void;
  setSelectedWatchRenderer: (renderer: WatchRenderer) => void;
  stopRun: () => Promise<void>;
  watchRunArtifact: (artifact: CheckpointArtifact) => Promise<void>;
  watchLaunchSettingsSaved: boolean;
  watchingArtifact: CheckpointArtifact | null;
}

export interface PendingForkAltBaselineChoice {
  artifact: CheckpointArtifact;
  count: number;
}

interface WatchLaunchPreference {
  artifact: CheckpointArtifact;
  device: WatchDevice;
  policyMode: PolicyPlaybackMode;
  renderer: WatchRenderer;
}

export function useRunWorkspaceActions({
  clearTrackSamplingState,
  onClearAltBaselines,
  onClearCourseAltBaselines,
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
  const [savedWatchPreference, setSavedWatchPreference] = useState<WatchLaunchPreference>(() =>
    readWatchLaunchPreference(run.config.environment.renderer),
  );
  const [selectedForkArtifact, setSelectedForkArtifact] = useState<CheckpointArtifact>("latest");
  const [selectedWatchArtifact, setSelectedWatchArtifact] = useState<CheckpointArtifact>(
    savedWatchPreference.artifact,
  );
  const [selectedWatchDevice, setSelectedWatchDevice] = useState<WatchDevice>(
    savedWatchPreference.device,
  );
  const [selectedWatchPolicyMode, setSelectedWatchPolicyMode] = useState<PolicyPlaybackMode>(
    savedWatchPreference.policyMode,
  );
  const [watchRendererSelection, setWatchRendererSelection] = useState<{
    renderer: WatchRenderer;
    runId: string;
  }>({
    renderer: savedWatchPreference.renderer,
    runId: run.id,
  });
  const [watchingArtifact, setWatchingArtifact] = useState<CheckpointArtifact | null>(null);
  const [isClearingAltBaselines, setIsClearingAltBaselines] = useState(false);
  const [clearingAltBaselineCourseKey, setClearingAltBaselineCourseKey] = useState<string | null>(
    null,
  );
  const [isResettingTrackPool, setIsResettingTrackPool] = useState(false);
  const [pendingForkAltBaselineChoice, setPendingForkAltBaselineChoice] =
    useState<PendingForkAltBaselineChoice | null>(null);
  const selectedWatchRenderer =
    watchRendererSelection.runId === run.id
      ? watchRendererSelection.renderer
      : run.config.environment.renderer;
  const setSelectedWatchRenderer = (renderer: WatchRenderer) => {
    setWatchRendererSelection({ renderer, runId: run.id });
  };
  const selectedWatchPreference: WatchLaunchPreference = {
    artifact: selectedWatchArtifact,
    device: selectedWatchDevice,
    policyMode: selectedWatchPolicyMode,
    renderer: selectedWatchRenderer,
  };
  const watchLaunchSettingsSaved = watchLaunchPreferencesEqual(
    selectedWatchPreference,
    savedWatchPreference,
  );

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

  async function renameRunLabel(name?: string) {
    const nextName = (name ?? runName).trim();
    if (nextName.length === 0) {
      setControlError("run name is required");
      return false;
    }
    setIsRenaming(true);
    setControlError(null);
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
    try {
      await onOpenDirectory(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to open run directory");
    } finally {
      setIsOpeningDirectory(false);
    }
  }

  async function forkRunArtifact(artifact: CheckpointArtifact) {
    if (run.active_alt_baseline_count > 0) {
      setControlError(null);
      setPendingForkAltBaselineChoice({
        artifact,
        count: run.active_alt_baseline_count,
      });
      return;
    }
    await executeForkRunArtifact(artifact, true);
  }

  async function confirmForkAltBaselineChoice(copyAltBaselines: boolean) {
    const pending = pendingForkAltBaselineChoice;
    if (pending === null) {
      return;
    }
    setPendingForkAltBaselineChoice(null);
    await executeForkRunArtifact(pending.artifact, copyAltBaselines);
  }

  function cancelForkAltBaselineChoice() {
    setPendingForkAltBaselineChoice(null);
  }

  async function executeForkRunArtifact(artifact: CheckpointArtifact, copyAltBaselines: boolean) {
    setIsForking(true);
    setControlError(null);
    try {
      await onFork(run.id, artifact, copyAltBaselines);
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
      const status = await onWatch(
        run.id,
        artifact,
        selectedWatchDevice,
        selectedWatchRenderer,
        selectedWatchPolicyMode,
      );
      if (status === "already_running") {
        setControlError(`${artifact} watch is already running`);
        return;
      }
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : `failed to watch ${artifact}`);
    } finally {
      setWatchingArtifact((current) => (current === artifact ? null : current));
    }
  }

  function saveWatchLaunchSettings() {
    writeWatchLaunchPreference(selectedWatchPreference);
    setSavedWatchPreference(selectedWatchPreference);
    setControlError(null);
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

  async function clearAltBaselines() {
    setIsClearingAltBaselines(true);
    setControlError(null);
    try {
      await onClearAltBaselines(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to clear alt baselines");
    } finally {
      setIsClearingAltBaselines(false);
    }
  }

  async function clearCourseAltBaselines(courseKey: string) {
    setClearingAltBaselineCourseKey(courseKey);
    setControlError(null);
    try {
      await onClearCourseAltBaselines(run.id, courseKey);
    } catch (caught) {
      setControlError(
        caught instanceof Error ? caught.message : `failed to clear alt baselines for ${courseKey}`,
      );
    } finally {
      setClearingAltBaselineCourseKey((current) => (current === courseKey ? null : current));
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
    clearingAltBaselineCourseKey,
    clearAltBaselines,
    clearCourseAltBaselines,
    cancelForkAltBaselineChoice,
    confirmForkAltBaselineChoice,
    createDraftFromRun,
    forkRunArtifact,
    isCreatingDraftFromRun,
    isClearingAltBaselines,
    isForking,
    isOpeningDirectory,
    isRenaming,
    isResettingTrackPool,
    isResuming,
    isStopping,
    openRunDirectoryInBrowser,
    pendingForkAltBaselineChoice,
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
    saveWatchLaunchSettings,
    setSelectedWatchRenderer,
    stopRun,
    watchRunArtifact,
    watchLaunchSettingsSaved,
    watchingArtifact,
  };
}

const WATCH_LAUNCH_PREFERENCE_STORAGE_KEY = "run-watch-launch-preference";

function readWatchLaunchPreference(defaultRenderer: WatchRenderer): WatchLaunchPreference {
  if (typeof window === "undefined") {
    return defaultWatchLaunchPreference(defaultRenderer);
  }
  try {
    const raw = window.localStorage.getItem(WATCH_LAUNCH_PREFERENCE_STORAGE_KEY);
    if (raw === null) {
      return defaultWatchLaunchPreference(defaultRenderer);
    }
    const value = JSON.parse(raw) as Partial<Record<keyof WatchLaunchPreference, unknown>>;
    return {
      artifact: value.artifact === "best" ? "best" : "latest",
      device: value.device === "cpu" ? "cpu" : "cuda",
      policyMode: value.policyMode === "stochastic" ? "stochastic" : "deterministic",
      renderer:
        value.renderer === "angrylion" || value.renderer === "gliden64"
          ? value.renderer
          : defaultRenderer,
    };
  } catch {
    return defaultWatchLaunchPreference(defaultRenderer);
  }
}

function writeWatchLaunchPreference(preference: WatchLaunchPreference) {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(WATCH_LAUNCH_PREFERENCE_STORAGE_KEY, JSON.stringify(preference));
  } catch {
    // Browser storage may be unavailable in private or restricted contexts.
  }
}

function defaultWatchLaunchPreference(defaultRenderer: WatchRenderer): WatchLaunchPreference {
  return {
    artifact: "latest",
    device: "cuda",
    policyMode: "deterministic",
    renderer: defaultRenderer,
  };
}

function watchLaunchPreferencesEqual(
  left: WatchLaunchPreference,
  right: WatchLaunchPreference,
): boolean {
  return (
    left.artifact === right.artifact &&
    left.device === right.device &&
    left.policyMode === right.policyMode &&
    left.renderer === right.renderer
  );
}
