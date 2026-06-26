// web/run-manager/src/widgets/saveGameWorkspace/useSaveGameWorkspaceController.ts
import { useCallback, useRef, useState } from "react";

import type { SaveGameSession } from "@/app/workspace/types";
import { nextUnlockTarget } from "@/entities/saveGame/model";
import { parseAttemptSeed } from "@/features/careerRunner/model/runnerSeed";
import { resolveSavedCourseSetup } from "@/features/saveGameCourseSetup/model/courseSetup";
import type {
  CareerModeRunnerLaunchRequest,
  ConfigMetadata,
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
  SaveEngineTuningCourseSetupRecommendation,
  SaveGameRunnerSettingsUpdateRequest,
  SavePolicyArtifact,
  SavePolicySourceKind,
} from "@/shared/api/contract";
import {
  launchableTargetStatus,
  parseTargetClearGoal,
  resolveLaunchCupVehicleId,
} from "@/widgets/saveGameWorkspace/model";

export interface SaveCourseSetupRequest {
  engineSettingRawValue: number;
  policyArtifact: SavePolicyArtifact;
  policySourceId: string;
  policySourceKind: SavePolicySourceKind;
  saveGameId: string;
  courseId?: string | null;
  cupId?: string | null;
  difficulty?: string | null;
}

export interface SaveCupSetupRequest {
  cupId: string;
  saveGameId: string;
  vehicleId: string;
  difficulty?: string | null;
}

export interface ImportSaveEngineTuningRequest {
  courseSetups: readonly {
    courseId: string;
    cupId: string;
    difficulty?: string | null;
    vehicleId: string;
  }[];
  policyArtifact: SavePolicyArtifact;
  policySourceId: string;
  policySourceKind: SavePolicySourceKind;
  saveGameId: string;
}

interface SaveGameWorkspaceControllerOptions {
  metadata: ConfigMetadata | null;
  onCreateSaveGame: (name: string) => Promise<ManagedSaveGame>;
  onGlobalError: (message: string | null) => void;
  onImportEngineTuning: (
    request: ImportSaveEngineTuningRequest,
  ) => Promise<readonly SaveEngineTuningCourseSetupRecommendation[]>;
  onOpenSaveGameDirectory: (saveGameId: string) => Promise<void>;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  onRefreshStatus: (saveGameId: string) => Promise<void>;
  onRenameSaveGame: (saveGameId: string, name: string) => Promise<void>;
  onStartCareerMode: (
    request: CareerModeRunnerLaunchRequest,
  ) => Promise<"started" | "already_running">;
  onUpdateRunnerSettings: (
    request: SaveGameRunnerSettingsUpdateRequest,
  ) => Promise<ManagedSaveGame>;
  onUpsertCourseSetup: (request: SaveCourseSetupRequest) => Promise<ManagedSaveGame>;
  onUpsertCupSetup: (request: SaveCupSetupRequest) => Promise<ManagedSaveGame>;
  saveGame: ManagedSaveGame | null;
  session: SaveGameSession;
}

export function useSaveGameWorkspaceController({
  metadata,
  onCreateSaveGame,
  onGlobalError,
  onImportEngineTuning,
  onOpenSaveGameDirectory,
  onPatchSession,
  onRefreshStatus,
  onRenameSaveGame,
  onStartCareerMode,
  onUpdateRunnerSettings,
  onUpsertCourseSetup,
  onUpsertCupSetup,
  saveGame,
  session,
}: SaveGameWorkspaceControllerOptions) {
  const [copiedDetail, setCopiedDetail] = useState<"id" | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [openingSaveGameId, setOpeningSaveGameId] = useState<string | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renamingSaveGameId, setRenamingSaveGameId] = useState<string | null>(null);
  const [startingRunnerSaveGameId, setStartingRunnerSaveGameId] = useState<string | null>(null);
  const [savingRunnerSettingsSaveGameId, setSavingRunnerSettingsSaveGameId] = useState<
    string | null
  >(null);
  const [runnerStatus, setRunnerStatus] = useState<string | null>(null);
  const [courseSetupDirty, setCourseSetupDirty] = useState(false);
  const [updatingSaveGameId, setUpdatingSaveGameId] = useState<string | null>(null);

  const patchSessionFromRunnerSettings = useCallback(
    function patchSessionFromRunnerSettings(target: ManagedSaveGame) {
      const settings = target.runner_settings;
      onPatchSession(session.sessionId, {
        attemptSeedText: settings.attempt_seed === null ? "" : String(settings.attempt_seed),
        policyMode: settings.policy_mode,
        recordingEnabled: settings.recording_enabled,
        recordingInputHudEnabled: settings.recording_input_hud_enabled,
        recordingUpscaleFactor: settings.recording_upscale_factor,
        runnerDevice: settings.device,
        runnerRenderer: settings.renderer,
        perfectRun: settings.target_restart_on_retire,
        targetClearGoalText: String(settings.target_clear_goal),
        keepFailedPerfectRunVideos: settings.keep_failed_recordings,
        reloadPolicyBetweenAttempts: settings.reload_policy_between_attempts,
      });
    },
    [onPatchSession, session.sessionId],
  );

  async function createSaveGame() {
    const name = session.nameText.trim();
    if (name.length === 0) {
      onGlobalError("career name is required");
      return;
    }
    onGlobalError(null);
    setIsCreating(true);
    try {
      const created = await onCreateSaveGame(name);
      onPatchSession(session.sessionId, {
        nameText: created.name,
        saveGameId: created.id,
        title: created.name,
      });
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to create career save");
    } finally {
      setIsCreating(false);
    }
  }

  async function openSaveDirectory(target: ManagedSaveGame) {
    onGlobalError(null);
    setOpeningSaveGameId(target.id);
    try {
      await onOpenSaveGameDirectory(target.id);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to open career folder");
    } finally {
      setOpeningSaveGameId(null);
    }
  }

  async function renameSaveGame(target: ManagedSaveGame, name: string) {
    onGlobalError(null);
    setRenamingSaveGameId(target.id);
    try {
      await onRenameSaveGame(target.id, name);
      onPatchSession(session.sessionId, {
        nameText: name,
        title: name,
      });
      setRenameDialogOpen(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to rename career save");
    } finally {
      setRenamingSaveGameId(null);
    }
  }

  const upsertCourseSetup = useCallback(
    async function upsertCourseSetup(request: SaveCourseSetupRequest) {
      onGlobalError(null);
      setUpdatingSaveGameId(request.saveGameId);
      try {
        return await onUpsertCourseSetup(request);
      } catch (caught) {
        onGlobalError(caught instanceof Error ? caught.message : "failed to save course setup");
        throw caught;
      } finally {
        setUpdatingSaveGameId(null);
      }
    },
    [onGlobalError, onUpsertCourseSetup],
  );

  const upsertCupSetup = useCallback(
    async function upsertCupSetup(request: SaveCupSetupRequest) {
      onGlobalError(null);
      setUpdatingSaveGameId(request.saveGameId);
      try {
        return await onUpsertCupSetup(request);
      } catch (caught) {
        onGlobalError(caught instanceof Error ? caught.message : "failed to save cup setup");
        throw caught;
      } finally {
        setUpdatingSaveGameId(null);
      }
    },
    [onGlobalError, onUpsertCupSetup],
  );

  const importEngineTuning = useCallback(
    async function importEngineTuning(request: ImportSaveEngineTuningRequest) {
      onGlobalError(null);
      setUpdatingSaveGameId(request.saveGameId);
      try {
        return await onImportEngineTuning(request);
      } catch (caught) {
        onGlobalError(caught instanceof Error ? caught.message : "failed to import engine tuning");
        throw caught;
      } finally {
        setUpdatingSaveGameId(null);
      }
    },
    [onGlobalError, onImportEngineTuning],
  );

  function runnerSettingsRequest(
    target: ManagedSaveGame,
  ): SaveGameRunnerSettingsUpdateRequest | null {
    const attemptSeed = parseAttemptSeed(session.attemptSeedText);
    if (attemptSeed === "invalid") {
      onGlobalError("runtime seed must be an integer from 0 to 4294967295");
      return null;
    }
    return {
      attemptSeed,
      device: session.runnerDevice,
      policyMode: session.policyMode,
      recordingEnabled: session.recordingEnabled,
      recordingInputHudEnabled: session.recordingInputHudEnabled,
      recordingUpscaleFactor: session.recordingUpscaleFactor,
      recordingPath: null,
      renderer: session.runnerRenderer,
      saveGameId: target.id,
      targetRestartOnRetire: session.perfectRun,
      targetClearGoal: parseTargetClearGoal(session.targetClearGoalText),
      keepFailedRecordings: session.keepFailedPerfectRunVideos,
      reloadPolicyBetweenAttempts: session.reloadPolicyBetweenAttempts,
    };
  }

  async function saveRunnerSettings(
    target: ManagedSaveGame,
    options: { showNotice: boolean } = { showNotice: true },
  ): Promise<ManagedSaveGame | null> {
    const request = runnerSettingsRequest(target);
    if (request === null) {
      return null;
    }
    onGlobalError(null);
    setSavingRunnerSettingsSaveGameId(target.id);
    try {
      const updated = await onUpdateRunnerSettings(request);
      patchSessionFromRunnerSettings(updated);
      if (options.showNotice) {
        setRunnerStatus("Runner settings saved.");
      }
      return updated;
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to save runner settings");
      return null;
    } finally {
      setSavingRunnerSettingsSaveGameId(null);
    }
  }

  async function startCareerMode(
    target: ManagedSaveGame,
    requestedTarget: ManagedSaveUnlockTarget | null = null,
  ) {
    const settingsRequest = runnerSettingsRequest(target);
    if (settingsRequest === null) {
      return;
    }
    const launchTarget = requestedTarget ?? nextUnlockTarget(target);
    if (launchTarget === null) {
      onGlobalError("career save has no pending unlock target");
      return;
    }
    if (!launchableTargetStatus(launchTarget)) {
      onGlobalError("selected unlock target is not playable");
      return;
    }
    if (metadata === null) {
      onGlobalError("run manager metadata is missing");
      return;
    }
    if (
      resolveSavedCourseSetup(target.course_setups, launchTarget, metadata.built_in_courses) ===
      null
    ) {
      onGlobalError("choose a policy for the selected cup");
      return;
    }
    if (
      resolveLaunchCupVehicleId(
        target.cup_setups,
        target.unlock_progress?.unlocked_vehicle_ids ?? [],
        launchTarget,
      ) === null
    ) {
      onGlobalError("choose a vehicle for the selected cup");
      return;
    }
    onGlobalError(null);
    setRunnerStatus(null);
    setStartingRunnerSaveGameId(target.id);
    try {
      const updated = await onUpdateRunnerSettings(settingsRequest);
      patchSessionFromRunnerSettings(updated);
      const targetRecordingEnabled = requestedTarget !== null && settingsRequest.recordingEnabled;
      const status = await onStartCareerMode({
        attemptSeed: settingsRequest.attemptSeed,
        device: settingsRequest.device,
        policyMode: settingsRequest.policyMode,
        recordingEnabled: settingsRequest.recordingEnabled,
        recordingInputHudEnabled: settingsRequest.recordingInputHudEnabled,
        recordingUpscaleFactor: settingsRequest.recordingUpscaleFactor,
        recordingPath: null,
        renderer: settingsRequest.renderer,
        saveGameId: target.id,
        singleTarget: requestedTarget !== null,
        perfectRun: requestedTarget !== null && settingsRequest.targetRestartOnRetire,
        keepFailedRecordings: !targetRecordingEnabled || settingsRequest.keepFailedRecordings,
        reloadPolicyBetweenAttempts: settingsRequest.reloadPolicyBetweenAttempts,
        targetClearGoal: targetRecordingEnabled ? settingsRequest.targetClearGoal : 0,
        target: requestedTarget,
      });
      setRunnerStatus(status === "started" ? "Runner started." : "Runner is already open.");
      await onRefreshStatus(target.id);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to start runner");
    } finally {
      setStartingRunnerSaveGameId(null);
    }
  }

  async function copyDetail(value: string, detail: "id") {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedDetail(detail);
      onGlobalError(null);
    } catch {
      onGlobalError("failed to copy value");
    }
  }

  const saveGameRef = useRef<ManagedSaveGame | null>(null);
  const startCareerModeRef = useRef<typeof startCareerMode | null>(null);
  saveGameRef.current = saveGame;
  startCareerModeRef.current = startCareerMode;

  const handleStartRunner = useCallback(() => {
    const currentSaveGame = saveGameRef.current;
    const currentStartCareerMode = startCareerModeRef.current;
    if (currentSaveGame !== null && currentStartCareerMode !== null) {
      void currentStartCareerMode(currentSaveGame);
    }
  }, []);

  const handleStartTarget = useCallback((target: ManagedSaveUnlockTarget) => {
    const currentSaveGame = saveGameRef.current;
    const currentStartCareerMode = startCareerModeRef.current;
    if (currentSaveGame !== null && currentStartCareerMode !== null) {
      void currentStartCareerMode(currentSaveGame, target);
    }
  }, []);

  return {
    copiedDetail,
    courseSetupDirty,
    createSaveGame,
    handleStartRunner,
    handleStartTarget,
    importEngineTuning,
    isCreating,
    openSaveDirectory,
    openingSaveGameId,
    renameDialogOpen,
    renameSaveGame,
    renamingSaveGameId,
    runnerStatus,
    saveRunnerSettings,
    savingRunnerSettingsSaveGameId,
    setCourseSetupDirty,
    setRenameDialogOpen,
    startingRunnerSaveGameId,
    updatingSaveGameId,
    upsertCourseSetup,
    upsertCupSetup,
    copyDetail,
  };
}
