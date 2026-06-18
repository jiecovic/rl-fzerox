// web/run-manager/src/widgets/saveGameWorkspace/SaveGameWorkspace.tsx
import { useCallback, useMemo, useRef, useState } from "react";

import type { SaveGameSession } from "@/app/workspace/types";
import {
  nextUnlockTarget,
  summarizeUnlockTargets,
  type UnlockTargetSummary,
  unlockCompletionFraction,
  unlockTargetKey,
} from "@/entities/saveGame/model";
import { SaveGameOverview } from "@/entities/saveGame/ui/SaveGameOverview";
import { parseAttemptSeed, randomAttemptSeedText } from "@/features/careerRunner/model/runnerSeed";
import { useSaveGameRunnerRefresh } from "@/features/careerRunner/model/useSaveGameRunnerRefresh";
import { RunnerControlPanel } from "@/features/careerRunner/ui/RunnerControlPanel";
import { CreateSaveGameForm } from "@/features/createSaveGame/ui/CreateSaveGameForm";
import {
  resolveSavedCourseSetup,
  resolveSavedCupSetup,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import { UnlockPathPanel } from "@/features/saveGameCourseSetup/ui/UnlockPathPanel";
import type {
  CareerModeRunnerLaunchRequest,
  ConfigMetadata,
  ManagedRun,
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
  SaveEngineTuningCourseSetupRecommendation,
  SaveGameRunnerSettingsUpdateRequest,
  SavePolicyArtifact,
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { Button } from "@/shared/ui/Button";
import { FolderIcon, RenameIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface SaveGameWorkspaceProps {
  metadata: ConfigMetadata | null;
  onGlobalError: (message: string | null) => void;
  onCreateSaveGame: (name: string) => Promise<ManagedSaveGame>;
  onOpenSaveGameDirectory: (saveGameId: string) => Promise<void>;
  onImportEngineTuning: (request: {
    courseSetups: readonly {
      courseId: string;
      cupId: string;
      difficulty?: string | null;
      vehicleId: string;
    }[];
    policyArtifact: SavePolicyArtifact;
    policyRunId: string;
    saveGameId: string;
  }) => Promise<readonly SaveEngineTuningCourseSetupRecommendation[]>;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  onRefreshStatus: (saveGameId: string) => Promise<void>;
  onRenameSaveGame: (saveGameId: string, name: string) => Promise<void>;
  onUpdateRunnerSettings: (
    request: SaveGameRunnerSettingsUpdateRequest,
  ) => Promise<ManagedSaveGame>;
  onUpsertCourseSetup: (request: {
    courseId?: string | null;
    cupId?: string | null;
    difficulty?: string | null;
    engineSettingRawValue: number;
    policyArtifact: SavePolicyArtifact;
    policyRunId: string;
    saveGameId: string;
  }) => Promise<ManagedSaveGame>;
  onUpsertCupSetup: (request: {
    cupId: string;
    difficulty?: string | null;
    saveGameId: string;
    vehicleId: string;
  }) => Promise<ManagedSaveGame>;
  onStartCareerMode: (
    request: CareerModeRunnerLaunchRequest,
  ) => Promise<"started" | "already_running">;
  runs: ManagedRun[];
  saveGame: ManagedSaveGame | null;
  session: SaveGameSession;
}

export function SaveGameWorkspace({
  metadata,
  onGlobalError,
  onCreateSaveGame,
  onOpenSaveGameDirectory,
  onImportEngineTuning,
  onPatchSession,
  onRefreshStatus,
  onRenameSaveGame,
  onUpdateRunnerSettings,
  onUpsertCourseSetup,
  onUpsertCupSetup,
  onStartCareerMode,
  runs,
  saveGame,
  session,
}: SaveGameWorkspaceProps) {
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
  const assignableRuns = useStableAssignableRuns(runs);

  useSaveGameRunnerRefresh({ onRefreshStatus, saveGame });

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
    async function upsertCourseSetup(request: {
      courseId?: string | null;
      cupId?: string | null;
      difficulty?: string | null;
      engineSettingRawValue: number;
      policyArtifact: SavePolicyArtifact;
      policyRunId: string;
      saveGameId: string;
    }) {
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
    async function upsertCupSetup(request: {
      cupId: string;
      difficulty?: string | null;
      saveGameId: string;
      vehicleId: string;
    }) {
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
    async function importEngineTuning(request: {
      courseSetups: readonly {
        courseId: string;
        cupId: string;
        difficulty?: string | null;
        vehicleId: string;
      }[];
      policyArtifact: SavePolicyArtifact;
      policyRunId: string;
      saveGameId: string;
    }) {
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

  const saveGameCourseSetups = saveGame?.course_setups ?? EMPTY_COURSE_SETUPS;
  const saveGameCupSetups = saveGame?.cup_setups ?? EMPTY_CUP_SETUPS;
  const saveGameUnlockProgress = saveGame?.unlock_progress ?? null;
  const builtInCourses = metadata?.built_in_courses ?? EMPTY_BUILT_IN_COURSES;
  const hasSaveGame = saveGame !== null;
  const runnerActive = saveGame?.runner_active ?? false;
  const unlockedVehicleIds = saveGameUnlockProgress?.unlocked_vehicle_ids ?? EMPTY_STRING_ARRAY;
  const unlockTargets = useStableUnlockTargets(
    saveGameUnlockProgress?.targets ?? EMPTY_UNLOCK_TARGETS,
  );
  const targetSummary = useMemo(() => summarizeUnlockTargets(unlockTargets), [unlockTargets]);
  const completion = useMemo(() => unlockCompletionFraction(targetSummary), [targetSummary]);
  const nextTarget = saveGameUnlockProgress?.next_target ?? null;
  const nextSetup = useMemo(
    () =>
      metadata === null || nextTarget === null
        ? null
        : resolveSavedCourseSetup(saveGameCourseSetups, nextTarget, builtInCourses),
    [builtInCourses, metadata, nextTarget, saveGameCourseSetups],
  );
  const nextCupSetup = useMemo(
    () => (nextTarget === null ? null : resolveSavedCupSetup(saveGameCupSetups, nextTarget)),
    [nextTarget, saveGameCupSetups],
  );
  const nextCupVehicleId = useMemo(
    () =>
      nextTarget === null
        ? null
        : resolveLaunchCupVehicleId(saveGameCupSetups, unlockedVehicleIds, nextTarget),
    [nextTarget, saveGameCupSetups, unlockedVehicleIds],
  );
  const startableTargetKeys = useMemo(
    () =>
      startableUnlockTargetKeys({
        builtInCourses,
        courseSetups: saveGameCourseSetups,
        cupSetups: saveGameCupSetups,
        disabled:
          metadata === null ||
          !hasSaveGame ||
          startingRunnerSaveGameId !== null ||
          runnerActive ||
          courseSetupDirty,
        targets: unlockTargets,
        unlockedVehicleIds,
      }),
    [
      builtInCourses,
      courseSetupDirty,
      hasSaveGame,
      metadata,
      runnerActive,
      saveGameCourseSetups,
      saveGameCupSetups,
      startingRunnerSaveGameId,
      unlockTargets,
      unlockedVehicleIds,
    ],
  );
  const canStartRunner =
    nextTarget !== null && startableTargetKeys.has(unlockTargetKey(nextTarget));
  const rendererOptions = useMemo(
    () =>
      metadata === null ? EMPTY_RENDERER_OPTIONS : rendererNames(metadata, session.runnerRenderer),
    [metadata, session.runnerRenderer],
  );

  if (session.saveGameId === null) {
    return (
      <Panel>
        <PanelHeader
          title="New Career Save"
          subtitle="Create a local game save before configuring the unlock path."
        />
        <CreateSaveGameForm
          isCreating={isCreating}
          session={session}
          onCreateSaveGame={() => void createSaveGame()}
          onPatchSession={onPatchSession}
        />
      </Panel>
    );
  }

  if (saveGame === null) {
    return (
      <Panel>
        <Notice tone="error">This career save is no longer available.</Notice>
      </Panel>
    );
  }

  if (metadata === null) {
    return (
      <Panel>
        <Notice tone="error">Run manager metadata is missing.</Notice>
      </Panel>
    );
  }

  const startLabel = saveGame.runner_active
    ? "Running"
    : nextTarget === null
      ? "Complete"
      : targetSummaryHasStarted(targetSummary)
        ? "Continue"
        : "Start";
  const startNote = saveGame.runner_active
    ? "Career Mode runner is active."
    : courseSetupDirty
      ? "Save course setups before starting."
      : nextTarget === null
        ? "All unlock targets are complete. Select a target below to replay it."
        : nextSetup === null
          ? "Choose a policy for the next cup."
          : nextCupSetup === null && nextCupVehicleId === null
            ? "Choose a vehicle for the next cup."
            : `${nextTarget.label} is ready.`;
  const runnerLabel = saveGame.runner_active
    ? "running"
    : targetSummaryHasStarted(targetSummary)
      ? "ready to continue"
      : "not started";
  const savingRunnerSettings = savingRunnerSettingsSaveGameId === saveGame.id;
  const runnerSettingsHaveChanges = runnerSettingsDirty(saveGame, session);
  const canSaveRunnerSettings =
    !saveGame.runner_active && startingRunnerSaveGameId === null && runnerSettingsHaveChanges;
  return (
    <Panel>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <PanelHeader
          title={
            <span className="inline-flex min-w-0 items-center gap-2">
              <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
                {saveGame.name}
              </span>
              <TooltipIconButton
                aria-label="Rename career save"
                disabled={renamingSaveGameId === saveGame.id}
                size="small"
                tooltip="Rename"
                onClick={() => setRenameDialogOpen(true)}
              >
                <RenameIcon />
              </TooltipIconButton>
            </span>
          }
          subtitle="Game save and unlock progress."
        />
        <Button
          className="gap-2"
          disabled={openingSaveGameId === saveGame.id}
          onClick={() => void openSaveDirectory(saveGame)}
        >
          <FolderIcon />
          <span>{openingSaveGameId === saveGame.id ? "Opening" : "Open folder"}</span>
        </Button>
      </div>
      {runnerStatus !== null ? <Notice>{runnerStatus}</Notice> : null}
      <RenameDialog
        busy={renamingSaveGameId === saveGame.id}
        initialName={saveGame.name}
        label="Career save name"
        open={renameDialogOpen}
        title="Rename career save"
        onClose={() => setRenameDialogOpen(false)}
        onSubmit={(name) => void renameSaveGame(saveGame, name)}
      />
      <RunnerControlPanel
        attemptSeedText={session.attemptSeedText}
        canStart={canStartRunner}
        canSaveSettings={canSaveRunnerSettings}
        rendererOptions={rendererOptions}
        runnerDevice={session.runnerDevice}
        runnerRenderer={session.runnerRenderer}
        policyMode={session.policyMode}
        recordingEnabled={session.recordingEnabled}
        recordingInputHudEnabled={session.recordingInputHudEnabled}
        recordingUpscaleFactor={session.recordingUpscaleFactor}
        startLabel={startLabel}
        startNote={startNote}
        savingSettings={savingRunnerSettings}
        starting={startingRunnerSaveGameId === saveGame.id}
        onAttemptSeedChange={(attemptSeedText) =>
          onPatchSession(session.sessionId, { attemptSeedText })
        }
        onRandomizeAttemptSeed={() =>
          onPatchSession(session.sessionId, {
            attemptSeedText: randomAttemptSeedText(),
          })
        }
        onRunnerDeviceChange={(runnerDevice) => onPatchSession(session.sessionId, { runnerDevice })}
        onRunnerRendererChange={(runnerRenderer) =>
          onPatchSession(session.sessionId, { runnerRenderer })
        }
        onPolicyModeChange={(policyMode) => onPatchSession(session.sessionId, { policyMode })}
        onRecordingEnabledChange={(recordingEnabled) =>
          onPatchSession(session.sessionId, { recordingEnabled })
        }
        onRecordingInputHudEnabledChange={(recordingInputHudEnabled) =>
          onPatchSession(session.sessionId, { recordingInputHudEnabled })
        }
        onRecordingUpscaleFactorChange={(recordingUpscaleFactor) =>
          onPatchSession(session.sessionId, { recordingUpscaleFactor })
        }
        onSaveSettings={() => void saveRunnerSettings(saveGame)}
        onStart={handleStartRunner}
      />
      <article className="grid content-start gap-5">
        <SaveGameOverview
          completion={completion}
          copiedSaveId={copiedDetail === "id"}
          nextTarget={nextTarget}
          runnerLabel={runnerLabel}
          saveGame={saveGame}
          targetSummary={targetSummary}
          onCopySaveId={() => void copyDetail(saveGame.id, "id")}
        />
        <UnlockPathPanel
          assignableRuns={assignableRuns}
          keepFailedPerfectRunVideos={session.keepFailedPerfectRunVideos}
          metadata={metadata}
          perfectRun={session.perfectRun}
          recordingEnabled={session.recordingEnabled}
          reloadPolicyBetweenAttempts={session.reloadPolicyBetweenAttempts}
          saveGame={saveGame}
          targetClearGoalText={session.targetClearGoalText}
          targets={unlockTargets}
          updating={updatingSaveGameId === saveGame.id}
          startableTargetKeys={startableTargetKeys}
          onCourseSetupDirtyChange={setCourseSetupDirty}
          onImportEngineTuning={importEngineTuning}
          onKeepFailedPerfectRunVideosChange={(keepFailedPerfectRunVideos) =>
            onPatchSession(session.sessionId, { keepFailedPerfectRunVideos })
          }
          onPerfectRunChange={(perfectRun) => onPatchSession(session.sessionId, { perfectRun })}
          onReloadPolicyBetweenAttemptsChange={(reloadPolicyBetweenAttempts) =>
            onPatchSession(session.sessionId, { reloadPolicyBetweenAttempts })
          }
          onStartTarget={handleStartTarget}
          onTargetClearGoalTextChange={(targetClearGoalText) =>
            onPatchSession(session.sessionId, { targetClearGoalText })
          }
          onUpsertCourseSetup={upsertCourseSetup}
          onUpsertCupSetup={upsertCupSetup}
        />
      </article>
    </Panel>
  );
}

function launchableTargetStatus(target: ManagedSaveUnlockTarget): boolean {
  return target.status === "pending" || target.status === "succeeded";
}

function parseTargetClearGoal(text: string): number {
  const parsed = Number.parseInt(text.trim(), 10);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, parsed);
}

function resolveLaunchCupVehicleId(
  cupSetups: ManagedSaveGame["cup_setups"],
  unlockedVehicleIds: readonly string[],
  target: ManagedSaveUnlockTarget,
): string | null {
  return resolveSavedCupSetup(cupSetups, target)?.vehicle_id ?? unlockedVehicleIds[0] ?? null;
}

function startableUnlockTargetKeys({
  builtInCourses,
  courseSetups,
  cupSetups,
  disabled,
  targets,
  unlockedVehicleIds,
}: {
  builtInCourses: ConfigMetadata["built_in_courses"];
  courseSetups: ManagedSaveGame["course_setups"];
  cupSetups: ManagedSaveGame["cup_setups"];
  disabled: boolean;
  targets: readonly ManagedSaveUnlockTarget[];
  unlockedVehicleIds: readonly string[];
}): ReadonlySet<string> {
  if (disabled) {
    return EMPTY_STRING_SET;
  }
  const keys = new Set<string>();
  for (const target of targets) {
    if (
      launchableTargetStatus(target) &&
      resolveSavedCourseSetup(courseSetups, target, builtInCourses) !== null &&
      resolveLaunchCupVehicleId(cupSetups, unlockedVehicleIds, target) !== null
    ) {
      keys.add(unlockTargetKey(target));
    }
  }
  return keys;
}

function targetSummaryHasStarted(summary: UnlockTargetSummary): boolean {
  return summary.succeeded > 0 || summary.failed > 0 || summary.skipped > 0;
}

function runnerSettingsDirty(saveGame: ManagedSaveGame, session: SaveGameSession): boolean {
  const settings = saveGame.runner_settings;
  const attemptSeed = parseAttemptSeed(session.attemptSeedText);
  const persistedAttemptSeed =
    settings.attempt_seed === null ? null : String(settings.attempt_seed);
  return (
    attemptSeed === "invalid" ||
    persistedAttemptSeed !== attemptSeed ||
    settings.device !== session.runnerDevice ||
    settings.renderer !== session.runnerRenderer ||
    settings.policy_mode !== session.policyMode ||
    settings.recording_enabled !== session.recordingEnabled ||
    settings.recording_input_hud_enabled !== session.recordingInputHudEnabled ||
    settings.recording_upscale_factor !== session.recordingUpscaleFactor ||
    settings.recording_path !== null ||
    settings.target_restart_on_retire !== session.perfectRun ||
    settings.target_clear_goal !== parseTargetClearGoal(session.targetClearGoalText) ||
    settings.keep_failed_recordings !== session.keepFailedPerfectRunVideos ||
    settings.reload_policy_between_attempts !== session.reloadPolicyBetweenAttempts
  );
}

function useStableAssignableRuns(runs: readonly ManagedRun[]): readonly ManagedRun[] {
  const stableAssignableRuns = useRef<{ key: string; runs: readonly ManagedRun[] } | null>(null);
  const key = useMemo(() => assignableRunsKey(runs), [runs]);
  if (stableAssignableRuns.current?.key !== key) {
    stableAssignableRuns.current = {
      key,
      runs: runs.filter((run) => run.status !== "created"),
    };
  }
  return stableAssignableRuns.current.runs;
}

const EMPTY_BUILT_IN_COURSES: ConfigMetadata["built_in_courses"] = [];
const EMPTY_COURSE_SETUPS: ManagedSaveGame["course_setups"] = [];
const EMPTY_CUP_SETUPS: ManagedSaveGame["cup_setups"] = [];
const EMPTY_RENDERER_OPTIONS: ReturnType<typeof rendererNames> = [];
const EMPTY_STRING_ARRAY: readonly string[] = [];
const EMPTY_STRING_SET: ReadonlySet<string> = new Set<string>();
const EMPTY_UNLOCK_TARGETS: readonly ManagedSaveUnlockTarget[] = [];

function useStableUnlockTargets(
  targets: readonly ManagedSaveUnlockTarget[],
): readonly ManagedSaveUnlockTarget[] {
  const stableTargets = useRef<{
    key: string;
    targets: readonly ManagedSaveUnlockTarget[];
  } | null>(null);
  const key = useMemo(() => unlockTargetsKey(targets), [targets]);
  if (stableTargets.current?.key !== key) {
    stableTargets.current = { key, targets };
  }
  return stableTargets.current.targets;
}

function unlockTargetsKey(targets: readonly ManagedSaveUnlockTarget[]): string {
  return targets
    .map((target) =>
      [
        target.sequence_index,
        target.kind,
        target.status,
        target.label,
        target.difficulty ?? "",
        target.cup_id ?? "",
        target.course_id ?? "",
      ].join("\u001f"),
    )
    .join("\u001e");
}

function assignableRunsKey(runs: readonly ManagedRun[]): string {
  return runs
    .filter((run) => run.status !== "created")
    .map((run) =>
      [
        run.id,
        run.name,
        run.status,
        run.vehicle_setup.engine_mode,
        run.vehicle_setup.engine_setting_raw_value,
        run.vehicle_setup.engine_setting_min_raw_value,
        run.vehicle_setup.engine_setting_max_raw_value,
        run.vehicle_setup.selected_vehicle_ids.join(","),
      ].join("\u001f"),
    )
    .join("\u001e");
}
