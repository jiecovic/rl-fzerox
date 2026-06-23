// web/run-manager/src/widgets/saveGameWorkspace/SaveGameWorkspace.tsx
import { useMemo, useRef } from "react";

import type { SaveGameSession } from "@/app/workspace/types";
import {
  summarizeUnlockTargets,
  unlockCompletionFraction,
  unlockTargetKey,
} from "@/entities/saveGame/model";
import { SaveGameOverview } from "@/entities/saveGame/ui/SaveGameOverview";
import { randomAttemptSeedText } from "@/features/careerRunner/model/runnerSeed";
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
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { Button } from "@/shared/ui/Button";
import { FolderIcon, RenameIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";
import {
  resolveLaunchCupVehicleId,
  runnerSettingsDirty,
  startableUnlockTargetKeys,
  targetSummaryHasStarted,
} from "@/widgets/saveGameWorkspace/model";
import {
  type ImportSaveEngineTuningRequest,
  type SaveCourseSetupRequest,
  type SaveCupSetupRequest,
  useSaveGameWorkspaceController,
} from "@/widgets/saveGameWorkspace/useSaveGameWorkspaceController";

interface SaveGameWorkspaceProps {
  metadata: ConfigMetadata | null;
  onGlobalError: (message: string | null) => void;
  onCreateSaveGame: (name: string) => Promise<ManagedSaveGame>;
  onOpenSaveGameDirectory: (saveGameId: string) => Promise<void>;
  onImportEngineTuning: (
    request: ImportSaveEngineTuningRequest,
  ) => Promise<readonly SaveEngineTuningCourseSetupRecommendation[]>;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  onRefreshStatus: (saveGameId: string) => Promise<void>;
  onRenameSaveGame: (saveGameId: string, name: string) => Promise<void>;
  onUpdateRunnerSettings: (
    request: SaveGameRunnerSettingsUpdateRequest,
  ) => Promise<ManagedSaveGame>;
  onUpsertCourseSetup: (request: SaveCourseSetupRequest) => Promise<ManagedSaveGame>;
  onUpsertCupSetup: (request: SaveCupSetupRequest) => Promise<ManagedSaveGame>;
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
  useSaveGameRunnerRefresh({ onRefreshStatus, saveGame });
  const {
    copiedDetail,
    courseSetupDirty,
    copyDetail,
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
  } = useSaveGameWorkspaceController({
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
  });
  const assignableRuns = useStableAssignableRuns(runs);

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

function useStableAssignableRuns(runs: readonly ManagedRun[]): readonly ManagedRun[] {
  const stableAssignableRuns = useRef<{ key: string; runs: readonly ManagedRun[] } | null>(null);
  const key = useMemo(() => assignableRunsKey(runs), [runs]);
  if (stableAssignableRuns.current?.key !== key) {
    stableAssignableRuns.current = {
      key,
      runs: runs.filter((run) => run.status !== "created" && run.status !== "archived"),
    };
  }
  return stableAssignableRuns.current.runs;
}

const EMPTY_BUILT_IN_COURSES: ConfigMetadata["built_in_courses"] = [];
const EMPTY_COURSE_SETUPS: ManagedSaveGame["course_setups"] = [];
const EMPTY_CUP_SETUPS: ManagedSaveGame["cup_setups"] = [];
const EMPTY_RENDERER_OPTIONS: ReturnType<typeof rendererNames> = [];
const EMPTY_STRING_ARRAY: readonly string[] = [];
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
    .filter((run) => run.status !== "created" && run.status !== "archived")
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
