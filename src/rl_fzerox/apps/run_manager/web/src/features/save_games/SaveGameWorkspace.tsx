// src/rl_fzerox/apps/run_manager/web/src/features/save_games/SaveGameWorkspace.tsx
import { useMemo, useState } from "react";

import type { SaveGameSession } from "@/app/workspace/types";
import { CreateSaveGameForm } from "@/features/save_games/CreateSaveGameForm";
import {
  careerSaveHasStarted,
  nextUnlockTarget,
  summarizeSaveGameTargets,
  unlockCompletionFraction,
} from "@/features/save_games/model";
import { RunnerControlPanel } from "@/features/save_games/RunnerControlPanel";
import { parseAttemptSeed, randomAttemptSeedText } from "@/features/save_games/runnerSeed";
import { SaveGameOverview } from "@/features/save_games/SaveGameOverview";
import { UnlockPathPanel } from "@/features/save_games/UnlockPathPanel";
import { resolveSavedCourseSetup } from "@/features/save_games/unlock_path/courseSetupModel";
import { useSaveGameRunnerRefresh } from "@/features/save_games/useSaveGameRunnerRefresh";
import type {
  ConfigMetadata,
  CourseSetupScope,
  ManagedRun,
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
  PolicyPlaybackMode,
  SavePolicyArtifact,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { Button } from "@/shared/ui/Button";
import { FloatingNotice } from "@/shared/ui/FloatingNotice";
import { FolderIcon, RenameIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface SaveGameWorkspaceProps {
  metadata: ConfigMetadata | null;
  onCreateSaveGame: (name: string) => Promise<ManagedSaveGame>;
  onOpenSaveGameDirectory: (saveGameId: string) => Promise<void>;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  onRefresh: () => Promise<void>;
  onRenameSaveGame: (saveGameId: string, name: string) => Promise<void>;
  onUpsertCourseSetup: (request: {
    courseId?: string | null;
    cupId?: string | null;
    difficulty?: string | null;
    engineSettingRawValue: number;
    policyArtifact: SavePolicyArtifact;
    policyRunId: string;
    saveGameId: string;
    scope: CourseSetupScope;
    vehicleId: string;
  }) => Promise<ManagedSaveGame>;
  onStartCareerMode: (
    saveGameId: string,
    device: WatchDevice,
    renderer: WatchRenderer | null,
    attemptSeed: string | null,
    policyMode: PolicyPlaybackMode,
    target: ManagedSaveUnlockTarget | null,
  ) => Promise<"started" | "already_running">;
  runs: ManagedRun[];
  saveGame: ManagedSaveGame | null;
  session: SaveGameSession;
}

export function SaveGameWorkspace({
  metadata,
  onCreateSaveGame,
  onOpenSaveGameDirectory,
  onPatchSession,
  onRefresh,
  onRenameSaveGame,
  onUpsertCourseSetup,
  onStartCareerMode,
  runs,
  saveGame,
  session,
}: SaveGameWorkspaceProps) {
  const [error, setError] = useState<string | null>(null);
  const [copiedDetail, setCopiedDetail] = useState<"id" | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [openingSaveGameId, setOpeningSaveGameId] = useState<string | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renamingSaveGameId, setRenamingSaveGameId] = useState<string | null>(null);
  const [startingRunnerSaveGameId, setStartingRunnerSaveGameId] = useState<string | null>(null);
  const [runnerStatus, setRunnerStatus] = useState<string | null>(null);
  const [courseSetupDirty, setCourseSetupDirty] = useState(false);
  const [updatingSaveGameId, setUpdatingSaveGameId] = useState<string | null>(null);
  const assignableRuns = useMemo(() => runs.filter((run) => run.status !== "created"), [runs]);

  useSaveGameRunnerRefresh({ onRefresh, saveGame });

  async function createSaveGame() {
    const name = session.nameText.trim();
    if (name.length === 0) {
      setError("career name is required");
      return;
    }
    setError(null);
    setIsCreating(true);
    try {
      const created = await onCreateSaveGame(name);
      onPatchSession(session.sessionId, {
        nameText: created.name,
        saveGameId: created.id,
        title: created.name,
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to create career save");
    } finally {
      setIsCreating(false);
    }
  }

  async function openSaveDirectory(target: ManagedSaveGame) {
    setError(null);
    setOpeningSaveGameId(target.id);
    try {
      await onOpenSaveGameDirectory(target.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to open career folder");
    } finally {
      setOpeningSaveGameId(null);
    }
  }

  async function renameSaveGame(target: ManagedSaveGame, name: string) {
    setError(null);
    setRenamingSaveGameId(target.id);
    try {
      await onRenameSaveGame(target.id, name);
      onPatchSession(session.sessionId, {
        nameText: name,
        title: name,
      });
      setRenameDialogOpen(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to rename career save");
    } finally {
      setRenamingSaveGameId(null);
    }
  }

  async function upsertCourseSetup(request: {
    courseId?: string | null;
    cupId?: string | null;
    difficulty?: string | null;
    engineSettingRawValue: number;
    policyArtifact: SavePolicyArtifact;
    policyRunId: string;
    saveGameId: string;
    scope: CourseSetupScope;
    vehicleId: string;
  }) {
    setError(null);
    setUpdatingSaveGameId(request.saveGameId);
    try {
      return await onUpsertCourseSetup(request);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to save course setup");
      throw caught;
    } finally {
      setUpdatingSaveGameId(null);
    }
  }

  async function startCareerMode(
    target: ManagedSaveGame,
    requestedTarget: ManagedSaveUnlockTarget | null = null,
  ) {
    const attemptSeed = parseAttemptSeed(session.attemptSeedText);
    if (attemptSeed === "invalid") {
      setError("runtime seed must be an integer from 0 to 4294967295");
      return;
    }
    const launchTarget = requestedTarget ?? nextUnlockTarget(target);
    if (launchTarget === null) {
      setError("career save has no pending unlock target");
      return;
    }
    if (launchTarget.status !== "pending") {
      setError("selected unlock target is not pending");
      return;
    }
    if (metadata === null) {
      setError("run manager metadata is missing");
      return;
    }
    if (
      resolveSavedCourseSetup(target.course_setups, launchTarget, metadata.built_in_courses) ===
      null
    ) {
      setError("choose a policy for the selected cup");
      return;
    }
    setError(null);
    setRunnerStatus(null);
    setStartingRunnerSaveGameId(target.id);
    try {
      const status = await onStartCareerMode(
        target.id,
        session.runnerDevice,
        session.runnerRenderer,
        attemptSeed,
        session.policyMode,
        requestedTarget,
      );
      setRunnerStatus(status === "started" ? "Runner started." : "Runner is already open.");
      await onRefresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to start runner");
    } finally {
      setStartingRunnerSaveGameId(null);
    }
  }

  async function copyDetail(value: string, detail: "id") {
    try {
      await navigator.clipboard.writeText(value);
      setCopiedDetail(detail);
      setError(null);
    } catch {
      setError("failed to copy value");
    }
  }

  if (session.saveGameId === null) {
    return (
      <Panel>
        <PanelHeader
          title="New Career Save"
          subtitle="Create a local game save before configuring the unlock path."
        />
        <CreateSaveGameForm
          error={error}
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

  const activeSaveGame = saveGame;
  const activeMetadata = metadata;
  const targetSummary = summarizeSaveGameTargets(activeSaveGame);
  const completion = unlockCompletionFraction(targetSummary);
  const nextTarget = nextUnlockTarget(activeSaveGame);
  const nextSetup =
    nextTarget === null
      ? null
      : resolveSavedCourseSetup(
          activeSaveGame.course_setups,
          nextTarget,
          activeMetadata.built_in_courses,
        );
  function canStartUnlockTarget(target: ManagedSaveUnlockTarget): boolean {
    return (
      startingRunnerSaveGameId === null &&
      !activeSaveGame.runner_active &&
      !courseSetupDirty &&
      target.status === "pending" &&
      resolveSavedCourseSetup(
        activeSaveGame.course_setups,
        target,
        activeMetadata.built_in_courses,
      ) !== null
    );
  }
  const canStartRunner = nextTarget !== null && canStartUnlockTarget(nextTarget);
  const startLabel = saveGame.runner_active
    ? "Running"
    : careerSaveHasStarted(saveGame)
      ? "Continue"
      : "Start";
  const startNote = saveGame.runner_active
    ? "Career Mode runner is active."
    : courseSetupDirty
      ? "Save course setups before starting."
      : nextTarget === null
        ? "All unlock targets are complete."
        : nextSetup === null
          ? "Choose a policy for the next cup."
          : `${nextTarget.label} is ready.`;
  const runnerLabel = saveGame.runner_active
    ? "running"
    : careerSaveHasStarted(saveGame)
      ? "ready to continue"
      : "not started";
  const rendererOptions = rendererNames(metadata, session.runnerRenderer);
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
      {error !== null ? <FloatingNotice tone="error">{error}</FloatingNotice> : null}
      {runnerStatus !== null ? <FloatingNotice>{runnerStatus}</FloatingNotice> : null}
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
        rendererOptions={rendererOptions}
        runnerDevice={session.runnerDevice}
        runnerRenderer={session.runnerRenderer}
        policyMode={session.policyMode}
        startLabel={startLabel}
        startNote={startNote}
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
        onStart={() => void startCareerMode(saveGame)}
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
          metadata={metadata}
          saveGame={saveGame}
          updating={updatingSaveGameId === saveGame.id}
          canStartTarget={canStartUnlockTarget}
          onCourseSetupDirtyChange={setCourseSetupDirty}
          onStartTarget={(target) => void startCareerMode(saveGame, target)}
          onUpsertCourseSetup={upsertCourseSetup}
        />
      </article>
    </Panel>
  );
}
