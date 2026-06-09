// src/rl_fzerox/apps/run_manager/web/src/features/save_games/SaveGameWorkspace.tsx
import { useEffect, useMemo, useState } from "react";

import type { SaveGameSession } from "@/app/workspace/types";
import {
  careerSaveHasStarted,
  nextUnlockTarget,
  summarizeSaveGameTargets,
  titleizeIdentifier,
  unlockCompletionFraction,
} from "@/features/save_games/model";
import { ProgressMeter } from "@/features/save_games/ProgressMeter";
import { UnlockPathPanel } from "@/features/save_games/UnlockPathPanel";
import type {
  ConfigMetadata,
  CourseSetupScope,
  ManagedRun,
  ManagedSaveCourseSetup,
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
  PolicyPlaybackMode,
  SavePolicyArtifact,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { FloatingNotice } from "@/shared/ui/FloatingNotice";
import { formatDate } from "@/shared/ui/format";
import { CopyIcon, FolderIcon, PlayIcon, RandomizeIcon, RenameIcon } from "@/shared/ui/icons";
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

  useEffect(() => {
    if (saveGame === null || (!saveGame.runner_active && saveGame.status !== "running")) {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      void onRefresh();
    }, 1500);
    return () => window.clearInterval(intervalId);
  }, [onRefresh, saveGame]);

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
        <section className="grid gap-5 border border-app-border bg-app-surface p-5 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="grid content-start gap-4">
            <div className="grid gap-1">
              <h3 className="m-0 text-lg font-bold text-app-text">Unlock progress</h3>
              <p className="m-0 text-sm text-app-muted">
                {targetSummary.succeeded.toLocaleString()} of {targetSummary.total.toLocaleString()}{" "}
                targets complete
              </p>
            </div>
            <ProgressMeter label={`${saveGame.name} progress`} value={completion} />
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <StatusCard label="Targets" value={targetSummary.total.toLocaleString()} />
              <StatusCard label="Status" value={titleizeIdentifier(saveGame.status)} />
              <StatusCard label="Next target" value={nextTarget?.label ?? "none"} />
              <StatusCard label="Runner" value={runnerLabel} />
            </div>
          </div>
          <dl className="grid content-start gap-3 border border-app-border bg-app-surface-muted p-3 text-sm">
            <DetailRow
              copyLabel={copiedDetail === "id" ? "Copied" : "Copy save id"}
              label="Save id"
              value={saveGame.id}
              monospace
              onCopy={() => void copyDetail(saveGame.id, "id")}
            />
            <DetailRow label="Save path" value={saveGame.save_path} monospace />
            <DetailRow label="Created" value={formatDate(saveGame.created_at)} />
            <DetailRow label="Updated" value={formatDate(saveGame.updated_at)} />
          </dl>
        </section>
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

function RunnerControlPanel({
  attemptSeedText,
  canStart,
  onAttemptSeedChange,
  onRandomizeAttemptSeed,
  onPolicyModeChange,
  onRunnerDeviceChange,
  onRunnerRendererChange,
  onStart,
  policyMode,
  rendererOptions,
  runnerDevice,
  runnerRenderer,
  startLabel,
  starting,
  startNote,
}: {
  attemptSeedText: string;
  canStart: boolean;
  onAttemptSeedChange: (attemptSeedText: string) => void;
  onRandomizeAttemptSeed: () => void;
  onPolicyModeChange: (policyMode: PolicyPlaybackMode) => void;
  onRunnerDeviceChange: (device: WatchDevice) => void;
  onRunnerRendererChange: (renderer: WatchRenderer) => void;
  onStart: () => void;
  policyMode: PolicyPlaybackMode;
  rendererOptions: readonly WatchRenderer[];
  runnerDevice: WatchDevice;
  runnerRenderer: WatchRenderer;
  startLabel: string;
  starting: boolean;
  startNote: string;
}) {
  const disabled = startLabel === "Running" || starting;
  return (
    <div className="mb-5 grid gap-3 border border-app-border bg-app-surface px-3 py-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,48ch)] lg:items-end">
      <div className="grid gap-2 md:grid-cols-[104px_140px_140px_minmax(180px,240px)_max-content] md:items-end">
        <FieldShell>
          <span>Device</span>
          <FieldSelect
            aria-label="Career Mode device"
            disabled={disabled}
            value={runnerDevice}
            onChange={(event) => onRunnerDeviceChange(event.currentTarget.value as WatchDevice)}
          >
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Renderer</span>
          <FieldSelect
            aria-label="Career Mode renderer"
            disabled={disabled}
            value={runnerRenderer}
            onChange={(event) => onRunnerRendererChange(event.currentTarget.value as WatchRenderer)}
          >
            {rendererOptions.map((renderer) => (
              <option key={renderer} value={renderer}>
                {renderer}
              </option>
            ))}
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Mode</span>
          <FieldSelect
            aria-label="Career Mode initial policy mode"
            disabled={disabled}
            value={policyMode}
            onChange={(event) =>
              onPolicyModeChange(event.currentTarget.value as PolicyPlaybackMode)
            }
          >
            <option value="deterministic">deterministic</option>
            <option value="stochastic">stochastic</option>
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Runtime seed</span>
          <span className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
            <FieldInput
              aria-label="Career Mode runtime seed"
              disabled={disabled}
              inputMode="numeric"
              value={attemptSeedText}
              onChange={(event) => onAttemptSeedChange(event.currentTarget.value)}
            />
            <TooltipIconButton
              aria-label="Randomize runtime seed"
              disabled={disabled}
              tooltip="Randomize runtime seed"
              onClick={onRandomizeAttemptSeed}
            >
              <RandomizeIcon />
            </TooltipIconButton>
          </span>
        </FieldShell>
        <Button
          className="w-fit gap-2 px-5"
          disabled={!canStart || starting}
          type="button"
          variant="primary"
          onClick={onStart}
        >
          <PlayIcon />
          <span>{starting ? "Opening" : startLabel}</span>
        </Button>
      </div>
      <span className="text-xs text-app-muted lg:pb-3">{startNote}</span>
    </div>
  );
}

function CreateSaveGameForm({
  error,
  isCreating,
  onCreateSaveGame,
  onPatchSession,
  session,
}: {
  error: string | null;
  isCreating: boolean;
  onCreateSaveGame: () => void;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  session: SaveGameSession;
}) {
  return (
    <section className="grid gap-4 border border-app-border bg-app-surface p-5">
      {error !== null ? <FloatingNotice tone="error">{error}</FloatingNotice> : null}
      <div className="grid gap-4 md:grid-cols-[minmax(280px,1fr)_auto] md:items-end">
        <FieldShell>
          <span>Name</span>
          <FieldInput
            aria-label="Save game name"
            value={session.nameText}
            onChange={(event) =>
              onPatchSession(session.sessionId, { nameText: event.currentTarget.value })
            }
          />
        </FieldShell>
        <Button disabled={isCreating} variant="primary" onClick={onCreateSaveGame}>
          {isCreating ? "Creating" : "Create"}
        </Button>
      </div>
    </section>
  );
}

function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 border border-app-border bg-app-surface-muted px-3 py-2">
      <div className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</div>
      <div className="mt-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-sm font-semibold text-app-text">
        {value}
      </div>
    </div>
  );
}

function DetailRow({
  copyLabel,
  label,
  monospace = false,
  onCopy,
  value,
}: {
  copyLabel?: string;
  label: string;
  monospace?: boolean;
  onCopy?: () => void;
  value: string;
}) {
  return (
    <div className="grid gap-1">
      <dt className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</dt>
      <dd className="m-0 grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
        <span
          className={`min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-app-text ${
            monospace ? "font-mono text-xs" : ""
          }`}
          title={value}
        >
          {value}
        </span>
        {onCopy === undefined ? null : (
          <TooltipIconButton
            aria-label={copyLabel ?? `Copy ${label.toLowerCase()}`}
            size="compact"
            tooltip={copyLabel ?? `Copy ${label.toLowerCase()}`}
            onClick={onCopy}
          >
            <CopyIcon />
          </TooltipIconButton>
        )}
      </dd>
    </div>
  );
}

function resolveSavedCourseSetup(
  setups: readonly ManagedSaveCourseSetup[],
  target: ManagedSaveUnlockTarget,
  courses: ConfigMetadata["built_in_courses"],
): ManagedSaveCourseSetup | null {
  if (target.course_id === null && target.cup_id !== null) {
    const cupCourses = courses
      .filter((course) => course.cup === target.cup_id)
      .sort((left, right) => left.course_index - right.course_index);
    if (cupCourses.length > 0) {
      const resolvedSetups = cupCourses.map((course) =>
        resolveSavedCourseSetupForCourse(setups, {
          ...target,
          course_id: course.id,
        }),
      );
      return resolvedSetups.every((setup) => setup !== null) ? (resolvedSetups[0] ?? null) : null;
    }
  }
  return resolveSavedCourseSetupForCourse(setups, target);
}

function resolveSavedCourseSetupForCourse(
  setups: readonly ManagedSaveCourseSetup[],
  target: ManagedSaveUnlockTarget,
): ManagedSaveCourseSetup | null {
  return (
    setups.find(
      (setup) =>
        setup.scope === "course" &&
        setup.course_id === target.course_id &&
        optionalMatch(setup.cup_id, target.cup_id) &&
        optionalMatch(setup.difficulty, target.difficulty),
    ) ??
    // Cup-scoped setup rows remain valid persisted data for cup unlock targets.
    // Course rows win; the course setup editor writes course rows for bulk edits.
    setups.find(
      (setup) =>
        setup.scope === "cup" &&
        setup.cup_id === target.cup_id &&
        optionalMatch(setup.difficulty, target.difficulty),
    ) ??
    setups.find(
      (setup) => setup.scope === "difficulty" && setup.difficulty === target.difficulty,
    ) ??
    setups.find((setup) => setup.scope === "global") ??
    null
  );
}

function optionalMatch(expected: string | null, actual: string | null): boolean {
  return expected === null || expected === actual;
}

function parseAttemptSeed(value: string): string | "invalid" | null {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return null;
  }
  if (!/^\d+$/.test(trimmed)) {
    return "invalid";
  }
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) && parsed >= 0 && parsed <= 0xffffffff
    ? String(parsed)
    : "invalid";
}

function randomAttemptSeedText(): string {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return String(values[0]);
}
