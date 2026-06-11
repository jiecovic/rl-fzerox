// web/run-manager/src/features/saveGameCourseSetup/ui/UnlockPathPanel.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { TrackCupBanner } from "@/entities/runConfig/ui/sections/tracks/TrackCupBanner";
import { formatUnlockTargetStatus, unlockTargetStatusClass } from "@/entities/saveGame/model";
import type {
  CourseSetupDraftMap,
  CourseSetupValues,
  CupSetupDraftMap,
  CupSetupValues,
  CupView,
  PolicyArtifactDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  countDirtyCourseSetups,
  countDirtyCupSetups,
  courseSetupDraftsFromSavedSetups,
  courseSetupKey,
  cupSetupDraftsFromSavedSetups,
  cupSetupKey,
  cupsWithCourses,
  dirtyCourseSetupDrafts,
  dirtyCupSetupDrafts,
  EMPTY_COURSE_SETUP_DRAFT,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  CourseSetupPanel,
  GlobalPolicyPanel,
} from "@/features/saveGameCourseSetup/ui/CourseSetupPanels";
import type {
  ConfigMetadata,
  ManagedRun,
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
  SavePolicyArtifact,
} from "@/shared/api/contract";

interface UnlockPathPanelProps {
  assignableRuns: readonly ManagedRun[];
  metadata: ConfigMetadata;
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
  onCourseSetupDirtyChange: (dirty: boolean) => void;
  canStartTarget: (target: ManagedSaveUnlockTarget) => boolean;
  onStartTarget: (target: ManagedSaveUnlockTarget) => void;
  saveGame: ManagedSaveGame;
  updating: boolean;
}

type PolicySelectionDraft = Pick<PolicyArtifactDraft, "policyArtifact" | "policyRunId">;

export function UnlockPathPanel({
  assignableRuns,
  canStartTarget,
  metadata,
  onCourseSetupDirtyChange,
  onStartTarget,
  onUpsertCourseSetup,
  onUpsertCupSetup,
  saveGame,
  updating,
}: UnlockPathPanelProps) {
  const targets = saveGame.unlock_progress?.targets ?? [];
  const cups = useMemo(() => cupsWithCourses(metadata), [metadata]);
  const savedCourseSetupDrafts = useMemo(
    () => courseSetupDraftsFromSavedSetups(saveGame.course_setups),
    [saveGame.course_setups],
  );
  const savedCupSetupDrafts = useMemo(
    () => cupSetupDraftsFromSavedSetups(saveGame.cup_setups),
    [saveGame.cup_setups],
  );
  const [courseSetupDrafts, setCourseSetupDrafts] = useState<CourseSetupDraftMap>(
    () => savedCourseSetupDrafts,
  );
  const [cupSetupDrafts, setCupSetupDrafts] = useState<CupSetupDraftMap>(() => savedCupSetupDrafts);
  const previousSavedCourseSetupDrafts = useRef<CourseSetupDraftMap>(savedCourseSetupDrafts);
  const previousSavedCupSetupDrafts = useRef<CupSetupDraftMap>(savedCupSetupDrafts);
  const [savingCourseSetups, setSavingSetups] = useState(false);

  useEffect(() => {
    const previousSavedDrafts = previousSavedCourseSetupDrafts.current;
    setCourseSetupDrafts((current) => {
      const next = { ...savedCourseSetupDrafts };
      for (const dirtyDraft of dirtyCourseSetupDrafts(current, previousSavedDrafts)) {
        next[courseSetupKey(dirtyDraft)] = dirtyDraft;
      }
      return next;
    });
    previousSavedCourseSetupDrafts.current = savedCourseSetupDrafts;
  }, [savedCourseSetupDrafts]);

  useEffect(() => {
    const previousSavedDrafts = previousSavedCupSetupDrafts.current;
    setCupSetupDrafts((current) => {
      const next = { ...savedCupSetupDrafts };
      for (const dirtyDraft of dirtyCupSetupDrafts(current, previousSavedDrafts)) {
        next[cupSetupKey(dirtyDraft)] = dirtyDraft;
      }
      return next;
    });
    previousSavedCupSetupDrafts.current = savedCupSetupDrafts;
  }, [savedCupSetupDrafts]);

  const dirtyCourseSetupCount = countDirtyCourseSetups(courseSetupDrafts, savedCourseSetupDrafts);
  const dirtyCupSetupCount = countDirtyCupSetups(cupSetupDrafts, savedCupSetupDrafts);
  const dirtySetupCount = dirtyCourseSetupCount + dirtyCupSetupCount;
  const courseSetupsDirty = dirtySetupCount > 0;

  useEffect(() => {
    onCourseSetupDirtyChange(courseSetupsDirty);
  }, [courseSetupsDirty, onCourseSetupDirtyChange]);

  function updateCourseSetupDraft(values: CourseSetupValues, draft: PolicyArtifactDraft) {
    setCourseSetupDrafts((current) => ({
      ...current,
      [courseSetupKey(values)]: {
        ...values,
        engineSettingRawValue: draft.engineSettingRawValue,
        policyArtifact: draft.policyArtifact,
        policyRunId: draft.policyRunId,
        vehicleId: draft.vehicleId,
      },
    }));
  }

  function updateCupSetupDraft(values: CupSetupValues, vehicleId: string) {
    setCupSetupDrafts((current) => ({
      ...current,
      [cupSetupKey(values)]: {
        ...values,
        vehicleId,
      },
    }));
  }

  function applyCourseSetupDrafts(
    setups: readonly CourseSetupValues[],
    draft: PolicySelectionDraft,
  ) {
    if (draft.policyRunId === "") {
      return;
    }
    setCourseSetupDrafts((current) => {
      const next = { ...current };
      for (const setup of setups) {
        const currentDraft = current[courseSetupKey(setup)] ?? {
          ...setup,
          ...EMPTY_COURSE_SETUP_DRAFT,
        };
        next[courseSetupKey(setup)] = {
          ...setup,
          engineSettingRawValue: currentDraft.engineSettingRawValue,
          policyArtifact: draft.policyArtifact,
          policyRunId: draft.policyRunId,
          vehicleId: currentDraft.vehicleId,
        };
      }
      return next;
    });
  }

  async function saveCourseSetups() {
    if (dirtySetupCount === 0 || savingCourseSetups) {
      return;
    }
    setSavingSetups(true);
    try {
      for (const draft of dirtyCourseSetupDrafts(courseSetupDrafts, savedCourseSetupDrafts)) {
        if (draft.policyRunId === "") {
          continue;
        }
        await onUpsertCourseSetup({
          courseId: draft.courseId ?? null,
          cupId: draft.cupId ?? null,
          difficulty: draft.difficulty ?? null,
          engineSettingRawValue: draft.engineSettingRawValue,
          policyArtifact: draft.policyArtifact,
          policyRunId: draft.policyRunId,
          saveGameId: saveGame.id,
        });
      }
      for (const draft of dirtyCupSetupDrafts(cupSetupDrafts, savedCupSetupDrafts)) {
        await onUpsertCupSetup({
          cupId: draft.cupId,
          difficulty: draft.difficulty ?? null,
          saveGameId: saveGame.id,
          vehicleId: draft.vehicleId,
        });
      }
    } finally {
      setSavingSetups(false);
    }
  }

  return (
    <section className="grid gap-5 border border-app-border bg-app-surface p-5">
      <div className="grid gap-1">
        <h3 className="m-0 text-lg font-bold text-app-text">Unlock path</h3>
        <p className="m-0 text-sm text-app-muted">
          Game-rule order for GP cup clears. Progress is read from the save file.
        </p>
      </div>
      <TargetMatrix
        canStartTarget={canStartTarget}
        cups={cups}
        metadata={metadata}
        targets={targets}
        onStartTarget={onStartTarget}
      />
      <GlobalPolicyPanel
        assignableRuns={assignableRuns}
        cups={cups}
        updating={updating || savingCourseSetups}
        onApplySetups={applyCourseSetupDrafts}
      />
      <CourseSetupPanel
        assignableRuns={assignableRuns}
        cups={cups}
        dirtySetupCount={dirtySetupCount}
        courseSetupDrafts={courseSetupDrafts}
        metadata={metadata}
        savingCourseSetups={savingCourseSetups}
        cupSetupDrafts={cupSetupDrafts}
        updating={updating}
        unlockedVehicleIds={saveGame.unlock_progress?.unlocked_vehicle_ids ?? []}
        onCourseSetupDraftChange={updateCourseSetupDraft}
        onCupSetupDraftChange={updateCupSetupDraft}
        onSaveSetups={() => void saveCourseSetups()}
      />
    </section>
  );
}

function TargetMatrix({
  canStartTarget,
  cups,
  metadata,
  onStartTarget,
  targets,
}: {
  canStartTarget: (target: ManagedSaveUnlockTarget) => boolean;
  cups: readonly CupView[];
  metadata: ConfigMetadata;
  onStartTarget: (target: ManagedSaveUnlockTarget) => void;
  targets: readonly ManagedSaveUnlockTarget[];
}) {
  const difficulties = metadata.gp_difficulties;
  return (
    <div className="grid gap-2">
      {difficulties.map((difficulty) => (
        <div
          key={difficulty.value}
          className="grid gap-2 md:grid-cols-[120px_repeat(4,minmax(110px,1fr))]"
        >
          <div className="flex items-center text-sm font-semibold text-app-muted">
            {difficulty.label}
          </div>
          {cups.map((cup) => {
            const target = targets.find(
              (candidate) =>
                candidate.difficulty === difficulty.value && candidate.cup_id === cup.id,
            );
            const status = target?.status ?? "pending";
            const startable = target !== undefined && canStartTarget(target);
            return (
              <button
                key={`${difficulty.value}:${cup.id}`}
                aria-label={
                  target === undefined
                    ? `${difficulty.label} ${cup.label} not targeted`
                    : startable
                      ? `Start ${target.label}`
                      : target.label
                }
                className={`grid min-w-0 gap-2 border p-2 text-left transition ${
                  startable ? "cursor-pointer hover:border-app-accent" : "cursor-default"
                } ${unlockTargetStatusClass(status)}`}
                disabled={!startable}
                type="button"
                onClick={() => {
                  if (target !== undefined && startable) {
                    onStartTarget(target);
                  }
                }}
              >
                <div className="flex items-center gap-2">
                  <TrackCupBanner cupId={cup.id} label={cup.label} />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-app-text">{cup.label}</div>
                    <div className="text-xs text-app-muted">{targetStatusLabel(target)}</div>
                  </div>
                </div>
                <div className="h-1 overflow-hidden bg-app-surface">
                  <div
                    className={`h-full ${
                      status === "succeeded" ? "bg-emerald-300" : "bg-app-accent"
                    }`}
                    style={{ width: `${targetCompletionPercent(target)}%` }}
                  />
                </div>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function targetStatusLabel(target: ManagedSaveUnlockTarget | undefined): string {
  return target === undefined ? "not targeted" : formatUnlockTargetStatus(target.status);
}

function targetCompletionPercent(target: ManagedSaveUnlockTarget | undefined): number {
  if (target === undefined) {
    return 0;
  }
  return target.status === "succeeded" ? 100 : 0;
}
