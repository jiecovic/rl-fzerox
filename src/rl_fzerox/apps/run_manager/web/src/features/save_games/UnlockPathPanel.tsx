// src/rl_fzerox/apps/run_manager/web/src/features/save_games/UnlockPathPanel.tsx
import { useEffect, useMemo, useState } from "react";

import { TrackCupBanner } from "@/features/configurator/sections/tracks/TrackCupBanner";
import { formatUnlockTargetStatus, unlockTargetStatusClass } from "@/features/save_games/model";
import type {
  CourseSetupDraftMap,
  CourseSetupScopeValues,
  CupView,
  PolicyArtifactDraft,
} from "@/features/save_games/unlock_path/courseSetup";
import {
  CourseSetupPanel,
  countDirtyCourseSetups,
  courseSetupDraftsFromSavedSetups,
  courseSetupKey,
  dirtyCourseSetupDrafts,
  GlobalPolicyPanel,
} from "@/features/save_games/unlock_path/courseSetup";
import type {
  ConfigMetadata,
  CourseSetupScope,
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
    policyArtifact: SavePolicyArtifact;
    policyRunId: string;
    saveGameId: string;
    scope: CourseSetupScope;
  }) => Promise<ManagedSaveGame>;
  onCourseSetupDirtyChange: (dirty: boolean) => void;
  saveGame: ManagedSaveGame;
  updating: boolean;
}

export function UnlockPathPanel({
  assignableRuns,
  metadata,
  onCourseSetupDirtyChange,
  onUpsertCourseSetup,
  saveGame,
  updating,
}: UnlockPathPanelProps) {
  const targets = saveGame.unlock_progress?.targets ?? [];
  const cups = useMemo(() => cupsWithCourses(metadata), [metadata]);
  const savedCourseSetupDrafts = useMemo(
    () => courseSetupDraftsFromSavedSetups(saveGame.course_setups),
    [saveGame.course_setups],
  );
  const [courseSetupDrafts, setCourseSetupDrafts] = useState<CourseSetupDraftMap>(
    () => savedCourseSetupDrafts,
  );
  const [savingCourseSetups, setSavingSetups] = useState(false);

  useEffect(() => {
    setCourseSetupDrafts(savedCourseSetupDrafts);
  }, [savedCourseSetupDrafts]);

  const dirtyCourseSetupCount = countDirtyCourseSetups(courseSetupDrafts, savedCourseSetupDrafts);
  const courseSetupsDirty = dirtyCourseSetupCount > 0;

  useEffect(() => {
    onCourseSetupDirtyChange(courseSetupsDirty);
  }, [courseSetupsDirty, onCourseSetupDirtyChange]);

  function updateCourseSetupDraft(scopeValues: CourseSetupScopeValues, draft: PolicyArtifactDraft) {
    setCourseSetupDrafts((current) => ({
      ...current,
      [courseSetupKey(scopeValues)]: {
        ...scopeValues,
        policyArtifact: draft.policyArtifact,
        policyRunId: draft.policyRunId,
      },
    }));
  }

  function applyCourseSetupDrafts(
    setups: readonly CourseSetupScopeValues[],
    draft: PolicyArtifactDraft,
  ) {
    if (draft.policyRunId === "") {
      return;
    }
    setCourseSetupDrafts((current) => {
      const next = { ...current };
      for (const setup of setups) {
        next[courseSetupKey(setup)] = {
          ...setup,
          policyArtifact: draft.policyArtifact,
          policyRunId: draft.policyRunId,
        };
      }
      return next;
    });
  }

  async function saveCourseSetups() {
    if (dirtyCourseSetupCount === 0 || savingCourseSetups) {
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
          policyArtifact: draft.policyArtifact,
          policyRunId: draft.policyRunId,
          saveGameId: saveGame.id,
          scope: draft.scope,
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
      <TargetMatrix cups={cups} metadata={metadata} targets={targets} />
      <GlobalPolicyPanel
        assignableRuns={assignableRuns}
        cups={cups}
        updating={updating || savingCourseSetups}
        onApplySetups={applyCourseSetupDrafts}
      />
      <CourseSetupPanel
        assignableRuns={assignableRuns}
        cups={cups}
        dirtyCourseSetupCount={dirtyCourseSetupCount}
        courseSetupDrafts={courseSetupDrafts}
        savingCourseSetups={savingCourseSetups}
        updating={updating}
        onApplySetups={applyCourseSetupDrafts}
        onCourseSetupDraftChange={updateCourseSetupDraft}
        onSaveSetups={() => void saveCourseSetups()}
      />
    </section>
  );
}

function TargetMatrix({
  cups,
  metadata,
  targets,
}: {
  cups: readonly CupView[];
  metadata: ConfigMetadata;
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
            const locked = isInitiallyLockedCup(cup.id);
            return (
              <div
                key={`${difficulty.value}:${cup.id}`}
                className={`grid min-w-0 gap-2 border p-2 ${
                  locked
                    ? "border-app-border bg-app-surface text-app-muted opacity-60"
                    : unlockTargetStatusClass(status)
                }`}
              >
                <div className="flex items-center gap-2">
                  <TrackCupBanner cupId={cup.id} label={cup.label} />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-app-text">{cup.label}</div>
                    <div className="text-xs text-app-muted">
                      {locked ? `locked · ${targetStatusLabel(target)}` : targetStatusLabel(target)}
                    </div>
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
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function cupsWithCourses(metadata: ConfigMetadata): CupView[] {
  return metadata.track_cups
    .filter((cup) => cup.id !== "x")
    .map((cup) => ({
      id: cup.id,
      label: cup.label,
      order: cup.order,
      courses: metadata.built_in_courses
        .filter((course) => course.cup === cup.id)
        .sort((left, right) => left.course_index - right.course_index),
    }))
    .sort((left, right) => left.order - right.order);
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

function isInitiallyLockedCup(cupId: string): boolean {
  return cupId === "joker";
}
