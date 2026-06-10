// src/rl_fzerox/apps/run_manager/web/src/features/saveGameCourseSetup/model/courseSetup.ts
import type {
  ConfigMetadata,
  CourseSetupScope,
  ManagedRun,
  ManagedSaveCourseSetup,
  ManagedSaveUnlockTarget,
  SavePolicyArtifact,
} from "@/shared/api/contract";

export interface CupView {
  courses: ConfigMetadata["built_in_courses"];
  id: string;
  label: string;
  order: number;
}

export type CourseSetupScopeValues = {
  courseId?: string | null;
  cupId?: string | null;
  difficulty?: string | null;
  scope: CourseSetupScope;
};

export type PolicyArtifactDraft = {
  engineSettingRawValue: number;
  policyArtifact: SavePolicyArtifact;
  policyRunId: string;
  vehicleId: string;
};

export type CourseSetupDraft = CourseSetupScopeValues & PolicyArtifactDraft;
export type CourseSetupDraftMap = Record<string, CourseSetupDraft>;

export const EMPTY_COURSE_SETUP_DRAFT: PolicyArtifactDraft = {
  engineSettingRawValue: 50,
  policyArtifact: "best",
  policyRunId: "",
  vehicleId: "blue_falcon",
};

export function cupsWithCourses(metadata: ConfigMetadata): CupView[] {
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

export function courseSetupsForCups(cups: readonly CupView[]): CourseSetupScopeValues[] {
  return cups.flatMap((cup) => cup.courses.map((course) => courseSetupScopeValues(cup, course.id)));
}

export function cupSetupScopeValues(cup: CupView): CourseSetupScopeValues {
  return {
    courseId: null,
    cupId: cup.id,
    scope: "cup",
  };
}

export function courseSetupScopeValues(cup: CupView, courseId: string): CourseSetupScopeValues {
  return {
    courseId,
    cupId: cup.id,
    scope: "course",
  };
}

export function courseSetupDraftsFromSavedSetups(
  setups: readonly ManagedSaveCourseSetup[],
): CourseSetupDraftMap {
  const drafts: CourseSetupDraftMap = {};
  for (const setup of setups) {
    const scopeValues = scopeValuesFromSetup(setup);
    drafts[courseSetupKey(scopeValues)] = {
      ...scopeValues,
      engineSettingRawValue: setup.engine_setting_raw_value,
      policyArtifact: setup.policy_artifact,
      policyRunId: setup.policy_run_id,
      vehicleId: setup.vehicle_id,
    };
  }
  return drafts;
}

export function exactCourseSetupDraft(
  courseSetupDrafts: CourseSetupDraftMap,
  scopeValues: CourseSetupScopeValues,
): CourseSetupDraft | null {
  return courseSetupDrafts[courseSetupKey(scopeValues)] ?? null;
}

export function sharedCourseDraft(
  drafts: readonly CourseSetupDraft[],
  expectedCount: number,
): CourseSetupDraft | null {
  if (expectedCount === 0 || drafts.length !== expectedCount) {
    return null;
  }
  const [firstDraft] = drafts;
  if (firstDraft === undefined) {
    return null;
  }
  return drafts.every((draft) => policyArtifactDraftsEqual(draft, firstDraft)) ? firstDraft : null;
}

export function countDirtyCourseSetups(
  current: CourseSetupDraftMap,
  saved: CourseSetupDraftMap,
): number {
  return dirtyCourseSetupDrafts(current, saved).length;
}

export function dirtyCourseSetupDrafts(
  current: CourseSetupDraftMap,
  saved: CourseSetupDraftMap,
): CourseSetupDraft[] {
  return Object.values(current).filter((draft) => {
    const savedDraft = saved[courseSetupKey(draft)] ?? null;
    return (
      draft.policyRunId !== "" &&
      (savedDraft === null ||
        savedDraft.policyRunId !== draft.policyRunId ||
        savedDraft.policyArtifact !== draft.policyArtifact ||
        savedDraft.vehicleId !== draft.vehicleId ||
        savedDraft.engineSettingRawValue !== draft.engineSettingRawValue)
    );
  });
}

export function preferredVehicleSetup({
  currentDraft,
  metadata,
  run,
  unlockedVehicleIds,
}: {
  currentDraft: PolicyArtifactDraft;
  metadata: ConfigMetadata;
  run: ManagedRun | null;
  unlockedVehicleIds: readonly string[];
}): Pick<PolicyArtifactDraft, "engineSettingRawValue" | "vehicleId"> {
  const unlockedVehicleSet = new Set(unlockedVehicleIds);
  const fallbackVehicleId = unlockedVehicleSet.has("blue_falcon")
    ? "blue_falcon"
    : (unlockedVehicleIds[0] ?? metadata.vehicles[0]?.id ?? currentDraft.vehicleId);
  if (run === null) {
    return {
      engineSettingRawValue: currentDraft.engineSettingRawValue,
      vehicleId: currentDraft.vehicleId,
    };
  }
  const trainedVehicleId = run.vehicle_setup.selected_vehicle_ids[0] ?? null;
  const vehicleId =
    trainedVehicleId !== null && unlockedVehicleSet.has(trainedVehicleId)
      ? trainedVehicleId
      : fallbackVehicleId;
  return {
    engineSettingRawValue: preferredEngineSetting(run),
    vehicleId,
  };
}

export function resolveSavedCourseSetup(
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

export function courseSetupKey(scopeValues: CourseSetupScopeValues): string {
  return [
    scopeValues.scope,
    scopeValues.difficulty ?? "",
    scopeValues.cupId ?? "",
    scopeValues.courseId ?? "",
  ].join(":");
}

function scopeValuesFromSetup(setup: ManagedSaveCourseSetup): CourseSetupScopeValues {
  return {
    courseId: setup.course_id,
    cupId: setup.cup_id,
    difficulty: setup.difficulty,
    scope: setup.scope,
  };
}

function policyArtifactDraftsEqual(left: PolicyArtifactDraft, right: PolicyArtifactDraft): boolean {
  return (
    left.policyRunId === right.policyRunId &&
    left.policyArtifact === right.policyArtifact &&
    left.vehicleId === right.vehicleId &&
    left.engineSettingRawValue === right.engineSettingRawValue
  );
}

function preferredEngineSetting(run: ManagedRun): number {
  const vehicle = run.vehicle_setup;
  if (vehicle.engine_mode === "fixed") {
    return vehicle.engine_setting_raw_value;
  }
  if (vehicle.engine_setting_min_raw_value === vehicle.engine_setting_max_raw_value) {
    return vehicle.engine_setting_min_raw_value;
  }
  return 50;
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
