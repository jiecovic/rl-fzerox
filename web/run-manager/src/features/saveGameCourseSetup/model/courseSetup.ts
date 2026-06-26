// web/run-manager/src/features/saveGameCourseSetup/model/courseSetup.ts
import type {
  CheckpointCatalogResponse,
  ConfigMetadata,
  ManagedEvaluation,
  ManagedRun,
  ManagedSaveCourseSetup,
  ManagedSaveCupSetup,
  ManagedSaveUnlockTarget,
  SavePolicyArtifact,
  SavePolicySourceKind,
} from "@/shared/api/contract";
import { enginePercentToSliderStep } from "@/shared/domain/engineBuckets";

export interface CupView {
  courses: ConfigMetadata["built_in_courses"];
  id: string;
  label: string;
  order: number;
}

export type CourseSetupValues = {
  courseId: string;
  cupId: string;
  difficulty?: string | null;
};

export type CupSetupValues = {
  cupId: string;
  difficulty?: string | null;
};

export type PolicyArtifactDraft = {
  engineSettingRawValue: number;
  policyArtifact: SavePolicyArtifact;
  policySourceId: string;
  policySourceKind: SavePolicySourceKind;
  vehicleId: string;
};

export type PolicySelectionDraft = Pick<
  PolicyArtifactDraft,
  "policyArtifact" | "policySourceId" | "policySourceKind"
>;
export type CourseSetupDraft = CourseSetupValues & PolicyArtifactDraft;
export type CourseSetupDraftMap = Record<string, CourseSetupDraft>;
export type CupSetupDraft = CupSetupValues & Pick<PolicyArtifactDraft, "vehicleId">;
export type CupSetupDraftMap = Record<string, CupSetupDraft>;
export type PolicySourceVehicleSetup = ManagedRun["vehicle_setup"];

export interface PolicySourceOption {
  artifact: SavePolicyArtifact;
  id: string;
  kind: SavePolicySourceKind;
  label: string;
  canImportEngineTuning: boolean;
  vehicleSetup: PolicySourceVehicleSetup;
}

export const NEUTRAL_ENGINE_SETTING_RAW_VALUE = enginePercentToSliderStep(50);

export const EMPTY_COURSE_SETUP_DRAFT: PolicyArtifactDraft = {
  engineSettingRawValue: NEUTRAL_ENGINE_SETTING_RAW_VALUE,
  policyArtifact: "best",
  policySourceId: "",
  policySourceKind: "run",
  vehicleId: "blue_falcon",
};

export const EMPTY_CUP_SETUP_DRAFT: Pick<PolicyArtifactDraft, "vehicleId"> = {
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

export function courseSetupsForCups(cups: readonly CupView[]): CourseSetupValues[] {
  return cups.flatMap((cup) => cup.courses.map((course) => courseSetupValues(cup, course.id)));
}

export function cupSetupValues(cup: CupView): CupSetupValues {
  return {
    cupId: cup.id,
  };
}

export function courseSetupValues(cup: CupView, courseId: string): CourseSetupValues {
  return {
    courseId,
    cupId: cup.id,
  };
}

export function courseSetupDraftsFromSavedSetups(
  setups: readonly ManagedSaveCourseSetup[],
): CourseSetupDraftMap {
  const drafts: CourseSetupDraftMap = {};
  for (const setup of setups) {
    const values = valuesFromSetup(setup);
    drafts[courseSetupKey(values)] = {
      ...values,
      engineSettingRawValue: setup.engine_setting_raw_value,
      policyArtifact: setup.policy_artifact,
      policySourceId: setup.policy_source_id,
      policySourceKind: setup.policy_source_kind,
      vehicleId: EMPTY_COURSE_SETUP_DRAFT.vehicleId,
    };
  }
  return drafts;
}

export function cupSetupDraftsFromSavedSetups(
  setups: readonly ManagedSaveCupSetup[],
): CupSetupDraftMap {
  const drafts: CupSetupDraftMap = {};
  for (const setup of setups) {
    const values: CupSetupValues = {
      cupId: setup.cup_id,
      difficulty: setup.difficulty,
    };
    drafts[cupSetupKey(values)] = {
      ...values,
      vehicleId: setup.vehicle_id,
    };
  }
  return drafts;
}

export function exactCourseSetupDraft(
  courseSetupDrafts: CourseSetupDraftMap,
  values: CourseSetupValues,
): CourseSetupDraft | null {
  return courseSetupDrafts[courseSetupKey(values)] ?? null;
}

export function sharedPolicySelectionDraft(
  drafts: readonly CourseSetupDraft[],
  expectedCount: number,
): PolicySelectionDraft | null {
  if (expectedCount === 0 || drafts.length !== expectedCount) {
    return null;
  }
  const [firstDraft] = drafts;
  if (firstDraft === undefined) {
    return null;
  }
  return drafts.every((draft) => policySelectionDraftsEqual(draft, firstDraft))
    ? {
        policyArtifact: firstDraft.policyArtifact,
        policySourceId: firstDraft.policySourceId,
        policySourceKind: firstDraft.policySourceKind,
      }
    : null;
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
      draft.policySourceId !== "" &&
      (savedDraft === null ||
        savedDraft.policySourceKind !== draft.policySourceKind ||
        savedDraft.policySourceId !== draft.policySourceId ||
        savedDraft.policyArtifact !== draft.policyArtifact ||
        savedDraft.engineSettingRawValue !== draft.engineSettingRawValue)
    );
  });
}

export function resetCourseEngineDrafts(
  current: CourseSetupDraftMap,
  setups: readonly CourseSetupValues[],
): CourseSetupDraftMap {
  const next = { ...current };
  for (const setup of setups) {
    const key = courseSetupKey(setup);
    const currentDraft = current[key] ?? {
      ...setup,
      ...EMPTY_COURSE_SETUP_DRAFT,
    };
    next[key] = {
      ...currentDraft,
      ...setup,
      engineSettingRawValue: NEUTRAL_ENGINE_SETTING_RAW_VALUE,
    };
  }
  return next;
}

export function countDirtyCupSetups(current: CupSetupDraftMap, saved: CupSetupDraftMap): number {
  return dirtyCupSetupDrafts(current, saved).length;
}

export function dirtyCupSetupDrafts(
  current: CupSetupDraftMap,
  saved: CupSetupDraftMap,
): CupSetupDraft[] {
  return Object.values(current).filter((draft) => {
    const savedDraft = saved[cupSetupKey(draft)] ?? null;
    return savedDraft === null || savedDraft.vehicleId !== draft.vehicleId;
  });
}

export function preferredVehicleSetup({
  currentDraft,
  metadata,
  source,
  unlockedVehicleIds,
}: {
  currentDraft: PolicyArtifactDraft;
  metadata: ConfigMetadata;
  source: PolicySourceOption | null;
  unlockedVehicleIds: readonly string[];
}): Pick<PolicyArtifactDraft, "engineSettingRawValue" | "vehicleId"> {
  const unlockedVehicleSet = new Set(unlockedVehicleIds);
  const fallbackVehicleId = unlockedVehicleSet.has("blue_falcon")
    ? "blue_falcon"
    : (unlockedVehicleIds[0] ?? metadata.vehicles[0]?.id ?? currentDraft.vehicleId);
  if (source === null) {
    return {
      engineSettingRawValue: currentDraft.engineSettingRawValue,
      vehicleId: currentDraft.vehicleId,
    };
  }
  const trainedVehicleId = source.vehicleSetup.selected_vehicle_ids[0] ?? null;
  const vehicleId =
    trainedVehicleId !== null && unlockedVehicleSet.has(trainedVehicleId)
      ? trainedVehicleId
      : fallbackVehicleId;
  return {
    engineSettingRawValue: preferredEngineSetting(source.vehicleSetup),
    vehicleId,
  };
}

export function policySourceOptions({
  checkpointCatalog,
  evaluations,
  runs,
}: {
  checkpointCatalog: CheckpointCatalogResponse | null;
  evaluations: readonly ManagedEvaluation[];
  runs: readonly ManagedRun[];
}): PolicySourceOption[] {
  return [
    ...runs
      .filter((run) => run.status !== "created" && run.status !== "archived")
      .map(
        (run): PolicySourceOption => ({
          artifact: "best",
          id: run.id,
          kind: "run",
          label: run.name,
          canImportEngineTuning: run.vehicle_setup.engine_mode === "adaptive_tuner",
          vehicleSetup: run.vehicle_setup,
        }),
      ),
    ...evaluations
      .filter((evaluation) => evaluation.status === "completed")
      .map(
        (evaluation): PolicySourceOption => ({
          artifact: evaluation.checkpoint.artifact,
          id: evaluation.id,
          kind: "evaluation",
          label: `${evaluation.name} · evaluation snapshot`,
          canImportEngineTuning: evaluation.config.vehicle.engine_mode === "adaptive_tuner",
          vehicleSetup: {
            selection_mode: evaluation.config.vehicle.selection_mode,
            selected_vehicle_ids: evaluation.config.vehicle.selected_vehicle_ids,
            engine_mode: evaluation.config.vehicle.engine_mode,
            engine_setting_raw_value: evaluation.config.vehicle.engine_setting_raw_value,
            engine_setting_min_raw_value: evaluation.config.vehicle.engine_setting_min_raw_value,
            engine_setting_max_raw_value: evaluation.config.vehicle.engine_setting_max_raw_value,
          },
        }),
      ),
    ...(checkpointCatalog?.installed_checkpoints ?? []).map(
      (checkpoint): PolicySourceOption => ({
        artifact: checkpoint.source_artifact,
        id: checkpoint.id,
        kind: "checkpoint",
        label: `${checkpoint.name} · release checkpoint`,
        canImportEngineTuning:
          checkpoint.has_engine_tuning_state &&
          checkpoint.config.vehicle.engine_mode === "adaptive_tuner",
        vehicleSetup: {
          selection_mode: checkpoint.config.vehicle.selection_mode,
          selected_vehicle_ids: checkpoint.config.vehicle.selected_vehicle_ids,
          engine_mode: checkpoint.config.vehicle.engine_mode,
          engine_setting_raw_value: checkpoint.config.vehicle.engine_setting_raw_value,
          engine_setting_min_raw_value: checkpoint.config.vehicle.engine_setting_min_raw_value,
          engine_setting_max_raw_value: checkpoint.config.vehicle.engine_setting_max_raw_value,
        },
      }),
    ),
  ];
}

export function policySourceKey(draft: PolicySelectionDraft): string {
  return draft.policySourceId === "" ? "" : `${draft.policySourceKind}:${draft.policySourceId}`;
}

export function policySourceOptionKey(source: PolicySourceOption): string {
  return `${source.kind}:${source.id}`;
}

export function selectedPolicySource(
  policySources: readonly PolicySourceOption[],
  draft: PolicySelectionDraft,
): PolicySourceOption | null {
  return (
    policySources.find(
      (source) => source.kind === draft.policySourceKind && source.id === draft.policySourceId,
    ) ?? null
  );
}

export function policySelectionDraftForSource(
  draft: PolicySelectionDraft,
  source: PolicySourceOption,
): PolicySelectionDraft {
  return {
    ...draft,
    policyArtifact:
      source.kind === "evaluation" || source.kind === "checkpoint"
        ? source.artifact
        : draft.policyArtifact,
    policySourceId: source.id,
    policySourceKind: source.kind,
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

export function courseSetupKey(values: CourseSetupValues): string {
  return [values.difficulty ?? "", values.cupId, values.courseId].join(":");
}

function valuesFromSetup(setup: ManagedSaveCourseSetup): CourseSetupValues {
  return {
    courseId: setup.course_id ?? "",
    cupId: setup.cup_id ?? "",
    difficulty: setup.difficulty,
  };
}

export function cupSetupKey(values: CupSetupValues): string {
  return [values.difficulty ?? "", values.cupId].join(":");
}

function policySelectionDraftsEqual(
  left: PolicySelectionDraft,
  right: PolicySelectionDraft,
): boolean {
  return (
    left.policySourceKind === right.policySourceKind &&
    left.policySourceId === right.policySourceId &&
    left.policyArtifact === right.policyArtifact
  );
}

export function preferredEngineSetting(vehicle: PolicySourceVehicleSetup): number {
  if (vehicle.engine_mode === "fixed") {
    return vehicle.engine_setting_raw_value;
  }
  return Math.round(
    (vehicle.engine_setting_min_raw_value + vehicle.engine_setting_max_raw_value) / 2,
  );
}

function resolveSavedCourseSetupForCourse(
  setups: readonly ManagedSaveCourseSetup[],
  target: ManagedSaveUnlockTarget,
): ManagedSaveCourseSetup | null {
  return (
    setups.find(
      (setup) =>
        setup.course_id === target.course_id &&
        optionalMatch(setup.cup_id, target.cup_id) &&
        optionalMatch(setup.difficulty, target.difficulty),
    ) ?? null
  );
}

export function resolveSavedCupSetup(
  setups: readonly ManagedSaveCupSetup[],
  target: ManagedSaveUnlockTarget,
): ManagedSaveCupSetup | null {
  if (target.cup_id === null) {
    return null;
  }
  return (
    setups.find(
      (setup) =>
        setup.cup_id === target.cup_id && optionalMatch(setup.difficulty, target.difficulty),
    ) ?? null
  );
}

function optionalMatch(expected: string | null, actual: string | null): boolean {
  return expected === null || expected === actual;
}
