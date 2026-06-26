// web/run-manager/src/pages/evaluations/sections/presetConfig/formModel.ts
import {
  allBuiltInCourseIds,
  defaultSelectedCourseIds,
  orderedCourseIds,
} from "@/entities/runConfig/ui/sections/tracks/coursePoolModel";
import type { PresetFormState } from "@/pages/evaluations/sections/presetConfig/types";
import type {
  ConfigMetadata,
  CreateEvaluationPresetRequest,
  ManagedEvaluationPreset,
} from "@/shared/api/contract";

export const EVALUATION_PRESET_FORM_SETTINGS = {
  defaults: {
    baselineVariantCount: "1",
    gpBaselineVariantCount: "10",
    name: "Custom evaluation preset",
    repeatsPerTarget: "10",
    seed: "2262218583",
  },
  limits: {
    baselineVariantCount: 16,
    repeatsPerTarget: 1000,
    seed: 2 ** 32 - 1,
  },
} as const;

export function defaultPresetForm(metadata: ConfigMetadata): PresetFormState {
  const allCourseIds = allBuiltInCourseIds(metadata);
  return {
    courseIds: defaultSelectedCourseIds(metadata, allCourseIds),
    difficulties: [],
    name: EVALUATION_PRESET_FORM_SETTINGS.defaults.name,
    renderer: "gliden64",
    baselineVariantCount: EVALUATION_PRESET_FORM_SETTINGS.defaults.baselineVariantCount,
    repeatsPerTarget: EVALUATION_PRESET_FORM_SETTINGS.defaults.repeatsPerTarget,
    seed: EVALUATION_PRESET_FORM_SETTINGS.defaults.seed,
    targetMode: "time_attack_course",
  };
}

export function presetFormFromPreset(
  preset: ManagedEvaluationPreset,
  metadata: ConfigMetadata,
  options: { copyName?: boolean } = {},
): PresetFormState {
  return {
    courseIds: courseIdsFromPresetTarget(preset.target, metadata),
    difficulties: difficultyFromPresetTarget(preset, metadata),
    name: options.copyName === true ? `${preset.name} copy` : preset.name,
    renderer: preset.renderer,
    baselineVariantCount: String(preset.target.baseline_variant_count),
    repeatsPerTarget: String(preset.target.repeats_per_target),
    seed: String(preset.seed),
    targetMode: preset.target.mode,
  };
}

export function defaultGpDifficulties(metadata: ConfigMetadata) {
  const configuredMaster = metadata.gp_difficulties.find((option) => option.value === "master");
  const fallback = configuredMaster ?? metadata.gp_difficulties[0];
  return fallback === undefined ? [] : [fallback.value];
}

export function selectGpDifficulty(difficulty: string, metadata: ConfigMetadata) {
  const valid = metadata.gp_difficulties.some((option) => option.value === difficulty);
  if (valid) {
    return [difficulty];
  }
  return defaultGpDifficulties(metadata);
}

export function presetRequestFromForm(
  form: PresetFormState,
): CreateEvaluationPresetRequest | string {
  const name = form.name.trim();
  if (name.length === 0) {
    return "Preset name is required.";
  }
  const seed = Number(form.seed);
  if (!Number.isInteger(seed) || seed < 0 || seed > EVALUATION_PRESET_FORM_SETTINGS.limits.seed) {
    return "Seed must be an integer from 0 to 4294967295.";
  }
  const repeatsPerTarget = Number(form.repeatsPerTarget);
  if (
    !Number.isInteger(repeatsPerTarget) ||
    repeatsPerTarget < 1 ||
    repeatsPerTarget > EVALUATION_PRESET_FORM_SETTINGS.limits.repeatsPerTarget
  ) {
    return "Repeats must be an integer from 1 to 1000.";
  }
  const baselineVariantCount =
    form.targetMode === "gp_course" ? Number(form.baselineVariantCount) : 1;
  if (
    !Number.isInteger(baselineVariantCount) ||
    baselineVariantCount < 1 ||
    baselineVariantCount > EVALUATION_PRESET_FORM_SETTINGS.limits.baselineVariantCount
  ) {
    return `Race-start variants must be an integer from 1 to ${EVALUATION_PRESET_FORM_SETTINGS.limits.baselineVariantCount}.`;
  }
  if (form.courseIds.length === 0) {
    return "Select at least one course.";
  }
  if (form.targetMode === "gp_course" && form.difficulties.length !== 1) {
    return "Select exactly one GP difficulty.";
  }
  return {
    courseIds: [...form.courseIds],
    cupIds: [],
    difficulties: form.targetMode === "gp_course" ? [...form.difficulties] : [],
    name,
    renderer: form.renderer,
    baselineVariantCount,
    repeatsPerTarget,
    seed,
    targetMode: form.targetMode,
  };
}

function difficultyFromPresetTarget(preset: ManagedEvaluationPreset, metadata: ConfigMetadata) {
  if (preset.target.mode !== "gp_course") {
    return [];
  }
  const validDifficulties = new Set(metadata.gp_difficulties.map((option) => option.value));
  const firstDifficulty = preset.target.difficulties.find((difficulty) =>
    validDifficulties.has(difficulty),
  );
  return firstDifficulty === undefined ? defaultGpDifficulties(metadata) : [firstDifficulty];
}

function courseIdsFromPresetTarget(
  target: ManagedEvaluationPreset["target"],
  metadata: ConfigMetadata,
) {
  const allCourseIds = allBuiltInCourseIds(metadata);
  if (target.course_ids.length > 0) {
    return orderedCourseIds(target.course_ids, allCourseIds);
  }
  if (target.cup_ids.length > 0) {
    const cupIds = new Set(target.cup_ids);
    const selectedCourseIds = metadata.track_cups
      .filter((cup) => cupIds.has(cup.id))
      .flatMap((cup) => cup.course_ids);
    return orderedCourseIds(selectedCourseIds, allCourseIds);
  }
  return allCourseIds;
}
