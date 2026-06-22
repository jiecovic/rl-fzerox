// web/run-manager/src/pages/evaluations/presets.ts

import { allBuiltInCourseIds } from "@/entities/runConfig/ui/sections/tracks/coursePoolModel";
import type {
  ConfigMetadata,
  EvaluationMode,
  ManagedRunConfig,
  ManagedRunDetail,
} from "@/shared/api/contract";

export type EvaluationPresetId = "time_attack_blue_falcon" | "gp_blue_falcon" | "source_run";

export interface EvaluationPreset {
  cacheKey: string;
  id: EvaluationPresetId;
  label: string;
  config: ManagedRunConfig;
}

export interface EvaluationTargetDraft {
  courseIds: string[];
  cupIds: string[];
  difficulties: string[];
  mode: EvaluationMode;
  vehicleIds: string[];
}

export function buildEvaluationPresets({
  defaultConfig,
  metadata,
  sourceRun,
}: {
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  sourceRun: ManagedRunDetail | null;
}): EvaluationPreset[] {
  const allCourses = allBuiltInCourseIds(metadata);
  const blueFalconId = metadata.vehicles.some((vehicle) => vehicle.id === "blue_falcon")
    ? "blue_falcon"
    : (metadata.vehicles[0]?.id ?? defaultConfig.vehicle.selected_vehicle_ids[0]);
  const allDifficulties = metadata.gp_difficulties.map(
    (difficulty) => difficulty.value as ManagedRunConfig["tracks"]["gp_difficulties"][number],
  );
  const presets: EvaluationPreset[] = [
    {
      cacheKey: "time_attack_blue_falcon",
      id: "time_attack_blue_falcon",
      label: "Time Attack · Blue Falcon · all courses",
      config: {
        ...clonePresetConfig(defaultConfig),
        tracks: {
          ...defaultConfig.tracks,
          baseline_variant_count: 1,
          gp_difficulties: [],
          include_x_cup: false,
          race_mode: "time_attack",
          selected_course_ids: allCourses,
        },
        vehicle: {
          ...defaultConfig.vehicle,
          selected_vehicle_ids: [blueFalconId],
          selection_mode: "fixed",
        },
      },
    },
    {
      cacheKey: "gp_blue_falcon",
      id: "gp_blue_falcon",
      label: "GP Race · Blue Falcon · all cups",
      config: {
        ...clonePresetConfig(defaultConfig),
        tracks: {
          ...defaultConfig.tracks,
          gp_difficulties: allDifficulties.length > 0 ? allDifficulties : ["novice"],
          include_x_cup: false,
          race_mode: "gp_race",
          selected_course_ids: allCourses,
        },
        vehicle: {
          ...defaultConfig.vehicle,
          selected_vehicle_ids: [blueFalconId],
          selection_mode: "fixed",
        },
      },
    },
  ];
  if (sourceRun !== null) {
    presets.push({
      cacheKey: `source_run:${sourceRun.id}:${sourceRun.config_hash}`,
      id: "source_run",
      label: "Source run config",
      config: clonePresetConfig(sourceRun.config),
    });
  }
  return presets;
}

export function clonePresetConfig(config: ManagedRunConfig): ManagedRunConfig {
  return structuredClone(config) as ManagedRunConfig;
}

export function evaluationTargetFromConfig(
  config: ManagedRunConfig,
  metadata: ConfigMetadata,
): EvaluationTargetDraft {
  const mode: EvaluationMode = config.tracks.race_mode === "gp_race" ? "gp_cup" : "time_attack";
  const courseIds = config.tracks.selected_course_ids;
  return {
    courseIds: [...courseIds],
    cupIds: mode === "gp_cup" ? cupIdsForCourses(courseIds, metadata) : [],
    difficulties: mode === "gp_cup" ? [...config.tracks.gp_difficulties] : [],
    mode,
    vehicleIds: [...config.vehicle.selected_vehicle_ids],
  };
}

function cupIdsForCourses(courseIds: readonly string[], metadata: ConfigMetadata) {
  const selected = new Set(courseIds);
  return metadata.track_cups
    .filter((cup) => cup.course_ids.some((courseId) => selected.has(courseId)))
    .map((cup) => cup.id);
}
