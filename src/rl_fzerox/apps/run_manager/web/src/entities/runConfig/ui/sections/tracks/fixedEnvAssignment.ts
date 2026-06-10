// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/tracks/fixedEnvAssignment.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

export interface FixedEnvAssignmentSummary {
  activeCourseCount: number;
  envsPerCourse: number | null;
  issue: string | null;
}

export function fixedEnvAssignmentSummary(config: ManagedRunConfig): FixedEnvAssignmentSummary {
  const activeCourseCount =
    config.tracks.selected_course_ids.length +
    (config.tracks.include_x_cup ? config.tracks.x_cup_course_count : 0);
  const numEnvs = config.train.num_envs;
  if (activeCourseCount <= 0) {
    return {
      activeCourseCount,
      envsPerCourse: null,
      issue: "Fixed env assignment needs at least one active course.",
    };
  }
  if (numEnvs < activeCourseCount) {
    return {
      activeCourseCount,
      envsPerCourse: null,
      issue: `${numEnvs} envs cannot cover ${activeCourseCount} active courses.`,
    };
  }
  if (numEnvs % activeCourseCount !== 0) {
    return {
      activeCourseCount,
      envsPerCourse: null,
      issue: `${numEnvs} envs cannot be evenly split across ${activeCourseCount} courses.`,
    };
  }
  return {
    activeCourseCount,
    envsPerCourse: numEnvs / activeCourseCount,
    issue: null,
  };
}

export function fixedEnvAssignmentIssue(config: ManagedRunConfig): string | null {
  if (config.tracks.sampling_mode !== "fixed_env") {
    return null;
  }
  return fixedEnvAssignmentSummary(config).issue;
}
