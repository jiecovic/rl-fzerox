// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/tracks/coursePoolModel.ts

import type { ConfigMetadata } from "@/shared/api/contract";
import type { BuiltInCourse, TrackCupView } from "@/widgets/configurator/sections/tracks/types";

export function allBuiltInCourseIds(metadata: ConfigMetadata) {
  return metadata.built_in_courses.map((course) => course.id);
}

export function defaultSelectedCourseIds(metadata: ConfigMetadata, fallbackCourseIds: string[]) {
  const selected = metadata.built_in_courses
    .filter((course) => course.default_selected)
    .map((course) => course.id);
  return selected.length > 0 ? selected : fallbackCourseIds;
}

export function buildTrackCupViews(metadata: ConfigMetadata): TrackCupView[] {
  const courseById = new Map(metadata.built_in_courses.map((course) => [course.id, course]));
  return metadata.track_cups
    .slice()
    .sort((left, right) => left.order - right.order)
    .map((cup) => ({
      ...cup,
      courses: cup.course_ids
        .map((courseId) => courseById.get(courseId))
        .filter((course): course is BuiltInCourse => course !== undefined)
        .sort((left, right) => left.course_index - right.course_index)
        .map((course, index) => ({ ...course, cup_order_index: index + 1 })),
    }))
    .filter((cup) => cup.courses.length > 0);
}

export function orderedCourseIds(nextIds: Iterable<string>, orderedCourseIds: readonly string[]) {
  const nextSet = new Set(nextIds);
  return orderedCourseIds.filter((courseId) => nextSet.has(courseId));
}

export function arraysEqual(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
