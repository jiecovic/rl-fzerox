// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/TracksSection.tsx
import { useMemo } from "react";

import { ConfigStack } from "@/features/configurator/ConfigLayout";
import { usePersistentCollapsedIds } from "@/features/configurator/disclosureState";
import { CoursePoolPanel } from "@/features/configurator/sections/tracks/CoursePoolPanel";
import {
  allBuiltInCourseIds,
  buildTrackCupViews,
  defaultSelectedCourseIds,
  orderedCourseIds,
} from "@/features/configurator/sections/tracks/coursePoolModel";
import { X_CUP } from "@/features/configurator/sections/tracks/options";
import {
  CourseSamplingPanel,
  RaceSetupPanels,
} from "@/features/configurator/sections/tracks/RaceSamplingPanels";
import type {
  GpDifficulty,
  TracksConfig,
  TracksSectionProps,
} from "@/features/configurator/sections/tracks/types";

export function TracksSection({ config, defaultConfig, metadata, setConfig }: TracksSectionProps) {
  const defaultGpDifficulty: GpDifficulty = defaultConfig.tracks.gp_difficulty ?? "novice";
  const allCourseIds = useMemo(() => allBuiltInCourseIds(metadata), [metadata]);
  const defaultCourseIds = useMemo(
    () => defaultSelectedCourseIds(metadata, allCourseIds),
    [allCourseIds, metadata],
  );
  const cups = useMemo(() => buildTrackCupViews(metadata), [metadata]);
  const collapsibleCupIds = useMemo(() => [...cups.map((cup) => cup.id), X_CUP.id], [cups]);
  const [collapsedCupIds, setCollapsedCupIds] = usePersistentCollapsedIds(
    "run-manager:tracks:cups",
    collapsibleCupIds,
  );
  const selectedCourseIds = config.tracks.selected_course_ids;
  const selectedCourseSet = useMemo(() => new Set(selectedCourseIds), [selectedCourseIds]);
  const collapsedCupIdSet = useMemo(() => new Set(collapsedCupIds), [collapsedCupIds]);

  const normalizedTracks = (tracks: TracksConfig) => {
    const nextTracks = {
      ...tracks,
      x_cup_auto_regeneration: { ...tracks.x_cup_auto_regeneration },
    };
    if (nextTracks.race_mode === "gp_race" && nextTracks.gp_difficulty == null) {
      nextTracks.gp_difficulty = defaultGpDifficulty;
    }
    if (nextTracks.race_mode !== "gp_race") {
      nextTracks.gp_difficulty = null;
      nextTracks.include_x_cup = false;
      if (nextTracks.selected_course_ids.length === 0) {
        nextTracks.selected_course_ids = defaultCourseIds;
      }
    }
    if (!nextTracks.include_x_cup) {
      nextTracks.x_cup_auto_regeneration.enabled = false;
    }
    return nextTracks;
  };

  const updateTracks = (patch: Partial<TracksConfig>) => {
    setConfig((currentConfig) => ({
      ...currentConfig,
      tracks: normalizedTracks({ ...currentConfig.tracks, ...patch }),
    }));
  };

  const samplingDefaults = {
    adaptive_step_balance_completion_weight:
      defaultConfig.tracks.adaptive_step_balance_completion_weight,
    adaptive_step_balance_confidence_scale:
      defaultConfig.tracks.adaptive_step_balance_confidence_scale,
    adaptive_step_balance_min_confidence_episodes:
      defaultConfig.tracks.adaptive_step_balance_min_confidence_episodes,
    adaptive_step_balance_target_completion:
      defaultConfig.tracks.adaptive_step_balance_target_completion,
    step_balance_ema_alpha: defaultConfig.tracks.step_balance_ema_alpha,
    step_balance_max_weight_scale: defaultConfig.tracks.step_balance_max_weight_scale,
    step_balance_update_episodes: defaultConfig.tracks.step_balance_update_episodes,
  } satisfies Pick<
    TracksConfig,
    | "adaptive_step_balance_completion_weight"
    | "adaptive_step_balance_confidence_scale"
    | "adaptive_step_balance_min_confidence_episodes"
    | "adaptive_step_balance_target_completion"
    | "step_balance_ema_alpha"
    | "step_balance_max_weight_scale"
    | "step_balance_update_episodes"
  >;

  const toggleCourse = (courseId: string) => {
    setConfig((currentConfig) => {
      const nextSet = new Set(currentConfig.tracks.selected_course_ids);
      if (nextSet.has(courseId)) {
        if (nextSet.size === 1 && !currentConfig.tracks.include_x_cup) {
          return currentConfig;
        }
        nextSet.delete(courseId);
      } else {
        nextSet.add(courseId);
      }
      const ordered = orderedCourseIds(nextSet, allCourseIds);
      if (ordered.length === 0 && !currentConfig.tracks.include_x_cup) {
        return currentConfig;
      }
      return {
        ...currentConfig,
        tracks: normalizedTracks({
          ...currentConfig.tracks,
          selected_course_ids: ordered,
        }),
      };
    });
  };

  const toggleCup = (courseIds: readonly string[]) => {
    setConfig((currentConfig) => {
      const nextSet = new Set(currentConfig.tracks.selected_course_ids);
      const allSelected = courseIds.every((courseId) => nextSet.has(courseId));
      if (allSelected) {
        if (nextSet.size === courseIds.length && !currentConfig.tracks.include_x_cup) {
          return currentConfig;
        }
        for (const courseId of courseIds) {
          nextSet.delete(courseId);
        }
      } else {
        for (const courseId of courseIds) {
          nextSet.add(courseId);
        }
      }
      const ordered = orderedCourseIds(nextSet, allCourseIds);
      if (ordered.length === 0 && !currentConfig.tracks.include_x_cup) {
        return currentConfig;
      }
      return {
        ...currentConfig,
        tracks: normalizedTracks({
          ...currentConfig.tracks,
          selected_course_ids: ordered,
        }),
      };
    });
  };

  const setCupCollapsed = (cupId: string, collapsed: boolean) => {
    setCollapsedCupIds((currentIds: readonly string[]) =>
      collapsed
        ? currentIds.includes(cupId)
          ? currentIds
          : [...currentIds, cupId]
        : currentIds.filter((currentId) => currentId !== cupId),
    );
  };

  return (
    <ConfigStack>
      <RaceSetupPanels
        config={config}
        defaultConfig={defaultConfig}
        defaultGpDifficulty={defaultGpDifficulty}
        metadata={metadata}
        updateTracks={updateTracks}
      />
      <CourseSamplingPanel
        config={config}
        defaultConfig={defaultConfig}
        metadata={metadata}
        samplingDefaults={samplingDefaults}
        updateTracks={updateTracks}
      />
      <CoursePoolPanel
        allCourseIds={allCourseIds}
        collapsibleCupIds={collapsibleCupIds}
        collapsedCupIdSet={collapsedCupIdSet}
        config={config}
        cups={cups}
        defaultConfig={defaultConfig}
        defaultCourseIds={defaultCourseIds}
        selectedCourseIds={selectedCourseIds}
        selectedCourseSet={selectedCourseSet}
        setCollapsedCupIds={setCollapsedCupIds}
        setCupCollapsed={setCupCollapsed}
        toggleCourse={toggleCourse}
        toggleCup={toggleCup}
        updateTracks={updateTracks}
      />
    </ConfigStack>
  );
}
