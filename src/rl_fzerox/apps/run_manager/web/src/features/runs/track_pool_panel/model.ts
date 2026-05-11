// src/rl_fzerox/apps/run_manager/web/src/features/runs/track_pool_panel/model.ts
import type {
  TrackPoolCourseView,
  TrackPoolCupView,
  TrackPoolView,
} from "@/features/runs/track_pool_panel/types";
import type {
  ConfigMetadata,
  ManagedRunDetail,
  TrackSamplingRuntimeEntry,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { formatRelativeTime } from "@/shared/ui/format";

export function buildTrackPoolView(
  metadata: ConfigMetadata,
  run: ManagedRunDetail,
  state: TrackSamplingRuntimeState | null,
): TrackPoolView {
  const cupOrder = metadata.track_cups.map((cup) => cup.id);
  const cupLabels = new Map(metadata.track_cups.map((cup) => [cup.id, cup.label]));
  const builtInCourses = new Map(metadata.built_in_courses.map((course) => [course.id, course]));
  const runtimeEntries = new Map<string, TrackSamplingRuntimeEntry>(
    (state?.entries ?? []).map((entry) => [entry.course_key, entry]),
  );
  const cups = new Map<string, TrackPoolCupView>();

  for (const courseId of run.config.tracks.selected_course_ids) {
    const courseInfo = builtInCourses.get(courseId);
    const runtimeEntry = runtimeEntries.get(courseId) ?? null;
    const cupId = courseInfo?.cup ?? "pool";
    const cupLabel = courseInfo?.cup_label ?? cupLabels.get(cupId) ?? "Pool";
    const cup: TrackPoolCupView = cups.get(cupId) ?? {
      completedEnvSteps: 0,
      currentProbability: 0,
      entries: [],
      episodeCount: 0,
      episodeShare: 0,
      finishedEpisodeCount: 0,
      id: cupId,
      label: cupLabel,
      stepShare: 0,
      successRate: null,
      successSampleCount: 0,
    };
    const courseView: TrackPoolCourseView = {
      completedEnvSteps: runtimeEntry?.completed_env_steps ?? null,
      currentProbability: runtimeEntry?.current_probability ?? null,
      episodeCount: runtimeEntry?.episode_count ?? null,
      episodeShare: runtimeEntry?.episode_share ?? null,
      finishedEpisodeCount: runtimeEntry?.finished_episode_count ?? null,
      id: courseId,
      label: runtimeEntry?.label ?? courseInfo?.display_name ?? courseId,
      stepShare: runtimeEntry?.step_share ?? null,
      successRate: runtimeEntry?.success_rate ?? null,
      successSampleCount: runtimeEntry?.success_sample_count ?? null,
    };
    cup.entries.push(courseView);
    cup.currentProbability += runtimeEntry?.current_probability ?? 0;
    cup.episodeCount += runtimeEntry?.episode_count ?? 0;
    cup.finishedEpisodeCount += runtimeEntry?.finished_episode_count ?? 0;
    cup.successSampleCount += runtimeEntry?.success_sample_count ?? 0;
    cup.episodeShare += runtimeEntry?.episode_share ?? 0;
    cup.completedEnvSteps += runtimeEntry?.completed_env_steps ?? 0;
    cup.stepShare += runtimeEntry?.step_share ?? 0;
    cup.successRate =
      cup.successSampleCount <= 0 ? null : cup.finishedEpisodeCount / cup.successSampleCount;
    cups.set(cupId, cup);
  }

  const sortedCups = [...cups.values()].sort(
    (left, right) => cupIndex(cupOrder, left.id) - cupIndex(cupOrder, right.id),
  );
  return {
    cups: sortedCups,
    totalCourses: run.config.tracks.selected_course_ids.length,
    totalEnvSteps: sortedCups.reduce((sum, cup) => sum + cup.completedEnvSteps, 0),
    totalEpisodes: sortedCups.reduce((sum, cup) => sum + cup.episodeCount, 0),
  };
}

export function showTrackSamplingState(state: TrackSamplingRuntimeState | null) {
  return state !== null && state.entries.length > 1;
}

export function expectsTrackSamplingState(run: ManagedRunDetail, totalCourses: number) {
  return run.config.tracks.sampling_mode === "step_balanced" && totalCourses > 1;
}

export function trackSamplingUpdatedLabel(run: ManagedRunDetail) {
  const updatedAt = run.runtime?.updated_at;
  if (updatedAt === undefined || updatedAt === null) {
    return "recently";
  }
  return formatRelativeTime(updatedAt);
}

export function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOptionalPercent(value: number | null) {
  return value === null ? "n/a" : formatPercent(value);
}

export function successLabel(entry: TrackPoolCourseView) {
  if ((entry.successSampleCount ?? 0) <= 0 || entry.successRate === null) {
    return "Finish rate not tracked yet";
  }
  return `${(entry.finishedEpisodeCount ?? 0).toLocaleString()} of ${(entry.successSampleCount ?? 0).toLocaleString()} finished · ${formatPercent(entry.successRate)}`;
}

export function successSummary(entry: TrackPoolCourseView) {
  if ((entry.successSampleCount ?? 0) <= 0 || entry.successRate === null) {
    return "finish n/a";
  }
  return `${formatPercent(entry.successRate)} finish`;
}

export function shortCupLabel(label: string) {
  return label.replace(/\s+Cup$/u, "");
}

export function trackPoolEmptyMessage(run: ManagedRunDetail) {
  return run.status === "running"
    ? "Waiting for the worker to publish the current track-pool distribution."
    : "No track-pool stats yet. Resume the run to accumulate a fresh distribution.";
}

function cupIndex(cupOrder: string[], cupId: string) {
  const index = cupOrder.indexOf(cupId);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}
