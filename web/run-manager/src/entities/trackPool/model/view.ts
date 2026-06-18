// web/run-manager/src/entities/trackPool/model/view.ts

import type {
  TrackPoolCourseView,
  TrackPoolCupView,
  TrackPoolView,
} from "@/entities/trackPool/model/types";
import type {
  ConfigMetadata,
  ManagedRunDetail,
  TrackSamplingRuntimeEntry,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { formatRelativeTime } from "@/shared/ui/format";

const X_CUP_TRACK_POOL = {
  courseKeyPrefix: "x_cup_",
  cupId: "x",
  label: "X Cup",
} as const;

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
  const consumedRuntimeCourseKeys = new Set<string>();

  for (const courseId of run.config.tracks.selected_course_ids) {
    const courseInfo = builtInCourses.get(courseId);
    const runtimeEntry = runtimeEntries.get(courseId) ?? null;
    if (runtimeEntry !== null) {
      consumedRuntimeCourseKeys.add(runtimeEntry.course_key);
    }
    const cupId = courseInfo?.cup ?? "pool";
    const cupLabel = courseInfo?.cup_label ?? cupLabels.get(cupId) ?? "Pool";
    const cup = cups.get(cupId) ?? emptyCup(cupId, cupLabel);
    addCourseToCup(
      cup,
      courseViewFromRuntime({
        id: courseId,
        label: runtimeEntry?.label ?? courseInfo?.display_name ?? courseId,
        runtimeEntry,
      }),
      runtimeEntry,
    );
    cups.set(cupId, cup);
  }

  if (run.config.tracks.include_x_cup) {
    const xCup = emptyCup(X_CUP_TRACK_POOL.cupId, X_CUP_TRACK_POOL.label);
    const xCupRuntimeEntries = (state?.entries ?? []).filter(
      (entry) =>
        entry.course_key.startsWith(X_CUP_TRACK_POOL.courseKeyPrefix) &&
        !consumedRuntimeCourseKeys.has(entry.course_key),
    );
    if (xCupRuntimeEntries.length > 0) {
      for (const entry of [...xCupRuntimeEntries].sort(compareXcupEntries)) {
        addCourseToCup(
          xCup,
          courseViewFromRuntime({
            id: entry.course_key,
            label: entry.label,
            runtimeEntry: entry,
          }),
          entry,
        );
      }
    } else {
      for (let index = 0; index < run.config.tracks.x_cup_course_count; index += 1) {
        xCup.entries.push(
          placeholderCourseView(
            `${X_CUP_TRACK_POOL.courseKeyPrefix}${index + 1}`,
            `${X_CUP_TRACK_POOL.label} ${index + 1}`,
          ),
        );
      }
    }
    if (xCup.entries.length > 0) {
      cups.set(X_CUP_TRACK_POOL.cupId, xCup);
    }
  }

  const sortedCups = [...cups.values()].sort(
    (left, right) => cupIndex(cupOrder, left.id) - cupIndex(cupOrder, right.id),
  );
  const samplingMode = state?.sampling_mode ?? run.config.tracks.sampling_mode;
  return {
    cups: sortedCups,
    deficitBudgetDifficultyMetric:
      state?.sampling_mode === "deficit_budget" ? state.deficit_budget_difficulty_metric : null,
    deficitBudgetWarmupMinEpisodesPerCourse:
      state?.sampling_mode === "deficit_budget"
        ? state.deficit_budget_warmup_min_episodes_per_course
        : null,
    sampleBarUsesTargetShare: samplingMode === "deficit_budget",
    stepMetricLabel:
      samplingMode === "deficit_budget"
        ? "tracked scheduler env steps"
        : "completed episode env steps",
    showStepTarget: run.config.tracks.sampling_mode !== "fixed_env",
    totalCourses: sortedCups.reduce((sum, cup) => sum + cup.entries.length, 0),
    totalEnvSteps: sortedCups.reduce((sum, cup) => sum + cup.completedEnvSteps, 0),
    totalEpisodes: sortedCups.reduce((sum, cup) => sum + cup.episodeCount, 0),
  };
}

export function showTrackSamplingState(state: TrackSamplingRuntimeState | null) {
  return state !== null && state.entries.length > 1;
}

export function expectsTrackSamplingState(run: ManagedRunDetail, totalCourses: number) {
  return usesDynamicTrackSampling(run.config.tracks.sampling_mode) && totalCourses > 1;
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
  const completionLabel = completionSummary(entry);
  const generationLabel = xCupGenerationSummary(entry);
  const prefix = generationLabel === null ? "" : `${generationLabel} · `;
  const successSampleCount = entry.successSampleCount ?? 0;
  if (successSampleCount <= 0 || entry.successRate === null) {
    return `${prefix}${generationLabel === null ? "" : "sampling: "}${
      completionLabel === null
        ? "No completed episode samples yet"
        : `No completed episode samples yet · ${completionLabel}`
    }`;
  }
  return `${prefix}${generationLabel === null ? "" : "sampling: "}${(entry.finishedEpisodeCount ?? 0).toLocaleString()} of ${successSampleCount.toLocaleString()} finished · ${formatPercent(entry.successRate)} finish${
    completionLabel === null ? "" : ` · ${completionLabel}`
  }`;
}

export function completionLabel(entry: TrackPoolCourseView) {
  const sampleCount = entry.completionSampleCount ?? 0;
  if (sampleCount <= 0 || entry.completionRate === null) {
    return "No completion samples yet";
  }
  return `${sampleCount.toLocaleString()} completion samples · ${formatPercent(entry.completionRate)} completion`;
}

export function successSummary(entry: TrackPoolCourseView) {
  if ((entry.successSampleCount ?? 0) <= 0 || entry.successRate === null) {
    return "no episode samples";
  }
  return `${formatPercent(entry.successRate)} finish`;
}

export function completionSummary(entry: TrackPoolCourseView) {
  if (entry.completionRate === null) {
    return null;
  }
  return `${formatPercent(entry.completionRate)} comp`;
}

export function samplerSignalSummary(entry: TrackPoolCourseView) {
  if (entry.currentProblemScore === null) {
    return null;
  }
  const parts = [`problem score ${formatPercent(entry.currentProblemScore)}`];
  if (entry.emaFinishRate !== null) {
    parts.push(`EMA finish ${formatPercent(entry.emaFinishRate)}`);
  }
  return parts.join(" · ");
}

export function xCupRegenerationSummary(
  entry: TrackPoolCourseView,
  threshold: number | null,
  minEpisodes: number | null,
) {
  if (!isGeneratedXcupEntry(entry)) {
    return null;
  }
  const sampleCount = entry.generationEpisodeCount ?? 0;
  const successSampleCount = entry.generationSuccessSampleCount ?? 0;
  const parts: string[] = [];
  parts.push(
    minEpisodes === null
      ? `${sampleCount.toLocaleString()} regen episodes`
      : `${sampleCount.toLocaleString()} / ${minEpisodes.toLocaleString()} regen episodes`,
  );
  if (successSampleCount <= 0 || entry.generationSuccessRate === null) {
    parts.push("finish n/a");
  } else {
    parts.push(
      `${(entry.generationFinishedEpisodeCount ?? 0).toLocaleString()} of ${successSampleCount.toLocaleString()} finished`,
      `${formatPercent(entry.generationSuccessRate)} finish`,
    );
  }
  if (entry.generationEmaCompletionFraction !== null) {
    const thresholdLabel = threshold === null ? "" : ` / ${formatPercent(threshold)} threshold`;
    parts.push(
      `${formatPercent(entry.generationEmaCompletionFraction)} regen comp${thresholdLabel}`,
    );
  } else if (threshold !== null) {
    parts.push(`${formatPercent(threshold)} threshold`);
  }
  return parts.join(" · ");
}

export function xCupGenerationSummary(entry: TrackPoolCourseView) {
  if (entry.generatedCourseSlot === null && entry.generatedCourseGeneration === null) {
    return null;
  }
  const parts: string[] = [];
  if (entry.generatedCourseSlot !== null) {
    parts.push(`slot ${entry.generatedCourseSlot + 1}`);
  }
  if (entry.generatedCourseGeneration !== null) {
    parts.push(`generated ${entry.generatedCourseGeneration}`);
  }
  return parts.join(" · ");
}

export function shortCupLabel(label: string) {
  return label.replace(/\s+Cup$/u, "");
}

export function trackPoolEmptyMessage(run: ManagedRunDetail) {
  return run.status === "running"
    ? "Waiting for the worker to publish the current track-pool distribution."
    : "No track-pool stats yet. Resume the run to accumulate a fresh distribution.";
}

export function trackSamplingModeLabel(
  samplingMode: ManagedRunDetail["config"]["tracks"]["sampling_mode"],
) {
  if (samplingMode === "fixed_env") {
    return "fixed env assignment";
  }
  if (samplingMode === "adaptive_step_balanced") {
    return "adaptive step-balanced";
  }
  if (samplingMode === "deficit_budget") {
    return "deficit budget";
  }
  if (samplingMode === "step_balanced") {
    return "step-balanced";
  }
  return samplingMode.replaceAll("_", " ");
}

export function displaySuccessRate(entry: TrackPoolCourseView) {
  return entry.successRate;
}

function usesDynamicTrackSampling(
  samplingMode: ManagedRunDetail["config"]["tracks"]["sampling_mode"],
) {
  return (
    samplingMode === "step_balanced" ||
    samplingMode === "adaptive_step_balanced" ||
    samplingMode === "deficit_budget" ||
    samplingMode === "fixed_env"
  );
}

function emptyCup(id: string, label: string): TrackPoolCupView {
  return {
    completedEnvSteps: 0,
    currentProbability: 0,
    entries: [],
    episodeCount: 0,
    episodeShare: 0,
    finishedEpisodeCount: 0,
    id,
    label,
    stepShare: 0,
    successRate: null,
    successSampleCount: 0,
  };
}

function courseViewFromRuntime({
  id,
  label,
  runtimeEntry,
}: {
  id: string;
  label: string;
  runtimeEntry: TrackSamplingRuntimeEntry | null;
}): TrackPoolCourseView {
  return {
    completedEnvSteps: runtimeEntry?.completed_env_steps ?? null,
    completionFractionTotal: runtimeEntry?.completion_fraction_total ?? null,
    completionRate: runtimeEntry?.completion_rate ?? null,
    completionSampleCount: runtimeEntry?.completion_sample_count ?? null,
    currentProbability: runtimeEntry?.current_probability ?? null,
    currentProblemScore: runtimeEntry?.current_problem_score ?? null,
    emaCompletionFraction: runtimeEntry?.ema_completion_fraction ?? null,
    emaFinishRate: runtimeEntry?.ema_finish_rate ?? null,
    episodeCount: runtimeEntry?.episode_count ?? null,
    episodeShare: runtimeEntry?.episode_share ?? null,
    finishedEpisodeCount: runtimeEntry?.finished_episode_count ?? null,
    generationEmaCompletionFraction: runtimeEntry?.generation_ema_completion_fraction ?? null,
    generationEpisodeCount: runtimeEntry?.generation_episode_count ?? null,
    generationFinishedEpisodeCount: runtimeEntry?.generation_finished_episode_count ?? null,
    generationSuccessRate: runtimeEntry?.generation_success_rate ?? null,
    generationSuccessSampleCount: runtimeEntry?.generation_success_sample_count ?? null,
    generatedCourseGeneration: runtimeEntry?.generated_course_generation ?? null,
    generatedCourseSlot: runtimeEntry?.generated_course_slot ?? null,
    id,
    label,
    stepShare: runtimeEntry?.step_share ?? null,
    targetStepShare: runtimeEntry?.target_step_share ?? null,
    successRate: runtimeEntry?.success_rate ?? null,
    successSampleCount: runtimeEntry?.success_sample_count ?? null,
  };
}

function placeholderCourseView(id: string, label: string): TrackPoolCourseView {
  return courseViewFromRuntime({ id, label, runtimeEntry: null });
}

function addCourseToCup(
  cup: TrackPoolCupView,
  courseView: TrackPoolCourseView,
  runtimeEntry: TrackSamplingRuntimeEntry | null,
) {
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
}

function cupIndex(cupOrder: string[], cupId: string) {
  if (cupId === X_CUP_TRACK_POOL.cupId) {
    return cupOrder.length;
  }
  const index = cupOrder.indexOf(cupId);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function compareXcupEntries(left: TrackSamplingRuntimeEntry, right: TrackSamplingRuntimeEntry) {
  const leftSlot = left.generated_course_slot ?? Number.MAX_SAFE_INTEGER;
  const rightSlot = right.generated_course_slot ?? Number.MAX_SAFE_INTEGER;
  if (leftSlot !== rightSlot) {
    return leftSlot - rightSlot;
  }
  return left.label.localeCompare(right.label);
}

function isGeneratedXcupEntry(entry: TrackPoolCourseView) {
  return entry.generatedCourseSlot !== null || entry.generatedCourseGeneration !== null;
}
