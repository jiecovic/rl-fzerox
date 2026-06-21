// web/run-manager/src/entities/engineTuning/ui/runEngineTuningPanel/metrics.ts
import type {
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";

import { formatOptionalRaceTime, formatPercent } from "./format";

export interface BanditObservedPerformanceMetric {
  axisLabel: string;
  bestValue: (candidate: EngineTuningRuntimeCandidateEstimate) => number | null;
  format: (value: number | null) => string;
  higherIsBetter: boolean;
  legendLabel: string;
  meanLabel: string;
  rangeLabel: string;
  title: string;
  value: (candidate: EngineTuningRuntimeCandidateEstimate) => number | null;
}

export function banditObservedPerformanceMetric(
  objective: EngineTuningRuntimeState["objective"],
): BanditObservedPerformanceMetric {
  if (objective === "finish_rate") {
    return {
      axisLabel: "measured finish rate",
      bestValue: (candidate) => candidate.finish_rate,
      format: formatPercent,
      higherIsBetter: true,
      legendLabel: "bucket finish rate",
      meanLabel: "finish rate",
      rangeLabel: "bucket finish rate",
      title: "Measured bucket finish rate",
      value: (candidate) => candidate.finish_rate,
    };
  }
  return {
    axisLabel: "measured mean finish",
    bestValue: (candidate) => candidate.best_finish_time_ms,
    format: formatOptionalRaceTime,
    higherIsBetter: false,
    legendLabel: "bucket mean finish",
    meanLabel: "mean",
    rangeLabel: "bucket mean finish time",
    title: "Measured bucket mean finish time",
    value: (candidate) => candidate.mean_finish_time_ms,
  };
}

export function candidateSampleCount(
  candidate: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (objective === "safe_finish_time" || objective === "finish_rate") {
    return candidate.episode_count;
  }
  return candidate.finish_count;
}

export function compareBanditBucketCandidates(
  left: EngineTuningRuntimeCandidateEstimate,
  right: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
) {
  const leftCount = candidateSampleCount(left, objective);
  const rightCount = candidateSampleCount(right, objective);
  if (leftCount <= 0 && rightCount <= 0) {
    return left.engine_setting_raw_value - right.engine_setting_raw_value;
  }
  if (leftCount <= 0) {
    return 1;
  }
  if (rightCount <= 0) {
    return -1;
  }
  if (isHigherScoreObjective(objective) && left.mean_score !== right.mean_score) {
    return right.mean_score - left.mean_score;
  }
  if (isHigherScoreObjective(objective) && left.best_score !== right.best_score) {
    return nullableScoreSortValue(right.best_score) - nullableScoreSortValue(left.best_score);
  }
  if (left.estimated_finish_time_ms !== right.estimated_finish_time_ms) {
    return left.estimated_finish_time_ms - right.estimated_finish_time_ms;
  }
  if (left.best_finish_time_ms !== right.best_finish_time_ms) {
    return (
      nullableRaceTimeSortValue(left.best_finish_time_ms) -
      nullableRaceTimeSortValue(right.best_finish_time_ms)
    );
  }
  if (right.finish_count !== left.finish_count) {
    return right.finish_count - left.finish_count;
  }
  return left.engine_setting_raw_value - right.engine_setting_raw_value;
}

export function isHigherScoreObjective(objective: EngineTuningRuntimeState["objective"]) {
  return objective === "finish_rate";
}

export function objectiveNoun(objective: EngineTuningRuntimeState["objective"]) {
  if (objective === "safe_finish_time") {
    return "safe finish";
  }
  if (objective === "finish_rate") {
    return "finish rate";
  }
  return "finish";
}

export function objectiveSampleLabel(
  candidate: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (objective === "safe_finish_time" || objective === "finish_rate") {
    return `${candidate.episode_count} episodes`;
  }
  return `${candidate.finish_count} successful finishes`;
}

function nullableRaceTimeSortValue(value: number | null) {
  return value ?? Number.POSITIVE_INFINITY;
}

function nullableScoreSortValue(value: number | null) {
  return value ?? Number.NEGATIVE_INFINITY;
}
