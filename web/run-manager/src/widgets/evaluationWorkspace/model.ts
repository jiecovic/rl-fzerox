// web/run-manager/src/widgets/evaluationWorkspace/model.ts
import type { EvaluationMetricSummary, ManagedEvaluation } from "@/shared/api/contract";
import {
  formatCourseRunRate as formatCourseRunRateValue,
  formatDate,
  formatDecimal,
  formatEnvStepRate as formatEnvStepRateValue,
  formatEtaSeconds,
  formatFrameRate as formatFrameRateValue,
  formatInteger,
  formatRaceTimeMs,
  formatRatioPercent,
} from "@/shared/ui/format";

export const EVALUATION_MODE_LABELS = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
} satisfies Record<ManagedEvaluation["target"]["mode"], string>;

export interface EvaluationRuntimeStats {
  courseRunsPerMinute: number | null;
  envStepsPerSecond: number | null;
  gameFramesPerSecond: number | null;
  etaSeconds: number | null;
}

export function evaluationRuntimeStats(
  evaluation: ManagedEvaluation,
): EvaluationRuntimeStats | null {
  const summary = evaluation.result_summary;
  const startedMs = timestampMs(summary?.started_at_utc ?? evaluation.started_at);
  if (summary === null || startedMs === null) {
    return null;
  }
  const endMs =
    evaluation.status === "running"
      ? Date.now()
      : (timestampMs(summary.closed_at_utc) ?? latestAttemptClosedMs(summary.attempts));
  if (endMs === null || endMs <= startedMs) {
    return null;
  }
  const elapsedSeconds = (endMs - startedMs) / 1000;
  const completed = summary.attempts.length;
  const total = evaluation.progress.total_attempts;
  const envSteps = summary.attempts.reduce((sum, attempt) => sum + (attempt.env_steps ?? 0), 0);
  const courseRunsPerSecond = completed > 0 ? completed / elapsedSeconds : null;
  const etaSeconds =
    evaluation.status === "running" &&
    total !== null &&
    total > completed &&
    courseRunsPerSecond !== null &&
    courseRunsPerSecond > 0
      ? (total - completed) / courseRunsPerSecond
      : null;
  const envStepsPerSecond = envSteps > 0 ? envSteps / elapsedSeconds : null;
  return {
    courseRunsPerMinute: courseRunsPerSecond === null ? null : courseRunsPerSecond * 60,
    envStepsPerSecond,
    gameFramesPerSecond:
      envStepsPerSecond === null
        ? null
        : envStepsPerSecond * Math.max(1, evaluation.config.action.action_repeat),
    etaSeconds,
  };
}

export function evaluationSubtitle(evaluation: ManagedEvaluation) {
  return `${EVALUATION_MODE_LABELS[evaluation.target.mode]} · ${
    evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "unknown run"
  } · ${evaluation.policy_mode}`;
}

export function progressLabel(evaluation: ManagedEvaluation) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  if (total !== null && total > 0) {
    return `${completed.toLocaleString()} / ${total.toLocaleString()} course runs`;
  }
  return completed > 0 ? `${completed.toLocaleString()} course runs` : "not started";
}

export function executionResultLabel(evaluation: ManagedEvaluation) {
  if (evaluation.result_json_path === null) {
    return "pending";
  }
  return evaluationResultStatusLabel(evaluation);
}

export function evaluationProgressStatusLabel(evaluation: ManagedEvaluation) {
  return evaluationStatusLabel(evaluation, evaluation.progress.result_status ?? evaluation.status);
}

export function evaluationResultStatusLabel(evaluation: ManagedEvaluation) {
  return evaluationStatusLabel(
    evaluation,
    evaluation.result_summary?.status ?? evaluation.progress.result_status ?? evaluation.status,
  );
}

export function evaluationRuntimeLabel(evaluation: ManagedEvaluation) {
  const runtime = evaluation.result_summary?.runtime ?? null;
  if (runtime === null) {
    return "-";
  }
  return `${runtime.device} · ${runtime.worker_count.toLocaleString()} ${pluralize(
    runtime.worker_count,
    "worker",
  )}`;
}

export function runCountDetail(evaluation: ManagedEvaluation, observedCourseCount: number) {
  const explicitCourseCount = evaluation.target.course_ids.length;
  const targetCount =
    explicitCourseCount > 0
      ? explicitCourseCount
      : observedCourseCount > 0
        ? observedCourseCount
        : null;
  const repeats = evaluation.target.repeats_per_target;
  return targetCount === null
    ? `${repeats.toLocaleString()} repeats per selected course`
    : `${targetCount.toLocaleString()} courses x ${repeats.toLocaleString()} repeats`;
}

export function weakestCourses(courses: readonly EvaluationMetricSummary[]) {
  return [...courses]
    .filter((course) => course.attempt_count > 0)
    .sort((left, right) => {
      const finishDelta = metricRate(left.finish_rate) - metricRate(right.finish_rate);
      if (finishDelta !== 0) {
        return finishDelta;
      }
      const completionDelta = metricRate(left.completion_rate) - metricRate(right.completion_rate);
      if (completionDelta !== 0) {
        return completionDelta;
      }
      return metricPosition(right.mean_position) - metricPosition(left.mean_position);
    });
}

export function barPercent(value: number | null) {
  if (value === null) {
    return "0%";
  }
  return `${Math.min(100, Math.max(0, value * 100))}%`;
}

export function targetSelectionLabel(target: ManagedEvaluation["target"]) {
  const parts = [
    selectionCountLabel(target.cup_ids, "cup"),
    selectionCountLabel(target.course_ids, "course"),
    difficultySelectionLabel(target.difficulties),
    selectionCountLabel(target.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

export function formatStepCount(value: number | null) {
  return value === null ? "step unknown" : `${formatInteger(value)} steps`;
}

export function formatPercent(value: number | null) {
  return value === null
    ? "-"
    : formatRatioPercent(value, { maximumFractionDigits: 1, minimumFractionDigits: 1 });
}

export function formatRaceTime(value: number | null) {
  return value === null ? "-" : formatRaceTimeMs(value);
}

export function formatNumber(value: number | null) {
  return value === null ? "-" : formatDecimal(value, { maximumFractionDigits: 2 });
}

export function formatFrameRate(value: number | null) {
  return value === null ? "-" : formatFrameRateValue(value);
}

export function formatSpeedDetail(runtimeStats: EvaluationRuntimeStats | null) {
  if (runtimeStats === null) {
    return undefined;
  }
  const parts = [
    formatCourseRunRate(runtimeStats.courseRunsPerMinute),
    formatEnvStepRate(runtimeStats.envStepsPerSecond),
  ].filter((part): part is string => part !== undefined && part !== "-");
  return parts.length === 0 ? undefined : parts.join(" · ");
}

export function formatEta(value: number | null) {
  return value === null ? "-" : formatEtaSeconds(value);
}

export function statusLabel(status: ManagedEvaluation["status"]) {
  return status.slice(0, 1).toUpperCase() + status.slice(1);
}

export function sourceSnapshotLabel(evaluation: ManagedEvaluation) {
  const sourceMtimeNs = evaluation.checkpoint.source_mtime_ns;
  if (sourceMtimeNs === null) {
    return "unknown";
  }
  try {
    const epochMs = Number(BigInt(sourceMtimeNs) / 1_000_000n);
    return formatDate(new Date(epochMs).toISOString());
  } catch {
    return "unknown";
  }
}

function latestAttemptClosedMs(
  attempts: NonNullable<ManagedEvaluation["result_summary"]>["attempts"],
) {
  let latest: number | null = null;
  for (const attempt of attempts) {
    const closed = timestampMs(attempt.closed_at_utc);
    if (closed !== null && (latest === null || closed > latest)) {
      latest = closed;
    }
  }
  return latest;
}

function timestampMs(value: string | null | undefined) {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function evaluationStatusLabel(evaluation: ManagedEvaluation, status: string) {
  if (status === "partial") {
    return evaluation.status === "running" || evaluation.status === "cancelling"
      ? "in progress"
      : "partial result";
  }
  return status.replace(/[_-]+/g, " ");
}

function metricRate(value: number | null) {
  return value ?? -1;
}

function metricPosition(value: number | null) {
  return value ?? 0;
}

function difficultySelectionLabel(difficulties: readonly string[]) {
  if (difficulties.length === 0) {
    return null;
  }
  return difficulties.map(titleLabel).join(", ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  return count === 1 ? singular : `${singular}s`;
}

function titleLabel(value: string) {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatCourseRunRate(value: number | null) {
  return value === null ? "-" : formatCourseRunRateValue(value);
}

function formatEnvStepRate(value: number | null) {
  if (value === null) {
    return undefined;
  }
  return formatEnvStepRateValue(value);
}
