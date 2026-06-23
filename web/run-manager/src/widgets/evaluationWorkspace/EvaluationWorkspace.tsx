// web/run-manager/src/widgets/evaluationWorkspace/EvaluationWorkspace.tsx
import { useState } from "react";

import type {
  EvaluationMetricSummary,
  ManagedEvaluation,
  StartEvaluationRequest,
  WatchDevice,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldSelect } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import { EvaluationTabIcon, PlayIcon, RenameIcon, StopIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface EvaluationWorkspaceProps {
  evaluation: ManagedEvaluation;
  onCancelEvaluation: (evaluation: ManagedEvaluation) => Promise<ManagedEvaluation>;
  onGlobalError: (message: string | null) => void;
  onRenameEvaluation: (evaluationId: string, name: string) => Promise<void>;
  onStartEvaluation: (
    evaluation: ManagedEvaluation,
    request: StartEvaluationRequest,
  ) => Promise<ManagedEvaluation>;
}

const EVALUATION_MODE_LABELS = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
} satisfies Record<ManagedEvaluation["target"]["mode"], string>;

export function EvaluationWorkspace({
  evaluation,
  onCancelEvaluation,
  onGlobalError,
  onRenameEvaluation,
  onStartEvaluation,
}: EvaluationWorkspaceProps) {
  const [device, setDevice] = useState<WatchDevice>("cuda");
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const canStart =
    evaluation.status === "created" ||
    evaluation.status === "failed" ||
    evaluation.status === "cancelled";
  const canCancel = evaluation.status === "running";
  const cancelRequested = evaluation.status === "cancelling";
  const runtimeStats = evaluationRuntimeStats(evaluation);

  async function startEvaluation() {
    if (!canStart || starting) {
      return;
    }
    setStarting(true);
    onGlobalError(null);
    try {
      await onStartEvaluation(evaluation, { device });
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to start evaluation");
    } finally {
      setStarting(false);
    }
  }

  async function cancelEvaluation() {
    if (!canCancel || cancelling) {
      return;
    }
    setCancelling(true);
    onGlobalError(null);
    try {
      await onCancelEvaluation(evaluation);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to cancel evaluation");
    } finally {
      setCancelling(false);
    }
  }

  async function submitRename(name: string) {
    setRenaming(true);
    onGlobalError(null);
    try {
      await onRenameEvaluation(evaluation.id, name);
      setRenameDialogOpen(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to rename evaluation");
    } finally {
      setRenaming(false);
    }
  }

  return (
    <Panel>
      <PanelHeader
        title={
          <span className="inline-flex min-w-0 items-center gap-2">
            <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
              {evaluation.name}
            </span>
            <TooltipIconButton
              aria-label="Rename evaluation"
              disabled={renaming}
              size="small"
              tooltip="Rename"
              onClick={() => setRenameDialogOpen(true)}
            >
              <RenameIcon />
            </TooltipIconButton>
          </span>
        }
        subtitle={evaluationSubtitle(evaluation)}
      />
      <RenameDialog
        busy={renaming}
        initialName={evaluation.name}
        label="Evaluation name"
        open={renameDialogOpen}
        title="Rename evaluation"
        onClose={() => setRenameDialogOpen(false)}
        onSubmit={(name) => void submitRename(name)}
      />

      <div className="grid gap-4">
        <section className="border border-app-border bg-app-surface-muted p-4">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_auto]">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
              <Metric label="Status" value={statusLabel(evaluation.status)} />
              <Metric
                label="Checkpoint"
                value={evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "-"}
                detail={`${evaluation.checkpoint.artifact} · ${formatStepCount(
                  evaluation.checkpoint.lineage_num_timesteps,
                )}`}
              />
              <Metric
                label="Target"
                value={`${EVALUATION_MODE_LABELS[evaluation.target.mode]} · ${
                  evaluation.target.repeats_per_target
                }x`}
                detail={targetSelectionLabel(evaluation.target)}
              />
              <Metric label="Seed" value={String(evaluation.seed)} />
              <Metric
                detail={formatSpeedDetail(runtimeStats)}
                label="Speed"
                value={formatFrameRate(runtimeStats?.gameFramesPerSecond ?? null)}
              />
              <Metric label="ETA" value={formatEta(runtimeStats?.etaSeconds ?? null)} />
            </div>

            <div className="flex flex-wrap items-start justify-end gap-2">
              {canCancel || cancelRequested ? (
                <Button
                  className="gap-2"
                  disabled={cancelling || cancelRequested}
                  tone="danger"
                  onClick={() => void cancelEvaluation()}
                >
                  <StopIcon />
                  <span>{cancelling || cancelRequested ? "Cancelling" : "Cancel"}</span>
                </Button>
              ) : (
                <>
                  <FieldSelect
                    aria-label="Evaluation runtime device"
                    className="min-w-[110px]"
                    disabled={!canStart || starting}
                    value={device}
                    onChange={(event) => setDevice(event.currentTarget.value as WatchDevice)}
                  >
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </FieldSelect>
                  <Button
                    className="gap-2"
                    disabled={!canStart || starting}
                    variant={canStart ? "primary" : undefined}
                    onClick={() => void startEvaluation()}
                  >
                    <PlayIcon />
                    <span>
                      {starting ? "Starting" : evaluation.status === "created" ? "Start" : "Retry"}
                    </span>
                  </Button>
                </>
              )}
            </div>
          </div>

          <div className="mt-4">
            <ProgressBar evaluation={evaluation} />
          </div>
        </section>

        {evaluation.error_message !== null ? (
          <Notice tone="error">{evaluation.error_message}</Notice>
        ) : null}

        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="border border-app-border bg-app-surface p-4">
            <h3 className="m-0 text-base font-semibold text-app-text">Checkpoint</h3>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Evaluation id" value={evaluation.id} />
              <Detail
                label="Source run"
                value={evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "-"}
              />
              <Detail label="Artifact" value={evaluation.checkpoint.artifact} />
              <Detail
                label="Source steps"
                value={formatStepCount(evaluation.checkpoint.lineage_num_timesteps)}
              />
              <Detail label="Policy mode" value={evaluation.policy_mode} />
              <Detail label="Renderer" value={evaluation.config.environment.renderer} />
              <Detail label="Snapshot time" value={sourceSnapshotLabel(evaluation)} />
            </dl>
          </div>

          <div className="border border-app-border bg-app-surface p-4">
            <h3 className="m-0 text-base font-semibold text-app-text">Execution</h3>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Created" value={formatDate(evaluation.created_at)} />
              <Detail
                label="Started"
                value={evaluation.started_at === null ? "-" : formatDate(evaluation.started_at)}
              />
              <Detail
                label="Finished"
                value={evaluation.finished_at === null ? "-" : formatDate(evaluation.finished_at)}
              />
              <Detail
                label="Result"
                value={evaluation.result_json_path === null ? "pending" : "written"}
              />
            </dl>
          </div>
        </section>

        <ResultsSection evaluation={evaluation} />
      </div>
    </Panel>
  );
}

function Metric({ detail, label, value }: { detail?: string; label: string; value: string }) {
  return (
    <div className="border border-app-border bg-app-surface p-3">
      <div className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</div>
      <div className="mt-2 text-lg font-semibold text-app-text">{value}</div>
      {detail === undefined ? null : <div className="mt-1 text-xs text-app-muted">{detail}</div>}
    </div>
  );
}

function Detail({ label, mono = false, value }: { label: string; mono?: boolean; value: string }) {
  return (
    <div className="grid gap-1">
      <dt className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</dt>
      <dd className={mono ? "m-0 break-all font-mono text-app-muted" : "m-0 text-app-muted"}>
        {value}
      </dd>
    </div>
  );
}

function ProgressBar({ evaluation }: { evaluation: ManagedEvaluation }) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  const percent = total === null || total <= 0 ? null : Math.min(100, (completed / total) * 100);
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-4 text-sm text-app-muted">
        <span>{progressLabel(evaluation)}</span>
        <span>{evaluation.progress.result_status ?? evaluation.status}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-sm bg-app-surface">
        <div
          className="h-full bg-app-accent"
          style={{ width: `${percent ?? (completed > 0 ? 100 : 0)}%` }}
        />
      </div>
    </div>
  );
}

function evaluationRuntimeStats(evaluation: ManagedEvaluation) {
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

function ResultsSection({ evaluation }: { evaluation: ManagedEvaluation }) {
  const summary = evaluation.result_summary;
  if (summary === null || summary.overall === null) {
    return (
      <section className="border border-app-border bg-app-surface p-4">
        <SectionTitle title="Results" />
        <p className="mt-3 mb-0 text-sm text-app-muted">No results yet.</p>
      </section>
    );
  }
  const recentAttempts = [...summary.attempts].slice(-12).reverse();
  const weakestCourse = weakestCourses(summary.courses)[0] ?? null;
  return (
    <section className="grid gap-4 border border-app-border bg-app-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionTitle title="Results" />
        <span className="text-sm text-app-muted">{summary.status}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
        <Metric
          detail={runCountDetail(evaluation, summary.courses.length)}
          label="Course runs"
          value={summary.overall.attempt_count.toLocaleString()}
        />
        <Metric label="Finish" value={formatPercent(summary.overall.finish_rate)} />
        <Metric label="Completion" value={formatPercent(summary.overall.completion_rate)} />
        <Metric label="Mean pos" value={formatNumber(summary.overall.mean_position)} />
        <Metric label="Mean return" value={formatNumber(summary.overall.mean_episode_return)} />
        <Metric
          detail={
            weakestCourse === null
              ? undefined
              : `${formatPercent(weakestCourse.finish_rate)} finish · ${formatPercent(
                  weakestCourse.completion_rate,
                )} completion`
          }
          label="Weakest"
          value={weakestCourse?.label ?? "-"}
        />
      </div>
      <CourseDistribution courses={summary.courses} />
      <ResultTable
        columns={["Run", "Target", "Status", "Time", "Pos", "Return", "Steps"]}
        emptyLabel="No course runs recorded."
        rows={recentAttempts.map((attempt) => [
          attempt.attempt_id,
          attempt.target_label ?? attempt.target_id,
          attempt.status,
          formatRaceTime(attempt.total_race_time_ms),
          attempt.position === null ? "-" : String(attempt.position),
          formatNumber(attempt.episode_return),
          attempt.env_steps === null ? "-" : attempt.env_steps.toLocaleString(),
        ])}
        title="Recent course runs"
      />
      <ResultTable
        columns={["Course", "Runs", "Finish", "Best time", "Mean pos", "Return", "Speed"]}
        emptyLabel="No course breakdown yet."
        rows={summary.courses.map((course) => [
          course.label,
          course.attempt_count.toLocaleString(),
          formatPercent(course.finish_rate),
          formatRaceTime(course.best_finish_time_ms),
          formatNumber(course.mean_position),
          formatNumber(course.mean_episode_return),
          formatNumber(course.average_speed),
        ])}
        title="Course breakdown"
      />
    </section>
  );
}

function CourseDistribution({ courses }: { courses: readonly EvaluationMetricSummary[] }) {
  const rankedCourses = weakestCourses(courses);
  if (rankedCourses.length === 0) {
    return null;
  }
  const compactBars = rankedCourses.length <= 4;
  return (
    <div className="grid gap-2">
      <h3 className="m-0 text-sm font-semibold text-app-text">Course finish distribution</h3>
      <div className="overflow-hidden border border-app-border p-3">
        <div
          className={
            compactBars ? "grid items-end justify-center gap-1.5" : "grid items-end gap-1.5"
          }
          style={{
            gridTemplateColumns: compactBars
              ? `repeat(${rankedCourses.length}, minmax(72px, 120px))`
              : `repeat(${rankedCourses.length}, minmax(0, 1fr))`,
          }}
        >
          {rankedCourses.map((course, index) => (
            <div
              className="grid min-w-0 grid-rows-[150px_auto_34px] gap-1"
              key={course.key || course.label}
              title={`${course.label}: ${formatPercent(course.finish_rate)} finish, ${formatPercent(
                course.completion_rate,
              )} completion, ${formatNumber(course.mean_position)} mean pos`}
            >
              <div className="flex h-[150px] items-end border-b border-app-border bg-app-surface-muted px-1">
                <div
                  className={index === 0 ? "w-full bg-app-danger" : "w-full bg-app-accent"}
                  style={{ height: barPercent(course.finish_rate) }}
                />
              </div>
              <span
                className={
                  index === 0
                    ? "text-center text-[11px] font-semibold whitespace-nowrap text-app-danger tabular-nums"
                    : "text-center text-[11px] whitespace-nowrap text-app-muted tabular-nums"
                }
              >
                {formatPercent(course.finish_rate)}
              </span>
              <span className="line-clamp-2 overflow-hidden text-center text-[10px] leading-tight text-app-muted">
                {course.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ title }: { title: string }) {
  return (
    <div className="flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
      <EvaluationTabIcon />
      <span>{title}</span>
    </div>
  );
}

function ResultTable({
  columns,
  emptyLabel,
  rows,
  title,
}: {
  columns: readonly string[];
  emptyLabel: string;
  rows: readonly (readonly string[])[];
  title: string;
}) {
  return (
    <div className="grid gap-2">
      <h3 className="m-0 text-sm font-semibold text-app-text">{title}</h3>
      {rows.length === 0 ? (
        <p className="m-0 text-sm text-app-muted">{emptyLabel}</p>
      ) : (
        <div className="overflow-x-auto border border-app-border">
          <table className="w-full min-w-[720px] border-collapse text-left text-sm">
            <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
              <tr>
                {columns.map((column) => (
                  <th className="px-3 py-2" key={column}>
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr className="border-b border-app-border last:border-b-0" key={row.join("|")}>
                  {columns.map((column, index) => (
                    <td className="px-3 py-2 text-app-muted" key={column}>
                      {row[index] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function evaluationSubtitle(evaluation: ManagedEvaluation) {
  return `${EVALUATION_MODE_LABELS[evaluation.target.mode]} · ${
    evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "unknown run"
  } · ${evaluation.policy_mode}`;
}

function progressLabel(evaluation: ManagedEvaluation) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  if (total !== null && total > 0) {
    return `${completed.toLocaleString()} / ${total.toLocaleString()} course runs`;
  }
  return completed > 0 ? `${completed.toLocaleString()} course runs` : "not started";
}

function runCountDetail(evaluation: ManagedEvaluation, observedCourseCount: number) {
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

function weakestCourses(courses: readonly EvaluationMetricSummary[]) {
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

function metricRate(value: number | null) {
  return value ?? -1;
}

function metricPosition(value: number | null) {
  return value ?? 0;
}

function barPercent(value: number | null) {
  if (value === null) {
    return "0%";
  }
  return `${Math.min(100, Math.max(0, value * 100))}%`;
}

function targetSelectionLabel(target: ManagedEvaluation["target"]) {
  const parts = [
    selectionCountLabel(target.cup_ids, "cup"),
    selectionCountLabel(target.course_ids, "course"),
    difficultySelectionLabel(target.difficulties),
    selectionCountLabel(target.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
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

function formatStepCount(value: number | null) {
  return value === null ? "step unknown" : `${value.toLocaleString()} steps`;
}

function formatPercent(value: number | null) {
  return value === null ? "-" : `${(value * 100).toFixed(1)}%`;
}

function formatRaceTime(value: number | null) {
  if (value === null) {
    return "-";
  }
  const milliseconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.floor((milliseconds % 60_000) / 1_000);
  const millis = milliseconds % 1_000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function formatNumber(value: number | null) {
  if (value === null) {
    return "-";
  }
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}

function formatCourseRunRate(value: number | null) {
  if (value === null) {
    return "-";
  }
  const formatted = value >= 10 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted} runs/min`;
}

function formatFrameRate(value: number | null) {
  if (value === null) {
    return "-";
  }
  return `${Math.round(value).toLocaleString()} fps`;
}

function formatEnvStepRate(value: number | null) {
  if (value === null) {
    return undefined;
  }
  return `${Math.round(value).toLocaleString()} env steps/s`;
}

function formatSpeedDetail(runtimeStats: ReturnType<typeof evaluationRuntimeStats>) {
  if (runtimeStats === null) {
    return undefined;
  }
  const parts = [
    formatCourseRunRate(runtimeStats.courseRunsPerMinute),
    formatEnvStepRate(runtimeStats.envStepsPerSecond),
  ].filter((part): part is string => part !== undefined && part !== "-");
  return parts.length === 0 ? undefined : parts.join(" · ");
}

function formatEta(value: number | null) {
  if (value === null) {
    return "-";
  }
  const totalSeconds = Math.max(0, Math.round(value));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  return `${seconds}s`;
}

function statusLabel(status: ManagedEvaluation["status"]) {
  return status.slice(0, 1).toUpperCase() + status.slice(1);
}

function sourceSnapshotLabel(evaluation: ManagedEvaluation) {
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
