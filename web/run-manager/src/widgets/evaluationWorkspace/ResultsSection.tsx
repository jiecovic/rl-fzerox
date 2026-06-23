// web/run-manager/src/widgets/evaluationWorkspace/ResultsSection.tsx
import type { EvaluationMetricSummary, ManagedEvaluation } from "@/shared/api/contract";

import {
  barPercent,
  evaluationResultStatusLabel,
  formatNumber,
  formatPercent,
  formatRaceTime,
  runCountDetail,
  weakestCourses,
} from "@/widgets/evaluationWorkspace/model";
import { Metric, SectionTitle } from "@/widgets/evaluationWorkspace/parts";

export function ResultsSection({ evaluation }: { evaluation: ManagedEvaluation }) {
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
        <span className="text-sm text-app-muted">{evaluationResultStatusLabel(evaluation)}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
        <Metric
          detail={runCountDetail(evaluation, summary.courses.length)}
          label="Course runs"
          value={summary.overall.attempt_count.toLocaleString()}
        />
        <Metric label="Finish" value={formatPercent(summary.overall.finish_rate)} />
        <Metric label="Completion" value={formatPercent(summary.overall.completion_rate)} />
        <Metric label="Mean finish pos" value={formatNumber(summary.overall.mean_position)} />
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
        columns={["Course", "Runs", "Finish", "Best time", "Finish pos", "Return", "Speed"]}
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
              )} completion, ${formatNumber(course.mean_position)} mean finish pos`}
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
