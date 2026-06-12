// web/run-manager/src/entities/engineTuning/ui/RunEngineTuningPanel.tsx
import { useEffect, useId, useMemo, useState } from "react";

import type {
  ConfigMetadata,
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";

interface RunEngineTuningPanelProps {
  artifact: "latest" | "best" | "final";
  enabled: boolean;
  metadata: ConfigMetadata;
  state: EngineTuningRuntimeState | null;
}

export function RunEngineTuningPanel({
  artifact,
  enabled,
  metadata,
  state,
}: RunEngineTuningPanelProps) {
  const contextSelectId = useId();
  const labels = useMemo(() => engineTuningLabels(metadata), [metadata]);
  const contexts = useMemo(() => sortedContexts(state?.contexts ?? [], labels), [state, labels]);
  const [selectedContextKey, setSelectedContextKey] = useState<string | null>(null);

  useEffect(() => {
    if (contexts.length === 0) {
      setSelectedContextKey(null);
      return;
    }
    if (
      selectedContextKey === null ||
      !contexts.some((context) => context.context_key === selectedContextKey)
    ) {
      setSelectedContextKey(contexts[0]?.context_key ?? null);
    }
  }, [contexts, selectedContextKey]);

  if (!enabled && state === null) {
    return null;
  }

  const selectedContext =
    contexts.find((context) => context.context_key === selectedContextKey) ?? contexts[0] ?? null;
  const observedCandidateCount =
    state?.candidates.filter((candidate) => candidate.finish_count > 0).length ?? 0;

  return (
    <section className="grid gap-3 border border-app-border bg-app-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1">
          <h3 className="m-0 text-base font-bold text-app-text">Engine tuning</h3>
          <p className="m-0 text-sm text-app-muted">
            Reset-time engine sampling probabilities from the {artifact} checkpoint.
          </p>
        </div>
        <div className="text-right text-xs tabular-nums text-app-muted">
          {state === null
            ? "no samples"
            : `${state.update_count.toLocaleString()} updates · ${contexts.length.toLocaleString()} contexts · ${observedCandidateCount.toLocaleString()} observed candidates`}
        </div>
      </div>
      {selectedContext === null ? (
        <p className="m-0 text-sm text-app-muted">
          {enabled
            ? "No engine tuning checkpoint data for this artifact yet. Values appear after a checkpoint save includes successful finish samples."
            : "Adaptive engine tuning is disabled for this run."}
        </p>
      ) : (
        <div className="grid gap-3">
          <div className="grid gap-1 md:max-w-[520px]">
            <label
              className="text-xs font-semibold tracking-[0.04em] text-app-muted uppercase"
              htmlFor={contextSelectId}
            >
              Context
            </label>
            <select
              className="min-h-10 border border-app-border bg-app-surface-muted px-3 py-2 text-sm text-app-text"
              id={contextSelectId}
              value={selectedContext.context_key}
              onChange={(event) => setSelectedContextKey(event.currentTarget.value)}
            >
              {contexts.map((context) => (
                <option key={context.context_key} value={context.context_key}>
                  {contextLabel(context, labels)} · {context.finish_count} finishes
                </option>
              ))}
            </select>
          </div>
          <div className="grid gap-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <strong className="text-sm text-app-text">
                {contextLabel(selectedContext, labels)}
              </strong>
              <span className="text-xs tabular-nums text-app-muted">
                greedy engine {selectedContext.recommended_engine_setting_raw_value} ·{" "}
                {selectedContext.finish_count.toLocaleString()} successful finishes
              </span>
            </div>
            <EngineSamplingProbabilityBars candidates={selectedContext.candidates} />
            <EnginePosteriorMeanBars
              candidates={selectedContext.candidates}
              recommendedEngineSettingRawValue={
                selectedContext.recommended_engine_setting_raw_value
              }
            />
          </div>
          <EngineDistributionTable candidates={selectedContext.candidates} />
        </div>
      )}
    </section>
  );
}

interface EngineTuningLabels {
  courses: ReadonlyMap<string, string>;
  vehicles: ReadonlyMap<string, string>;
}

function sortedContexts(
  contexts: readonly EngineTuningRuntimeContext[],
  labels: EngineTuningLabels,
) {
  return [...contexts].sort((left, right) => {
    const labelOrder = contextLabel(left, labels).localeCompare(contextLabel(right, labels));
    return labelOrder === 0 ? left.context_key.localeCompare(right.context_key) : labelOrder;
  });
}

function engineTuningLabels(metadata: ConfigMetadata): EngineTuningLabels {
  return {
    courses: new Map([
      ...metadata.built_in_courses.map((course) => [course.id, course.display_name] as const),
      ["x_cup", "X Cup"],
    ]),
    vehicles: new Map(
      metadata.vehicles.map((vehicle) => [vehicle.id, vehicle.display_name] as const),
    ),
  };
}

function contextLabel(context: EngineTuningRuntimeContext, labels: EngineTuningLabels) {
  const courseLabel = labels.courses.get(context.course_key) ?? humanizeKey(context.course_key);
  const vehicleLabel = labels.vehicles.get(context.vehicle_id) ?? humanizeKey(context.vehicle_id);
  return `${courseLabel} · ${vehicleLabel}`;
}

function EngineSamplingProbabilityBars({
  candidates,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
}) {
  const firstCandidate = candidates[0]?.engine_setting_raw_value ?? 0;
  const lastCandidate = candidates.at(-1)?.engine_setting_raw_value ?? 100;
  const barWidth = 100 / Math.max(1, candidates.length);
  const maxProbability = Math.max(
    ...candidates.map((candidate) => candidate.selection_probability),
    0.000001,
  );

  return (
    <div className="grid gap-1">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-app-muted">
        <div className="flex flex-wrap gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent" aria-hidden="true" />
            successful finish observed
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent/60" aria-hidden="true" />
            model prediction only
          </span>
        </div>
        <span className="tabular-nums">y: probability 0-{formatPercent(maxProbability)}</span>
      </div>
      <svg
        aria-label="Engine sampling probability by raw engine value"
        className="h-36 w-full border border-app-border bg-app-surface-muted"
        preserveAspectRatio="none"
        role="img"
        viewBox="0 0 100 100"
      >
        <title>Stochastic engine-selection probability</title>
        {candidates.map((candidate, index) => {
          const height = (candidate.selection_probability / maxProbability) * 96;
          const width = Math.max(0.2, barWidth * 0.86);
          const x = index * barWidth + (barWidth - width) / 2;
          const y = 100 - height;
          const fill =
            candidate.finish_count > 0
              ? "var(--accent)"
              : "color-mix(in srgb, var(--accent) 58%, transparent)";
          return (
            <rect
              fill={fill}
              height={height}
              key={candidate.engine_setting_raw_value}
              width={width}
              x={x}
              y={y}
              vectorEffect="non-scaling-stroke"
            >
              <title>{`engine ${candidate.engine_setting_raw_value} · ${formatPercent(candidate.selection_probability)} probability · estimated ${formatRaceTime(candidate.estimated_finish_time_ms)} · best ${formatOptionalRaceTime(candidate.best_finish_time_ms)} · ${candidate.finish_count} successful finishes`}</title>
            </rect>
          );
        })}
      </svg>
      <div className="flex justify-between text-xs tabular-nums text-app-muted">
        <span>{firstCandidate}</span>
        <span>engine raw value</span>
        <span>{lastCandidate}</span>
      </div>
    </div>
  );
}

function EnginePosteriorMeanBars({
  candidates,
  recommendedEngineSettingRawValue,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
  recommendedEngineSettingRawValue: number;
}) {
  const firstCandidate = candidates[0]?.engine_setting_raw_value ?? 0;
  const lastCandidate = candidates.at(-1)?.engine_setting_raw_value ?? 100;
  const barWidth = 100 / Math.max(1, candidates.length);
  const estimatedTimes = candidates.map((candidate) => candidate.estimated_finish_time_ms);
  const fastestEstimate = Math.min(...estimatedTimes, Number.POSITIVE_INFINITY);
  const slowestEstimate = Math.max(...estimatedTimes, 1);
  const timeSpan = Math.max(1, slowestEstimate - fastestEstimate);

  return (
    <div className="grid gap-1">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-app-muted">
        <div className="flex flex-wrap gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent/45" aria-hidden="true" />
            posterior mean performance
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent" aria-hidden="true" />
            deterministic greedy
          </span>
        </div>
        <span className="tabular-nums">
          estimated finish time {formatRaceTime(fastestEstimate)}-{formatRaceTime(slowestEstimate)}
        </span>
      </div>
      <svg
        aria-label="Engine posterior mean performance by raw engine value"
        className="h-32 w-full border border-app-border bg-app-surface-muted"
        preserveAspectRatio="none"
        role="img"
        viewBox="0 0 100 100"
      >
        <title>Posterior mean performance from estimated finish time</title>
        {candidates.map((candidate, index) => {
          const isRecommended =
            candidate.engine_setting_raw_value === recommendedEngineSettingRawValue;
          const height =
            3 + ((slowestEstimate - candidate.estimated_finish_time_ms) / timeSpan) * 93;
          const width = Math.max(0.2, barWidth * 0.86);
          const x = index * barWidth + (barWidth - width) / 2;
          const y = 100 - height;
          return (
            <rect
              fill={
                isRecommended
                  ? "var(--accent)"
                  : "color-mix(in srgb, var(--accent) 45%, transparent)"
              }
              height={height}
              key={candidate.engine_setting_raw_value}
              width={width}
              x={x}
              y={y}
              vectorEffect="non-scaling-stroke"
            >
              <title>{`engine ${candidate.engine_setting_raw_value}${isRecommended ? " · deterministic greedy" : ""} · estimated ${formatRaceTime(candidate.estimated_finish_time_ms)} · best ${formatOptionalRaceTime(candidate.best_finish_time_ms)} · ${candidate.finish_count} successful finishes`}</title>
            </rect>
          );
        })}
      </svg>
      <div className="flex justify-between text-xs tabular-nums text-app-muted">
        <span>{firstCandidate}</span>
        <span>engine raw value · taller means faster estimated finish</span>
        <span>{lastCandidate}</span>
      </div>
    </div>
  );
}

function EngineDistributionTable({
  candidates,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
}) {
  const topCandidates = [...candidates]
    .sort((left, right) => {
      if (right.selection_probability !== left.selection_probability) {
        return right.selection_probability - left.selection_probability;
      }
      return right.posterior_mean - left.posterior_mean;
    })
    .slice(0, 8);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs tabular-nums">
        <thead className="text-app-muted">
          <tr className="border-b border-app-border">
            <th className="py-1.5 pr-3 font-semibold">Engine</th>
            <th className="py-1.5 pr-3 font-semibold">Prob</th>
            <th className="py-1.5 pr-3 font-semibold">Estimated</th>
            <th className="py-1.5 pr-3 font-semibold">Best</th>
            <th className="py-1.5 font-semibold">Finishes</th>
          </tr>
        </thead>
        <tbody>
          {topCandidates.map((candidate) => (
            <tr
              className="border-b border-app-border/70 last:border-b-0"
              key={candidate.engine_setting_raw_value}
            >
              <td className="py-1.5 pr-3 text-app-text">{candidate.engine_setting_raw_value}</td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatPercent(candidate.selection_probability)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatRaceTime(candidate.estimated_finish_time_ms)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatOptionalRaceTime(candidate.best_finish_time_ms)}
              </td>
              <td className="py-1.5 text-app-text">{candidate.finish_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function humanizeKey(key: string) {
  return key
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

function formatPercent(value: number | null) {
  return value === null ? "-" : `${(value * 100).toFixed(1)}%`;
}

function formatOptionalRaceTime(value: number | null) {
  return value === null ? "no time yet" : formatRaceTime(value);
}

function formatRaceTime(value: number) {
  const safeValue = Math.max(0, Math.round(value));
  const minutes = Math.floor(safeValue / 60_000);
  const seconds = Math.floor((safeValue % 60_000) / 1000);
  const millis = safeValue % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}
