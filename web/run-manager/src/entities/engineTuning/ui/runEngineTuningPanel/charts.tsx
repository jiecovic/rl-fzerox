// web/run-manager/src/entities/engineTuning/ui/runEngineTuningPanel/charts.tsx
import type {
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";
import { ENGINE_SLIDER } from "@/shared/domain/engineBuckets";

import { engineStepLabel, formatOptionalRaceTime, formatPercent, formatRaceTime } from "./format";
import type { EngineTuningViewMode } from "./labels";
import { banditObservedPerformanceMetric, objectiveNoun, objectiveSampleLabel } from "./metrics";

export function EngineSamplingProbabilityBars({
  candidates,
  mode,
  objective,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
  mode: EngineTuningViewMode;
  objective: EngineTuningRuntimeState["objective"];
}) {
  const firstCandidate = candidates[0]?.engine_setting_raw_value ?? 0;
  const lastCandidate = candidates.at(-1)?.engine_setting_raw_value ?? ENGINE_SLIDER.maxStep;
  const barWidth = 100 / Math.max(1, candidates.length);
  const maxProbability = Math.max(
    ...candidates.map((candidate) => candidate.selection_probability),
    0.000001,
  );

  return (
    <div className="grid gap-1">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-app-muted">
        <div className="flex flex-wrap gap-3">
          {mode !== "model" ? (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-3 bg-app-accent" aria-hidden="true" />
              {mode === "bandit"
                ? `bucket ${objectiveNoun(objective)} aggregate`
                : "successful finish aggregate"}
            </span>
          ) : null}
          <span className="inline-flex items-center gap-1.5">
            <span
              className={
                mode !== "model" ? "h-2.5 w-3 bg-app-accent/60" : "h-2.5 w-3 bg-app-accent/70"
              }
              aria-hidden="true"
            />
            reset sampler probability
          </span>
        </div>
        <span className="tabular-nums">y: probability 0-{formatPercent(maxProbability)}</span>
      </div>
      <svg
        aria-label="Engine sampling probability by slider step"
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
            mode !== "model" && candidate.finish_count > 0
              ? "var(--accent)"
              : mode !== "model"
                ? "color-mix(in srgb, var(--accent) 58%, transparent)"
                : "color-mix(in srgb, var(--accent) 70%, transparent)";
          const rawLabel = mode === "bandit" ? "bucket" : "engine";
          const engineLabel = engineStepLabel(candidate.engine_setting_raw_value);
          const banditMetric = banditObservedPerformanceMetric(objective);
          const estimateLabel = mode === "bandit" ? banditMetric.meanLabel : "estimated";
          const estimateValue =
            mode === "bandit"
              ? banditMetric.format(banditMetric.value(candidate))
              : formatRaceTime(candidate.estimated_finish_time_ms);
          const bestValue =
            mode === "bandit"
              ? banditMetric.format(banditMetric.bestValue(candidate))
              : formatOptionalRaceTime(candidate.best_finish_time_ms);
          const sampleLabel = objectiveSampleLabel(candidate, objective);
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
              <title>{`${rawLabel} ${engineLabel} · ${formatPercent(candidate.selection_probability)} probability · ${estimateLabel} ${estimateValue} · best ${bestValue} · ${sampleLabel}`}</title>
            </rect>
          );
        })}
      </svg>
      <div className="flex justify-between text-xs tabular-nums text-app-muted">
        <span>{engineStepLabel(firstCandidate)}</span>
        <span>engine slider step</span>
        <span>{engineStepLabel(lastCandidate)}</span>
      </div>
    </div>
  );
}

export function EngineMeanPerformanceBars({
  candidates,
  mode,
  objective,
  recommendedEngineSettingRawValue,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
  mode: EngineTuningViewMode;
  objective: EngineTuningRuntimeState["objective"];
  recommendedEngineSettingRawValue: number;
}) {
  const firstCandidate = candidates[0]?.engine_setting_raw_value ?? 0;
  const lastCandidate = candidates.at(-1)?.engine_setting_raw_value ?? ENGINE_SLIDER.maxStep;
  const barWidth = 100 / Math.max(1, candidates.length);
  const banditMetric = banditObservedPerformanceMetric(objective);
  const values =
    mode === "bandit"
      ? candidates
          .map((candidate) => banditMetric.value(candidate))
          .filter((value): value is number => value !== null)
      : candidates.map((candidate) => candidate.estimated_finish_time_ms);
  const lowValue = Math.min(...values, Number.POSITIVE_INFINITY);
  const highValue = Math.max(...values, Number.NEGATIVE_INFINITY);
  const hasValues = values.length > 0;
  const valueSpan = Math.max(0.000001, highValue - lowValue);

  return (
    <div className="grid gap-1">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-app-muted">
        <div className="flex flex-wrap gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent/45" aria-hidden="true" />
            {mode === "bandit" ? banditMetric.legendLabel : "predicted mean performance"}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent" aria-hidden="true" />
            {mode === "bandit" ? "greedy bucket" : "deterministic greedy"}
          </span>
        </div>
        <span className="tabular-nums">
          {hasValues
            ? `${mode === "bandit" ? banditMetric.rangeLabel : "estimated finish time"} ${banditMetric.format(lowValue)}-${banditMetric.format(highValue)}`
            : `${mode === "bandit" ? banditMetric.rangeLabel : "estimated finish time"} no observations`}
        </span>
      </div>
      <svg
        aria-label="Engine predicted mean performance by slider step"
        className="h-32 w-full border border-app-border bg-app-surface-muted"
        preserveAspectRatio="none"
        role="img"
        viewBox="0 0 100 100"
      >
        <title>
          {mode === "bandit"
            ? banditMetric.title
            : "Predicted mean performance from estimated finish time"}
        </title>
        {candidates.map((candidate, index) => {
          const isRecommended =
            candidate.engine_setting_raw_value === recommendedEngineSettingRawValue;
          const value =
            mode === "bandit" ? banditMetric.value(candidate) : candidate.estimated_finish_time_ms;
          const height =
            value === null || !hasValues
              ? 0
              : 3 +
                ((banditMetric.higherIsBetter ? value - lowValue : highValue - value) / valueSpan) *
                  93;
          const width = Math.max(0.2, barWidth * 0.86);
          const x = index * barWidth + (barWidth - width) / 2;
          const y = 100 - height;
          const rawLabel = mode === "bandit" ? "bucket" : "engine";
          const engineLabel = engineStepLabel(candidate.engine_setting_raw_value);
          const estimateLabel = mode === "bandit" ? banditMetric.meanLabel : "estimated";
          const estimateValue =
            mode === "bandit"
              ? banditMetric.format(value)
              : formatRaceTime(candidate.estimated_finish_time_ms);
          const bestValue =
            mode === "bandit"
              ? banditMetric.format(banditMetric.bestValue(candidate))
              : formatOptionalRaceTime(candidate.best_finish_time_ms);
          const sampleLabel = objectiveSampleLabel(candidate, objective);
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
              <title>{`${rawLabel} ${engineLabel}${isRecommended ? ` · ${mode === "bandit" ? "greedy bucket" : "deterministic greedy"}` : ""} · ${estimateLabel} ${estimateValue} · best ${bestValue} · ${sampleLabel}`}</title>
            </rect>
          );
        })}
      </svg>
      <div className="flex justify-between text-xs tabular-nums text-app-muted">
        <span>{engineStepLabel(firstCandidate)}</span>
        <span>
          engine slider step · taller means faster{" "}
          {mode === "bandit" ? banditMetric.axisLabel : "estimated finish"}
        </span>
        <span>{engineStepLabel(lastCandidate)}</span>
      </div>
    </div>
  );
}
