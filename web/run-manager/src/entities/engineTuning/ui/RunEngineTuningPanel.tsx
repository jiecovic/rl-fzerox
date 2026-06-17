// web/run-manager/src/entities/engineTuning/ui/RunEngineTuningPanel.tsx
import { useEffect, useId, useMemo, useState } from "react";

import type {
  ConfigMetadata,
  EngineTuningRuntimeCandidate,
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";
import { ENGINE_SLIDER_STEP_MAX, engineSliderStepLabel } from "@/shared/domain/engineBuckets";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { ResetIcon } from "@/shared/ui/icons";

const ENGINE_TUNING_TABLE_ROW_LIMIT = 12;

interface RunEngineTuningPanelProps {
  artifact: "latest" | "best" | "final";
  canReset: boolean;
  enabled: boolean;
  expanded: boolean;
  isResetting: boolean;
  metadata: ConfigMetadata;
  state: EngineTuningRuntimeState | null;
  onExpandedChange: (expanded: boolean) => void;
  onReset: () => void;
}

export function RunEngineTuningPanel({
  artifact,
  canReset,
  enabled,
  expanded,
  isResetting,
  metadata,
  state,
  onExpandedChange,
  onReset,
}: RunEngineTuningPanelProps) {
  const contextSelectId = useId();
  const labels = useMemo(() => engineTuningLabels(metadata), [metadata]);
  const contexts = useMemo(() => sortedContexts(state?.contexts ?? [], labels), [state, labels]);
  const [confirmResetOpen, setConfirmResetOpen] = useState(false);
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
  const backend = state?.model_backend ?? null;
  const viewMode = engineTuningViewMode(backend);
  const observedCandidateCount = state?.candidates.length ?? 0;
  const statusText =
    state === null
      ? "no checkpoint data"
      : state.model_backend === "mlp_ensemble"
        ? `${backendLabel(state.model_backend)} · ${state.update_count.toLocaleString()} updates · ${contexts.length.toLocaleString()} contexts`
        : `${backendLabel(state.model_backend)} · ${state.update_count.toLocaleString()} updates · ${contexts.length.toLocaleString()} contexts · ${observedCandidateCount.toLocaleString()} aggregate candidates`;

  return (
    <>
      <ConfigDisclosure
        defaultOpen={false}
        open={expanded}
        title="Engine tuning"
        onToggle={onExpandedChange}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="grid gap-1">
            <p className="m-0 text-sm text-app-muted">
              Reset-time engine sampling probabilities from the {artifact} checkpoint.
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <span className="text-right text-xs tabular-nums text-app-muted">{statusText}</span>
            <Button
              className="h-8 gap-1.5 px-3 text-xs"
              disabled={!canReset || !enabled || isResetting}
              tone="danger"
              onClick={() => setConfirmResetOpen(true)}
            >
              <ResetIcon />
              <span>{isResetting ? "Resetting" : "Reset tuner"}</span>
            </Button>
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
                  {selectedContext.model_ready ? "greedy engine" : "warmup engine"}{" "}
                  {engineStepLabel(selectedContext.recommended_engine_setting_raw_value)} ·{" "}
                  {selectedContext.finish_count.toLocaleString()} successful finishes
                </span>
              </div>
              <EngineSamplingProbabilityBars
                candidates={selectedContext.candidates}
                mode={viewMode}
              />
              <EngineMeanPerformanceBars
                candidates={selectedContext.candidates}
                mode={viewMode}
                recommendedEngineSettingRawValue={
                  selectedContext.recommended_engine_setting_raw_value
                }
              />
            </div>
            {backend === "bandit" ? (
              <EngineBanditBucketTable candidates={selectedContext.candidates} />
            ) : backend === "gaussian_process" ? (
              <EngineMeasuredCandidateTable
                estimates={selectedContext.candidates}
                measuredCandidates={measuredCandidatesForContext(
                  state?.candidates ?? [],
                  selectedContext.context_key,
                )}
              />
            ) : selectedContext.model_ready ? (
              <EngineModelCandidateTable candidates={selectedContext.candidates} />
            ) : null}
          </div>
        )}
      </ConfigDisclosure>
      <ConfirmDialog
        busy={isResetting}
        busyLabel="Resetting..."
        confirmLabel="Reset tuner"
        description="Remove engine-tuning checkpoint sidecars for this run? Policy checkpoints remain, and resume will start the tuner from scratch."
        open={confirmResetOpen}
        title="Reset engine tuner"
        onClose={() => setConfirmResetOpen(false)}
        onConfirm={() => {
          setConfirmResetOpen(false);
          onReset();
        }}
      />
    </>
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

function backendLabel(backend: EngineTuningRuntimeState["model_backend"]) {
  if (backend === "mlp_ensemble") {
    return "MLP ensemble (experimental)";
  }
  if (backend === "gaussian_process") {
    return "GP (experimental)";
  }
  if (backend === "bandit") {
    return "Bandit";
  }
  return "no model";
}

type EngineTuningViewMode = "bandit" | "aggregate" | "model";

function engineTuningViewMode(
  backend: EngineTuningRuntimeState["model_backend"],
): EngineTuningViewMode {
  if (backend === "bandit") {
    return "bandit";
  }
  if (backend === "mlp_ensemble") {
    return "model";
  }
  return "aggregate";
}

function EngineSamplingProbabilityBars({
  candidates,
  mode,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
  mode: EngineTuningViewMode;
}) {
  const firstCandidate = candidates[0]?.engine_setting_raw_value ?? 0;
  const lastCandidate = candidates.at(-1)?.engine_setting_raw_value ?? ENGINE_SLIDER_STEP_MAX;
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
              {mode === "bandit" ? "bucket finish aggregate" : "successful finish aggregate"}
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
          const estimateLabel = mode === "bandit" ? "mean" : "estimated";
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
              <title>{`${rawLabel} ${engineLabel} · ${formatPercent(candidate.selection_probability)} probability · ${estimateLabel} ${formatRaceTime(candidate.estimated_finish_time_ms)} · best ${formatOptionalRaceTime(candidate.best_finish_time_ms)} · ${candidate.finish_count} successful finishes`}</title>
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

function EngineMeanPerformanceBars({
  candidates,
  mode,
  recommendedEngineSettingRawValue,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
  mode: EngineTuningViewMode;
  recommendedEngineSettingRawValue: number;
}) {
  const firstCandidate = candidates[0]?.engine_setting_raw_value ?? 0;
  const lastCandidate = candidates.at(-1)?.engine_setting_raw_value ?? ENGINE_SLIDER_STEP_MAX;
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
            {mode === "bandit" ? "bucket mean finish" : "predicted mean performance"}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent" aria-hidden="true" />
            {mode === "bandit" ? "greedy bucket" : "deterministic greedy"}
          </span>
        </div>
        <span className="tabular-nums">
          {mode === "bandit" ? "bucket mean finish time" : "estimated finish time"}{" "}
          {formatRaceTime(fastestEstimate)}-{formatRaceTime(slowestEstimate)}
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
            ? "Measured bucket mean finish time"
            : "Predicted mean performance from estimated finish time"}
        </title>
        {candidates.map((candidate, index) => {
          const isRecommended =
            candidate.engine_setting_raw_value === recommendedEngineSettingRawValue;
          const height =
            3 + ((slowestEstimate - candidate.estimated_finish_time_ms) / timeSpan) * 93;
          const width = Math.max(0.2, barWidth * 0.86);
          const x = index * barWidth + (barWidth - width) / 2;
          const y = 100 - height;
          const rawLabel = mode === "bandit" ? "bucket" : "engine";
          const engineLabel = engineStepLabel(candidate.engine_setting_raw_value);
          const estimateLabel = mode === "bandit" ? "mean" : "estimated";
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
              <title>{`${rawLabel} ${engineLabel}${isRecommended ? ` · ${mode === "bandit" ? "greedy bucket" : "deterministic greedy"}` : ""} · ${estimateLabel} ${formatRaceTime(candidate.estimated_finish_time_ms)} · best ${formatOptionalRaceTime(candidate.best_finish_time_ms)} · ${candidate.finish_count} successful finishes`}</title>
            </rect>
          );
        })}
      </svg>
      <div className="flex justify-between text-xs tabular-nums text-app-muted">
        <span>{engineStepLabel(firstCandidate)}</span>
        <span>
          engine slider step · taller means faster{" "}
          {mode === "bandit" ? "bucket mean finish" : "estimated finish"}
        </span>
        <span>{engineStepLabel(lastCandidate)}</span>
      </div>
    </div>
  );
}

function EngineBanditBucketTable({
  candidates,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
}) {
  if (candidates.length === 0) {
    return null;
  }
  const rows = [...candidates].sort(compareBanditBucketCandidates);
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs tabular-nums">
        <caption className="sr-only">Measured bandit bucket finishes</caption>
        <thead className="text-app-muted">
          <tr className="border-b border-app-border">
            <th className="py-1.5 pr-3 font-semibold">Bucket</th>
            <th className="py-1.5 pr-3 font-semibold">Mean</th>
            <th className="py-1.5 pr-3 font-semibold">Best</th>
            <th className="py-1.5 pr-3 font-semibold">Prob</th>
            <th className="py-1.5 font-semibold">Finishes</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((candidate) => (
            <tr
              className="border-b border-app-border/70 last:border-b-0"
              key={candidate.engine_setting_raw_value}
            >
              <td className="py-1.5 pr-3 text-app-text">
                {engineStepLabel(candidate.engine_setting_raw_value)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {candidate.finish_count <= 0
                  ? "-"
                  : formatRaceTime(candidate.estimated_finish_time_ms)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatOptionalRaceTime(candidate.best_finish_time_ms)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatPercent(candidate.selection_probability)}
              </td>
              <td className="py-1.5 text-app-text">{candidate.finish_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function compareBanditBucketCandidates(
  left: EngineTuningRuntimeCandidateEstimate,
  right: EngineTuningRuntimeCandidateEstimate,
) {
  if (left.finish_count <= 0 && right.finish_count <= 0) {
    return left.engine_setting_raw_value - right.engine_setting_raw_value;
  }
  if (left.finish_count <= 0) {
    return 1;
  }
  if (right.finish_count <= 0) {
    return -1;
  }
  if (left.estimated_finish_time_ms !== right.estimated_finish_time_ms) {
    return left.estimated_finish_time_ms - right.estimated_finish_time_ms;
  }
  if (right.finish_count !== left.finish_count) {
    return right.finish_count - left.finish_count;
  }
  return left.engine_setting_raw_value - right.engine_setting_raw_value;
}

function EngineMeasuredCandidateTable({
  estimates,
  measuredCandidates,
}: {
  estimates: readonly EngineTuningRuntimeCandidateEstimate[];
  measuredCandidates: readonly EngineTuningRuntimeCandidate[];
}) {
  if (measuredCandidates.length === 0) {
    return null;
  }
  const estimateByEngine = new Map(
    estimates.map((estimate) => [estimate.engine_setting_raw_value, estimate]),
  );
  const rows = [...measuredCandidates]
    .sort(compareMeasuredCandidates)
    .slice(0, ENGINE_TUNING_TABLE_ROW_LIMIT);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs tabular-nums">
        <caption className="sr-only">Measured successful engine finishes</caption>
        <thead className="text-app-muted">
          <tr className="border-b border-app-border">
            <th className="py-1.5 pr-3 font-semibold">Engine</th>
            <th className="py-1.5 pr-3 font-semibold">Best</th>
            <th className="py-1.5 pr-3 font-semibold">Estimated</th>
            <th className="py-1.5 pr-3 font-semibold">Prob</th>
            <th className="py-1.5 font-semibold">Finishes</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((candidate) => {
            const estimate = estimateByEngine.get(candidate.engine_setting_raw_value);
            return (
              <tr
                className="border-b border-app-border/70 last:border-b-0"
                key={candidate.engine_setting_raw_value}
              >
                <td className="py-1.5 pr-3 text-app-text">
                  {engineStepLabel(candidate.engine_setting_raw_value)}
                </td>
                <td className="py-1.5 pr-3 text-app-text">
                  {formatOptionalRaceTime(candidate.best_finish_time_ms)}
                </td>
                <td className="py-1.5 pr-3 text-app-text">
                  {estimate === undefined ? "-" : formatRaceTime(estimate.estimated_finish_time_ms)}
                </td>
                <td className="py-1.5 pr-3 text-app-text">
                  {estimate === undefined ? "-" : formatPercent(estimate.selection_probability)}
                </td>
                <td className="py-1.5 text-app-text">{candidate.finish_count}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function EngineModelCandidateTable({
  candidates,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
}) {
  const topCandidates = [...candidates]
    .sort((left, right) => {
      if (left.estimated_finish_time_ms !== right.estimated_finish_time_ms) {
        return left.estimated_finish_time_ms - right.estimated_finish_time_ms;
      }
      return right.selection_probability - left.selection_probability;
    })
    .slice(0, ENGINE_TUNING_TABLE_ROW_LIMIT);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs tabular-nums">
        <thead className="text-app-muted">
          <tr className="border-b border-app-border">
            <th className="py-1.5 pr-3 font-semibold">Engine</th>
            <th className="py-1.5 pr-3 font-semibold">Prob</th>
            <th className="py-1.5 pr-3 font-semibold">Estimated</th>
          </tr>
        </thead>
        <tbody>
          {topCandidates.map((candidate) => (
            <tr
              className="border-b border-app-border/70 last:border-b-0"
              key={candidate.engine_setting_raw_value}
            >
              <td className="py-1.5 pr-3 text-app-text">
                {engineStepLabel(candidate.engine_setting_raw_value)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatPercent(candidate.selection_probability)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatRaceTime(candidate.estimated_finish_time_ms)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function measuredCandidatesForContext(
  candidates: readonly EngineTuningRuntimeCandidate[],
  contextKey: string,
) {
  return candidates.filter(
    (candidate) => candidate.context_key === contextKey && candidate.finish_count > 0,
  );
}

function compareMeasuredCandidates(
  left: EngineTuningRuntimeCandidate,
  right: EngineTuningRuntimeCandidate,
) {
  const leftBest = left.best_finish_time_ms ?? Number.POSITIVE_INFINITY;
  const rightBest = right.best_finish_time_ms ?? Number.POSITIVE_INFINITY;
  if (leftBest !== rightBest) {
    return leftBest - rightBest;
  }
  if (right.finish_count !== left.finish_count) {
    return right.finish_count - left.finish_count;
  }
  return left.engine_setting_raw_value - right.engine_setting_raw_value;
}

function humanizeKey(key: string) {
  return key
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

function formatPercent(value: number | null) {
  if (value === null) {
    return "-";
  }
  const percent = value * 100;
  return `${percent.toFixed(Math.abs(percent) < 1 ? 2 : 1)}%`;
}

function engineStepLabel(step: number) {
  return engineSliderStepLabel(step);
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
