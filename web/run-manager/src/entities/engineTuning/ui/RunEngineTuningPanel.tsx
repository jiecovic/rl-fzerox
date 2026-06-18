// web/run-manager/src/entities/engineTuning/ui/RunEngineTuningPanel.tsx
import { useEffect, useId, useMemo, useState } from "react";

import type {
  ConfigMetadata,
  EngineTuningRuntimeCandidate,
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";
import { ENGINE_SLIDER, engineSliderStepLabel } from "@/shared/domain/engineBuckets";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { ResetIcon } from "@/shared/ui/icons";

const ENGINE_TUNING_TABLE_ROW_LIMIT = 12;

type EngineBanditSortKey =
  | "rank"
  | "bucket"
  | "mean_time"
  | "best_time"
  | "mean_return"
  | "best_return"
  | "completion"
  | "finish_rate"
  | "failure_rate"
  | "probability"
  | "samples"
  | "episodes";
type EngineBanditSortDirection = "asc" | "desc";

interface EngineBanditSortState {
  direction: EngineBanditSortDirection;
  key: EngineBanditSortKey;
}

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
  const objective = state?.objective ?? "finish_time";
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
                    {contextLabel(context, labels)} · {objectiveCountLabel(context, objective)}
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
                  {objectiveCountLabel(selectedContext, objective)}
                </span>
              </div>
              <EngineSamplingProbabilityBars
                candidates={selectedContext.candidates}
                mode={viewMode}
                objective={objective}
              />
              <EngineMeanPerformanceBars
                candidates={selectedContext.candidates}
                mode={viewMode}
                objective={objective}
                recommendedEngineSettingRawValue={
                  selectedContext.recommended_engine_setting_raw_value
                }
              />
            </div>
            {backend === "bandit" ? (
              <EngineBanditBucketTable
                candidates={selectedContext.candidates}
                objective={objective}
              />
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

function objectiveCountLabel(
  context: EngineTuningRuntimeContext,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (objective === "episode_return") {
    return `${context.score_count.toLocaleString()} samples`;
  }
  if (objective === "completion" || objective === "finish_rate") {
    return `${context.episode_count.toLocaleString()} episodes`;
  }
  return `${context.finish_count.toLocaleString()} successful finishes`;
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
          const estimateLabel =
            mode === "bandit" && isHigherScoreObjective(objective)
              ? `mean ${objectiveNoun(objective)}`
              : mode === "bandit"
                ? "mean"
                : "estimated";
          const estimateValue =
            mode === "bandit" && isHigherScoreObjective(objective)
              ? formatObjectiveScore(candidate.mean_score, objective)
              : formatRaceTime(candidate.estimated_finish_time_ms);
          const bestValue =
            mode === "bandit" && isHigherScoreObjective(objective)
              ? formatOptionalObjectiveScore(candidate.best_score, objective)
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

function EngineMeanPerformanceBars({
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
  const estimatedTimes = candidates.map((candidate) => candidate.estimated_finish_time_ms);
  const fastestEstimate = Math.min(...estimatedTimes, Number.POSITIVE_INFINITY);
  const slowestEstimate = Math.max(...estimatedTimes, 1);
  const timeSpan = Math.max(1, slowestEstimate - fastestEstimate);
  const returnScores = candidates.map((candidate) => candidate.mean_score);
  const bestReturn = Math.max(...returnScores, 0);
  const worstReturn = Math.min(...returnScores, bestReturn - 1);
  const returnSpan = Math.max(0.000001, bestReturn - worstReturn);
  const showsScore = mode === "bandit" && isHigherScoreObjective(objective);

  return (
    <div className="grid gap-1">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-app-muted">
        <div className="flex flex-wrap gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent/45" aria-hidden="true" />
            {showsScore
              ? `bucket mean ${objectiveNoun(objective)}`
              : mode === "bandit"
                ? "bucket mean finish"
                : "predicted mean performance"}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-3 bg-app-accent" aria-hidden="true" />
            {mode === "bandit" ? "greedy bucket" : "deterministic greedy"}
          </span>
        </div>
        <span className="tabular-nums">
          {showsScore
            ? `bucket mean ${objectiveNoun(objective)} ${formatObjectiveScore(worstReturn, objective)}-${formatObjectiveScore(bestReturn, objective)}`
            : `${mode === "bandit" ? "bucket mean finish time" : "estimated finish time"} ${formatRaceTime(fastestEstimate)}-${formatRaceTime(slowestEstimate)}`}
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
            ? showsScore
              ? `Measured bucket mean ${objectiveNoun(objective)}`
              : "Measured bucket mean finish time"
            : "Predicted mean performance from estimated finish time"}
        </title>
        {candidates.map((candidate, index) => {
          const isRecommended =
            candidate.engine_setting_raw_value === recommendedEngineSettingRawValue;
          const height = showsScore
            ? 3 + ((candidate.mean_score - worstReturn) / returnSpan) * 93
            : 3 + ((slowestEstimate - candidate.estimated_finish_time_ms) / timeSpan) * 93;
          const width = Math.max(0.2, barWidth * 0.86);
          const x = index * barWidth + (barWidth - width) / 2;
          const y = 100 - height;
          const rawLabel = mode === "bandit" ? "bucket" : "engine";
          const engineLabel = engineStepLabel(candidate.engine_setting_raw_value);
          const estimateLabel = showsScore
            ? `mean ${objectiveNoun(objective)}`
            : mode === "bandit"
              ? "mean"
              : "estimated";
          const estimateValue = showsScore
            ? formatObjectiveScore(candidate.mean_score, objective)
            : formatRaceTime(candidate.estimated_finish_time_ms);
          const bestValue = showsScore
            ? formatOptionalObjectiveScore(candidate.best_score, objective)
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
          {showsScore
            ? `bucket mean ${objectiveNoun(objective)}`
            : mode === "bandit"
              ? "bucket mean finish"
              : "estimated finish"}
        </span>
        <span>{engineStepLabel(lastCandidate)}</span>
      </div>
    </div>
  );
}

function EngineBanditBucketTable({
  candidates,
  objective,
}: {
  candidates: readonly EngineTuningRuntimeCandidateEstimate[];
  objective: EngineTuningRuntimeState["objective"];
}) {
  const [sortState, setSortState] = useState<EngineBanditSortState>({
    direction: "asc",
    key: "rank",
  });
  if (candidates.length === 0) {
    return null;
  }
  const rows = [...candidates].sort((left, right) =>
    compareBanditBucketCandidatesForSort(left, right, objective, sortState),
  );
  const bestValues = engineBanditBestColumnValues(candidates, objective);
  function setSortKey(key: EngineBanditSortKey) {
    setSortState((current) =>
      current.key === key
        ? { key, direction: current.direction === "asc" ? "desc" : "asc" }
        : { key, direction: defaultBanditSortDirection(key) },
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs tabular-nums">
        <caption className="sr-only">Measured bandit bucket samples</caption>
        <thead className="text-app-muted">
          <tr className="border-b border-app-border">
            <SortableEngineBanditHeader
              active={sortState.key === "bucket"}
              direction={sortState.direction}
              label="Bucket"
              sortKey="bucket"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "mean_time"}
              direction={sortState.direction}
              label="Mean time"
              sortKey="mean_time"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "best_time"}
              direction={sortState.direction}
              label="Best time"
              sortKey="best_time"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "mean_return"}
              direction={sortState.direction}
              label="Mean return"
              sortKey="mean_return"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "best_return"}
              direction={sortState.direction}
              label="Best return"
              sortKey="best_return"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "completion"}
              direction={sortState.direction}
              label="Completion"
              sortKey="completion"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "finish_rate"}
              direction={sortState.direction}
              label="Finish"
              sortKey="finish_rate"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "failure_rate"}
              direction={sortState.direction}
              label="Fail"
              sortKey="failure_rate"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "probability"}
              direction={sortState.direction}
              label="Prob"
              sortKey="probability"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "samples"}
              direction={sortState.direction}
              label="Samples"
              sortKey="samples"
              onSort={setSortKey}
            />
            <SortableEngineBanditHeader
              active={sortState.key === "episodes"}
              direction={sortState.direction}
              label="Episodes"
              sortKey="episodes"
              onSort={setSortKey}
              trailing
            />
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
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.mean_finish_time_ms, bestValues.meanTime),
                )}
              >
                {formatOptionalRaceTime(candidate.mean_finish_time_ms)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.best_finish_time_ms, bestValues.bestTime),
                )}
              >
                {formatOptionalRaceTime(candidate.best_finish_time_ms)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.mean_return_score, bestValues.meanReturn),
                )}
              >
                {formatOptionalScore(candidate.mean_return_score)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.best_return_score, bestValues.bestReturn),
                )}
              >
                {formatOptionalScore(candidate.best_return_score)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.mean_completion_score, bestValues.completion),
                )}
              >
                {formatPercent(candidate.mean_completion_score)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.finish_rate, bestValues.finishRate),
                )}
              >
                {formatPercent(candidate.finish_rate)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.failure_rate, bestValues.failureRate),
                )}
              >
                {formatPercent(candidate.failure_rate)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.selection_probability, bestValues.probability),
                )}
              >
                {formatPercent(candidate.selection_probability)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidateSampleCount(candidate, objective), bestValues.samples),
                  true,
                )}
              >
                {candidateSampleCount(candidate, objective)}
              </td>
              <td
                className={engineBanditValueCellClass(
                  isBestBanditValue(candidate.episode_count, bestValues.episodes),
                  true,
                )}
              >
                {candidate.episode_count}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface EngineBanditBestColumnValues {
  bestReturn: number | null;
  bestTime: number | null;
  completion: number | null;
  episodes: number | null;
  failureRate: number | null;
  finishRate: number | null;
  meanReturn: number | null;
  meanTime: number | null;
  probability: number | null;
  samples: number | null;
}

function engineBanditBestColumnValues(
  candidates: readonly EngineTuningRuntimeCandidateEstimate[],
  objective: EngineTuningRuntimeState["objective"],
): EngineBanditBestColumnValues {
  return {
    bestReturn: maxNullable(candidates.map((candidate) => candidate.best_return_score)),
    bestTime: minNullable(candidates.map((candidate) => candidate.best_finish_time_ms)),
    completion: maxNullable(candidates.map((candidate) => candidate.mean_completion_score)),
    episodes: positiveMaxNullable(candidates.map((candidate) => candidate.episode_count)),
    failureRate: minNullable(candidates.map((candidate) => candidate.failure_rate)),
    finishRate: maxNullable(candidates.map((candidate) => candidate.finish_rate)),
    meanReturn: maxNullable(candidates.map((candidate) => candidate.mean_return_score)),
    meanTime: minNullable(candidates.map((candidate) => candidate.mean_finish_time_ms)),
    probability: maxNullable(candidates.map((candidate) => candidate.selection_probability)),
    samples: positiveMaxNullable(
      candidates.map((candidate) => candidateSampleCount(candidate, objective)),
    ),
  };
}

function engineBanditValueCellClass(isBest: boolean, trailing = false) {
  return `${trailing ? "py-1.5" : "py-1.5 pr-3"} text-app-text${isBest ? " font-semibold" : ""}`;
}

function isBestBanditValue(value: number | null, best: number | null) {
  return best !== null && value === best;
}

function SortableEngineBanditHeader({
  active,
  direction,
  label,
  sortKey,
  trailing = false,
  onSort,
}: {
  active: boolean;
  direction: EngineBanditSortDirection;
  label: string;
  sortKey: EngineBanditSortKey;
  trailing?: boolean;
  onSort: (key: EngineBanditSortKey) => void;
}) {
  return (
    <th
      aria-sort={active ? (direction === "asc" ? "ascending" : "descending") : "none"}
      className={trailing ? "py-1.5 font-semibold" : "py-1.5 pr-3 font-semibold"}
    >
      <button
        className="inline-flex items-center gap-1 text-left text-inherit hover:text-app-text"
        type="button"
        onClick={() => onSort(sortKey)}
      >
        <span>{label}</span>
        <span className="min-w-7 text-[10px] font-semibold text-app-muted" aria-hidden="true">
          {active ? direction : ""}
        </span>
      </button>
    </th>
  );
}

function compareBanditBucketCandidatesForSort(
  left: EngineTuningRuntimeCandidateEstimate,
  right: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
  sortState: EngineBanditSortState,
) {
  if (sortState.key === "rank") {
    return compareBanditBucketCandidates(left, right, objective);
  }
  const compared = compareBanditBucketSortValue(left, right, objective, sortState);
  if (compared !== 0) {
    return compared;
  }
  return compareBanditBucketCandidates(left, right, objective);
}

function compareBanditBucketSortValue(
  left: EngineTuningRuntimeCandidateEstimate,
  right: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
  sortState: EngineBanditSortState,
) {
  if (sortState.key === "bucket") {
    return (
      sortDirectionMultiplier(sortState.direction) *
      numericCompare(left.engine_setting_raw_value, right.engine_setting_raw_value)
    );
  }
  if (sortState.key === "probability") {
    return (
      sortDirectionMultiplier(sortState.direction) *
      numericCompare(left.selection_probability, right.selection_probability)
    );
  }
  if (sortState.key === "samples") {
    return (
      sortDirectionMultiplier(sortState.direction) *
      numericCompare(candidateSampleCount(left, objective), candidateSampleCount(right, objective))
    );
  }
  if (sortState.key === "episodes") {
    return (
      sortDirectionMultiplier(sortState.direction) *
      numericCompare(left.episode_count, right.episode_count)
    );
  }
  return compareNullableSortValues(
    banditSortMetricValue(left, objective, sortState.key),
    banditSortMetricValue(right, objective, sortState.key),
    sortState.direction,
  );
}

function banditSortMetricValue(
  candidate: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
  key: EngineBanditSortKey,
) {
  if (banditMetricObservationCount(candidate, objective, key) <= 0) {
    return null;
  }
  if (key === "mean_time") {
    return candidate.mean_finish_time_ms;
  }
  if (key === "best_time") {
    return candidate.best_finish_time_ms;
  }
  if (key === "mean_return") {
    return candidate.mean_return_score;
  }
  if (key === "best_return") {
    return candidate.best_return_score;
  }
  if (key === "completion") {
    return candidate.mean_completion_score;
  }
  if (key === "finish_rate") {
    return candidate.finish_rate;
  }
  if (key === "failure_rate") {
    return candidate.failure_rate;
  }
  return null;
}

function banditMetricObservationCount(
  candidate: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
  key: EngineBanditSortKey,
) {
  if (key === "mean_time" || key === "best_time") {
    return candidate.finish_count;
  }
  if (key === "mean_return" || key === "best_return") {
    return candidate.return_count;
  }
  if (key === "completion" || key === "finish_rate" || key === "failure_rate") {
    return candidate.episode_count;
  }
  return candidateSampleCount(candidate, objective);
}

function candidateSampleCount(
  candidate: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (objective === "completion" || objective === "finish_rate") {
    return candidate.episode_count;
  }
  return objective === "episode_return" ? candidate.score_count : candidate.finish_count;
}

function compareNullableSortValues(
  left: number | null,
  right: number | null,
  direction: EngineBanditSortDirection,
) {
  if (left === null && right === null) {
    return 0;
  }
  if (left === null) {
    return 1;
  }
  if (right === null) {
    return -1;
  }
  return sortDirectionMultiplier(direction) * numericCompare(left, right);
}

function defaultBanditSortDirection(key: EngineBanditSortKey): EngineBanditSortDirection {
  if (key === "bucket" || key === "mean_time" || key === "best_time" || key === "failure_rate") {
    return "asc";
  }
  return "desc";
}

function sortDirectionMultiplier(direction: EngineBanditSortDirection) {
  return direction === "asc" ? 1 : -1;
}

function numericCompare(left: number, right: number) {
  return left - right;
}

function minNullable(values: readonly (number | null)[]) {
  const measured = values.filter((value): value is number => value !== null);
  return measured.length === 0 ? null : Math.min(...measured);
}

function maxNullable(values: readonly (number | null)[]) {
  const measured = values.filter((value): value is number => value !== null);
  return measured.length === 0 ? null : Math.max(...measured);
}

function positiveMaxNullable(values: readonly number[]) {
  const maxValue = Math.max(...values, 0);
  return maxValue <= 0 ? null : maxValue;
}

function compareBanditBucketCandidates(
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

function nullableRaceTimeSortValue(value: number | null) {
  return value ?? Number.POSITIVE_INFINITY;
}

function nullableScoreSortValue(value: number | null) {
  return value ?? Number.NEGATIVE_INFINITY;
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

function isHigherScoreObjective(objective: EngineTuningRuntimeState["objective"]) {
  return objective !== "finish_time";
}

function objectiveNoun(objective: EngineTuningRuntimeState["objective"]) {
  if (objective === "episode_return") {
    return "return";
  }
  if (objective === "completion") {
    return "completion";
  }
  if (objective === "finish_rate") {
    return "finish rate";
  }
  return "finish";
}

function formatObjectiveScore(value: number, objective: EngineTuningRuntimeState["objective"]) {
  if (objective === "completion" || objective === "finish_rate") {
    return formatPercent(value);
  }
  return formatScore(value);
}

function formatOptionalObjectiveScore(
  value: number | null,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (value === null) {
    return objective === "completion" || objective === "finish_rate" ? "-" : "no score yet";
  }
  return formatObjectiveScore(value, objective);
}

function objectiveSampleLabel(
  candidate: EngineTuningRuntimeCandidateEstimate,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (objective === "episode_return") {
    return `${candidate.score_count} return samples`;
  }
  if (objective === "completion" || objective === "finish_rate") {
    return `${candidate.episode_count} episodes`;
  }
  return `${candidate.finish_count} successful finishes`;
}

function engineStepLabel(step: number) {
  return engineSliderStepLabel(step);
}

function formatOptionalRaceTime(value: number | null) {
  return value === null ? "no time yet" : formatRaceTime(value);
}

function formatOptionalScore(value: number | null) {
  return value === null ? "no score yet" : formatScore(value);
}

function formatScore(value: number) {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: Math.abs(value) < 10 ? 2 : 1,
    minimumFractionDigits: 0,
  });
}

function formatRaceTime(value: number) {
  const safeValue = Math.max(0, Math.round(value));
  const minutes = Math.floor(safeValue / 60_000);
  const seconds = Math.floor((safeValue % 60_000) / 1000);
  const millis = safeValue % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}
