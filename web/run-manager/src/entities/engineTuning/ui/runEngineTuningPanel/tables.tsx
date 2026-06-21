// web/run-manager/src/entities/engineTuning/ui/runEngineTuningPanel/tables.tsx
import { useState } from "react";

import type {
  EngineTuningRuntimeCandidate,
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";

import {
  engineStepLabel,
  formatOptionalRaceTime,
  formatOptionalScore,
  formatPercent,
  formatRaceTime,
} from "./format";
import { candidateSampleCount, compareBanditBucketCandidates } from "./metrics";

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
  | "episodes";
type EngineBanditSortDirection = "asc" | "desc";

interface EngineBanditSortState {
  direction: EngineBanditSortDirection;
  key: EngineBanditSortKey;
}

export function EngineBanditBucketTable({
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
  const bestValues = engineBanditBestColumnValues(candidates);
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

export function EngineMeasuredCandidateTable({
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

export function EngineModelCandidateTable({
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

function engineBanditBestColumnValues(
  candidates: readonly EngineTuningRuntimeCandidateEstimate[],
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
  };
}

function engineBanditValueCellClass(isBest: boolean, trailing = false) {
  return `${trailing ? "py-1.5" : "py-1.5 pr-3"} text-app-text${isBest ? " font-semibold" : ""}`;
}

function isBestBanditValue(value: number | null, best: number | null) {
  return best !== null && value === best;
}

function maxNullable(values: readonly (number | null)[]) {
  const measured = values.filter((value): value is number => value !== null);
  return measured.length === 0 ? null : Math.max(...measured);
}

function minNullable(values: readonly (number | null)[]) {
  const measured = values.filter((value): value is number => value !== null);
  return measured.length === 0 ? null : Math.min(...measured);
}

function numericCompare(left: number, right: number) {
  return left - right;
}

function positiveMaxNullable(values: readonly number[]) {
  const maxValue = Math.max(...values, 0);
  return maxValue <= 0 ? null : maxValue;
}

function sortDirectionMultiplier(direction: EngineBanditSortDirection) {
  return direction === "asc" ? 1 : -1;
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
