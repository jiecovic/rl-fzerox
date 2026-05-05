import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { SegmentedChoiceStrip } from "@/features/configurator/fields/choices";
import { RunPlotCard, type RunPlotPoint } from "@/features/runs/charts/RunPlotCard";
import {
  fetchFreshRunMetrics,
  getCachedRunMetrics,
  type RunMetricRangeMode,
} from "@/shared/api/client";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";
import { ChevronIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

const RUN_CHART_STYLE = {
  seriesPalette: ["var(--accent)", "var(--run-accent)", "#b7791f", "#7c3aed", "#c2410c", "#0f766e"],
} as const;

const RUN_CHART_GROUPS = [
  { id: "progress", title: "Progress" },
  { id: "rollout", title: "Rollout" },
  { id: "timing", title: "Timing" },
  { id: "optimization", title: "PPO" },
  { id: "reward", title: "Reward" },
  { id: "action", title: "Actions" },
  { id: "state", title: "State" },
  { id: "episode", title: "Episodes" },
  { id: "curriculum", title: "Curriculum" },
  { id: "sampling", title: "Sampling" },
  { id: "other", title: "Other" },
] as const;

const EXPLICIT_CHARTS = [
  {
    id: "episode_reward_mean",
    emptyText: "Waiting for rollout reward samples.",
    group: "rollout",
    metricKeys: ["rollout/ep_rew_mean"],
    title: "Episode reward",
    buildPoints: (_run: ManagedRun, samples: ManagedRunMetricSample[]) =>
      metricPoints(samples, "rollout/ep_rew_mean"),
  },
  {
    id: "episode_length_mean",
    emptyText: "Waiting for rollout length samples.",
    group: "rollout",
    metricKeys: ["rollout/ep_len_mean"],
    title: "Episode length",
    buildPoints: (_run: ManagedRun, samples: ManagedRunMetricSample[]) =>
      metricPoints(samples, "rollout/ep_len_mean"),
  },
  {
    id: "env_step_rate",
    emptyText: "Waiting for timing samples.",
    group: "timing",
    metricKeys: ["time/fps"],
    title: "Env step / s",
    buildPoints: buildEnvStepRatePoints,
  },
] as const satisfies readonly RunChartDescriptor[];

const METRIC_TITLE_OVERRIDES: Record<string, string> = {
  "curriculum/batch_size": "Batch size",
  "curriculum/clip_range": "Clip range",
  "curriculum/ent_coef": "Entropy coefficient",
  "curriculum/learning_rate": "Learning rate",
  "curriculum/n_epochs": "Epochs",
  "episode/airborne_episode_rate": "Airborne episode rate",
  "episode/airborne_failure_rate": "Airborne failure rate",
  "episode/airborne_finish_rate": "Airborne finish rate",
  "episode/boost_pad_entries_mean": "Boost-pad entries",
  "episode/boost_pad_entries_per_lap_mean": "Boost-pad entries / lap",
  "episode/crashed_rate": "Crash rate",
  "episode/energy_depleted_rate": "Energy depleted rate",
  "episode/falling_off_track_rate": "Fell off track rate",
  "episode/final_position_mean": "Final position",
  "episode/finish_position_mean": "Finish position",
  "episode/finish_steps_mean": "Finish steps",
  "episode/finish_time_s_mean": "Finish time (s)",
  "episode/finished_rate": "Finished rate",
  "episode/progress_stalled_rate": "Progress stalled rate",
  "episode/race_laps_completed_mean": "Laps completed",
  "episode/retired_rate": "Retired rate",
  "episode/spinning_out_rate": "Spin-out rate",
  "episode/timeout_rate": "Timeout rate",
  "reward/step_raw_mean": "Raw step reward",
  "reward_clip/abs_excess_mean": "Clip excess",
  "reward_clip/any_step_rate": "Any clip rate",
  "reward_clip/negative_step_rate": "Negative clip rate",
  "reward_clip/positive_step_rate": "Positive clip rate",
  "rollout/ep_len_mean": "Episode length",
  "rollout/ep_rew_mean": "Episode reward",
  "state/airborne_frame_ratio": "Airborne frame ratio",
  "state/boost_pad_entry_step_rate": "Boost-pad entry rate",
  "state/collision_recoil_entry_rate": "Collision recoil rate",
  "state/damage_taken_step_rate": "Damage-taken rate",
  "state/lap_mean": "Lap",
  "state/position_mean": "Position",
  "state/race_distance_mean": "Race distance",
  "state/race_laps_completed_mean": "Laps completed",
  "state/speed_kph_mean": "Speed (kph)",
  "time/fps": "Env step / s",
  "time/iterations": "Iterations",
  "time/time_elapsed": "Elapsed time",
  "time/total_timesteps": "Total steps",
  "track_sampling/coverage": "Track coverage",
  "train/approx_kl": "Approx KL",
  "train/clip_fraction": "Clip fraction",
  "train/clip_range": "Clip range",
  "train/entropy_loss": "Entropy loss",
  "train/explained_variance": "Explained variance",
  "train/learning_rate": "Learning rate",
  "train/loss": "Loss",
  "train/n_updates": "Updates",
  "train/policy_gradient_loss": "Policy gradient loss",
  "train/std": "Policy std",
  "train/value_loss": "Value loss",
};

const LEGACY_SAMPLE_FIELD_BY_METRIC_KEY: Partial<Record<string, keyof ManagedRunMetricSample>> = {
  "rollout/ep_len_mean": "episode_length_mean",
  "rollout/ep_rew_mean": "episode_reward_mean",
  "train/approx_kl": "approx_kl",
  "train/entropy_loss": "entropy_loss",
  "train/policy_gradient_loss": "policy_gradient_loss",
  "train/value_loss": "value_loss",
};

type RunChartGroupId = (typeof RUN_CHART_GROUPS)[number]["id"];

type RunChartDescriptor = {
  id: string;
  emptyText: string;
  group: RunChartGroupId;
  metricKeys?: readonly string[];
  title: string;
  buildPoints: (run: ManagedRun, samples: ManagedRunMetricSample[]) => RunPlotPoint[];
};

type RunChartGroup = {
  id: RunChartGroupId;
  title: string;
  charts: RunChartDescriptor[];
};

type LineageInfo = {
  label: string;
  lineageId: string;
  totalRunCount: number;
};

type LineageRunGroup = LineageInfo & {
  runs: ManagedRun[];
};

type LineageSelectionState = "none" | "partial" | "all";

const INITIAL_GROUP_OPEN: Record<RunChartGroupId, boolean> = {
  action: true,
  curriculum: false,
  episode: true,
  optimization: true,
  other: false,
  progress: true,
  reward: true,
  rollout: true,
  sampling: false,
  state: true,
  timing: true,
};

const CHART_RANGE_OPTIONS: readonly { label: string; value: RunMetricRangeMode }[] = [
  { label: "Recent", value: "recent" },
  { label: "From start", value: "full" },
];

const DEFAULT_CHART_RANGE_MODE: RunMetricRangeMode = "full";
const CHART_SELECTION_STORAGE_KEY = "run-chart-selected-runs";

interface ChartsPanelProps {
  focusedRunId?: string | null;
  onOpenRun?: (run: ManagedRun) => void;
  runs: ManagedRun[];
}

export function ChartsPanel({ focusedRunId = null, onOpenRun, runs }: ChartsPanelProps) {
  const [groupOpen, setGroupOpen] = usePersistentDisclosureMap(
    "run-chart-groups",
    INITIAL_GROUP_OPEN,
  );
  const previousFocusedRunIdRef = useRef<string | null>(focusedRunId);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>(() =>
    readStoredSelectedRunIds(runs, focusedRunId),
  );
  const [rangeMode, setRangeMode] = useState<RunMetricRangeMode>(DEFAULT_CHART_RANGE_MODE);
  const [metricsByRun, setMetricsByRun] = useState<Record<string, ManagedRunMetricSample[]>>(() =>
    cachedMetricsByRun(defaultSelectedRunIds(runs, focusedRunId), DEFAULT_CHART_RANGE_MODE),
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  const setSelectedRuns = useCallback((nextValue: string[] | ((current: string[]) => string[])) => {
    setSelectedRunIds((current) => {
      const next = typeof nextValue === "function" ? nextValue(current) : nextValue;
      writeStoredSelectedRunIds(next);
      return next;
    });
  }, []);

  useEffect(() => {
    const available = new Set(runs.map((run) => run.id));
    setSelectedRuns((current) => {
      const filtered = current.filter((runId) => available.has(runId));
      return filtered.length === 0 ? defaultSelectedRunIds(runs, focusedRunId) : filtered;
    });
  }, [focusedRunId, runs, setSelectedRuns]);

  useEffect(() => {
    const previousFocusedRunId = previousFocusedRunIdRef.current;
    previousFocusedRunIdRef.current = focusedRunId;
    if (focusedRunId === null || focusedRunId === previousFocusedRunId) {
      return;
    }
    if (!runs.some((run) => run.id === focusedRunId)) {
      return;
    }
    setSelectedRuns((current) => [
      focusedRunId,
      ...current.filter((runId) => runId !== focusedRunId),
    ]);
  }, [focusedRunId, runs, setSelectedRuns]);

  useEffect(() => {
    const allowed = new Set(selectedRunIds);
    setMetricsByRun((current) => ({
      ...Object.fromEntries(Object.entries(current).filter(([runId]) => allowed.has(runId))),
      ...cachedMetricsByRun(selectedRunIds, rangeMode),
    }));
  }, [rangeMode, selectedRunIds]);

  useEffect(() => {
    if (selectedRunIds.length === 0) {
      setLoadError(null);
      return undefined;
    }

    let ignore = false;

    async function loadMetrics() {
      try {
        setMetricsByRun((current) => ({
          ...current,
          ...cachedMetricsByRun(selectedRunIds, rangeMode),
        }));
        const samples = await Promise.all(
          selectedRunIds.map(
            async (runId) => [runId, await fetchFreshRunMetrics(runId, rangeMode)] as const,
          ),
        );
        if (!ignore) {
          setMetricsByRun((current) => ({ ...current, ...Object.fromEntries(samples) }));
          setLoadError(null);
        }
      } catch (caught) {
        if (!ignore) {
          setLoadError(caught instanceof Error ? caught.message : "failed to load run metrics");
        }
      }
    }

    void loadMetrics();
    const intervalId = window.setInterval(() => {
      void loadMetrics();
    }, 5_000);
    return () => {
      ignore = true;
      window.clearInterval(intervalId);
    };
  }, [rangeMode, selectedRunIds]);

  const runsById = useMemo(() => new Map(runs.map((run) => [run.id, run] as const)), [runs]);
  const colorByRunId = useMemo(
    () => new Map(runs.map((run, index) => [run.id, chartSeriesColor(index)] as const)),
    [runs],
  );
  const lineageInfoById = useMemo(() => buildLineageInfoById(runs), [runs]);
  const selectionLineageGroups = useMemo(
    () => buildLineageRunGroups(runs, lineageInfoById),
    [lineageInfoById, runs],
  );
  const selectionLineageDisclosureDefaults = useMemo(
    (): Record<string, boolean> =>
      Object.fromEntries(selectionLineageGroups.map((group) => [group.lineageId, true] as const)),
    [selectionLineageGroups],
  );
  const [openLineageById, setOpenLineageById] = usePersistentDisclosureMap(
    "run-chart-selection-open-lineages",
    selectionLineageDisclosureDefaults,
  );
  const selectedRuns = useMemo(
    () =>
      selectedRunIds
        .map((runId) => runsById.get(runId))
        .filter((run): run is ManagedRun => run !== undefined),
    [runsById, selectedRunIds],
  );
  const selectedLineageGroups = useMemo(
    () => buildLineageRunGroups(selectedRuns, lineageInfoById),
    [lineageInfoById, selectedRuns],
  );
  const chartGroups = useMemo(
    () => buildChartGroups(selectedRuns, metricsByRun),
    [metricsByRun, selectedRuns],
  );

  return (
    <Panel>
      <div className="panel-header-row">
        <PanelHeader title="Charts" subtitle="Sampled training metrics." />
        <div className="section-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={() => setSelectedRuns(defaultSelectedRunIds(runs, focusedRunId))}
          >
            Select latest
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => setSelectedRuns(runs.map((run) => run.id))}
          >
            Select all
          </button>
          <button className="secondary-button" type="button" onClick={() => setSelectedRuns([])}>
            Clear
          </button>
          <SegmentedChoiceStrip
            ariaLabel="Chart range"
            options={CHART_RANGE_OPTIONS.map((option) => ({
              active: rangeMode === option.value,
              key: option.value,
              label: option.label,
              onClick: () => setRangeMode(option.value),
            }))}
          />
          <DisclosureToolbar
            collapseLabel="Collapse all chart groups"
            expandLabel="Expand all chart groups"
            onCollapseAll={() => setAllGroupsOpen(false)}
            onExpandAll={() => setAllGroupsOpen(true)}
          />
        </div>
      </div>

      <div className="run-chart-selection-panel">
        <div className="run-chart-selection-header">
          <strong>Selected runs</strong>
          <span>{selectedRunIds.length === 0 ? "none" : `${selectedRunIds.length} active`}</span>
        </div>
        {runs.length === 0 ? (
          <Notice>No launched runs yet.</Notice>
        ) : (
          <div className="run-chart-selection-list">
            {selectionLineageGroups.map((group) => {
              const selectionState = lineageSelectionState(group.runs, selectedRunIds);
              return (
                <section
                  aria-label={`${group.label} lineage runs`}
                  className="run-chart-lineage-group run-chart-selection-lineage-group"
                  data-selection-state={selectionState}
                  key={group.lineageId}
                >
                  <div className="run-chart-lineage-header">
                    <button
                      aria-expanded={openLineageById[group.lineageId] ?? true}
                      aria-label={`${(openLineageById[group.lineageId] ?? true) ? "Collapse" : "Expand"} lineage ${group.label}`}
                      className="run-chart-lineage-disclosure"
                      type="button"
                      onClick={() => toggleCollapsedLineage(group.lineageId)}
                    >
                      <span
                        aria-hidden="true"
                        className={
                          (openLineageById[group.lineageId] ?? true)
                            ? "run-lineage-chevron is-open"
                            : "run-lineage-chevron"
                        }
                      >
                        <ChevronIcon />
                      </span>
                      <span className="run-chart-lineage-heading">
                        <strong>{group.label}</strong>
                        <span className="run-chart-lineage-meta">
                          {group.runs.length === group.totalRunCount
                            ? `${group.totalRunCount} runs`
                            : `${group.runs.length}/${group.totalRunCount} runs`}
                        </span>
                      </span>
                    </button>
                    <div className="run-chart-lineage-header-actions">
                      <LineageSelectionCheckbox
                        label={group.label}
                        state={selectionState}
                        onChange={() => toggleSelectedLineage(group.runs)}
                      />
                    </div>
                  </div>
                  {(openLineageById[group.lineageId] ?? true) ? (
                    <div className="run-chart-lineage-run-grid">
                      {group.runs.map((run, index) => {
                        const checked = selectedRunIds.includes(run.id);
                        const stroke = colorByRunId.get(run.id) ?? chartSeriesColor(index);
                        return (
                          <label className="run-chart-selection-row" key={run.id}>
                            <input
                              checked={checked}
                              type="checkbox"
                              onChange={() => toggleSelectedRun(run.id)}
                            />
                            <span
                              aria-hidden="true"
                              className="run-chart-selection-swatch"
                              style={{ background: stroke }}
                            />
                            <span className="run-chart-selection-copy">
                              <strong>{run.name}</strong>
                              <span>
                                {run.status}
                                {run.runtime === null
                                  ? ""
                                  : ` · ${(run.runtime.progress_fraction * 100).toFixed(1)}% · ${run.runtime.num_timesteps.toLocaleString()} steps`}
                              </span>
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  ) : null}
                </section>
              );
            })}
          </div>
        )}
      </div>

      {loadError !== null ? <Notice tone="error">{loadError}</Notice> : null}

      {selectedRuns.length === 0 ? (
        <Notice>Select at least one run to render comparison plots.</Notice>
      ) : (
        <div className="run-chart-content">
          <section className="run-chart-global-legend" aria-label="Selected run colors">
            {selectedLineageGroups.map((group) => (
              <div
                className="run-chart-lineage-group run-chart-legend-lineage-group"
                key={group.lineageId}
              >
                <div className="run-chart-lineage-header">
                  <strong>{group.label}</strong>
                  <span>{group.runs.length} selected</span>
                </div>
                <ul className="run-chart-lineage-legend-list">
                  {group.runs.map((run, index) => (
                    <li className="run-chart-global-legend-row" key={run.id}>
                      <button
                        className="run-chart-global-legend-button"
                        type="button"
                        onClick={() => onOpenRun?.(run)}
                      >
                        <span
                          aria-hidden="true"
                          className="run-chart-legend-swatch"
                          style={{
                            background: colorByRunId.get(run.id) ?? chartSeriesColor(index),
                          }}
                        />
                        <span className="run-chart-global-legend-name">{run.name}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </section>
          <div className="run-chart-group-stack">
            {chartGroups.map((group) => (
              <ConfigDisclosure
                key={group.id}
                open={groupOpen[group.id]}
                title={group.title}
                onToggle={(open) => setGroupOpen((current) => ({ ...current, [group.id]: open }))}
              >
                <div className="run-chart-grid">
                  {group.charts.map((chart) => (
                    <RunComparisonChart
                      key={chart.id}
                      buildPoints={chart.buildPoints}
                      colorByRunId={colorByRunId}
                      emptyText={chart.emptyText}
                      metricsByRun={metricsByRun}
                      runs={selectedRuns}
                      title={chart.title}
                    />
                  ))}
                </div>
              </ConfigDisclosure>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );

  function toggleSelectedRun(runId: string) {
    setSelectedRuns((current) =>
      current.includes(runId) ? current.filter((value) => value !== runId) : [...current, runId],
    );
  }

  function toggleSelectedLineage(lineageRuns: readonly ManagedRun[]) {
    const lineageRunIds = lineageRuns.map((run) => run.id);
    setSelectedRuns((current) => {
      const currentSet = new Set(current);
      const allSelected = lineageRunIds.every((runId) => currentSet.has(runId));
      if (allSelected) {
        return current.filter((runId) => !lineageRunIds.includes(runId));
      }
      const next = [...current];
      for (const runId of lineageRunIds) {
        if (!currentSet.has(runId)) {
          next.push(runId);
        }
      }
      return next;
    });
  }

  function setAllGroupsOpen(open: boolean) {
    const nextState = RUN_CHART_GROUPS.reduce<Record<RunChartGroupId, boolean>>(
      (state, group) => {
        state[group.id] = open;
        return state;
      },
      { ...INITIAL_GROUP_OPEN },
    );
    setGroupOpen(nextState);
  }

  function toggleCollapsedLineage(lineageId: string) {
    setOpenLineageById((current) => ({
      ...current,
      [lineageId]: !(current[lineageId] ?? true),
    }));
  }
}

function cachedMetricsByRun(runIds: readonly string[], rangeMode: RunMetricRangeMode) {
  return Object.fromEntries(
    runIds
      .map((runId) => [runId, getCachedRunMetrics(runId, rangeMode)] as const)
      .filter((entry): entry is readonly [string, ManagedRunMetricSample[]] => entry[1] !== null),
  );
}

function LineageSelectionCheckbox({
  label,
  onChange,
  state,
}: {
  label: string;
  onChange: () => void;
  state: LineageSelectionState;
}) {
  const checked = state === "all";
  const inputRef = useCallback(
    (element: HTMLInputElement | null) => {
      if (element !== null) {
        element.indeterminate = state === "partial";
      }
    },
    [state],
  );

  return (
    <label className="run-chart-lineage-toggle" title={label}>
      <input
        aria-label={`Select lineage ${label}`}
        checked={checked}
        ref={inputRef}
        type="checkbox"
        onChange={onChange}
      />
    </label>
  );
}

function RunComparisonChart({
  buildPoints,
  colorByRunId,
  emptyText,
  metricsByRun,
  runs,
  title,
}: {
  buildPoints: (run: ManagedRun, samples: ManagedRunMetricSample[]) => RunPlotPoint[];
  colorByRunId: ReadonlyMap<string, string>;
  emptyText: string;
  metricsByRun: Record<string, ManagedRunMetricSample[]>;
  runs: ManagedRun[];
  title: string;
}) {
  const series = useMemo(
    () =>
      runs.map((run, index) => {
        const points = buildPoints(run, metricsByRun[run.id] ?? []);
        return {
          color: colorByRunId.get(run.id) ?? chartSeriesColor(index),
          latest: latestPointValue(points),
          name: run.name,
          points,
          runId: run.id,
        };
      }),
    [buildPoints, colorByRunId, metricsByRun, runs],
  );
  const formatValue = useMemo(
    () => (value: number | null) => formatChartValue(value, title),
    [title],
  );

  return (
    <RunPlotCard emptyText={emptyText} formatValue={formatValue} series={series} title={title} />
  );
}

function buildChartGroups(
  runs: ManagedRun[],
  metricsByRun: Record<string, ManagedRunMetricSample[]>,
): RunChartGroup[] {
  const charts = [...EXPLICIT_CHARTS, ...buildMetricCharts(runs, metricsByRun)];
  return RUN_CHART_GROUPS.map((group) => ({
    id: group.id,
    title: group.title,
    charts: charts.filter((chart) => chart.group === group.id),
  })).filter((group) => group.charts.length > 0);
}

function buildMetricCharts(
  runs: ManagedRun[],
  metricsByRun: Record<string, ManagedRunMetricSample[]>,
): RunChartDescriptor[] {
  const explicitMetricKeys = new Set<string>(
    EXPLICIT_CHARTS.flatMap((chart) => chart.metricKeys ?? []),
  );
  const metricKeys = new Set<string>();
  for (const run of runs) {
    const samples = metricsByRun[run.id] ?? [];
    for (const sample of samples) {
      for (const key of Object.keys(sample.metrics)) {
        if (key !== "time/total_timesteps" && !explicitMetricKeys.has(key)) {
          metricKeys.add(key);
        }
      }
    }
  }
  return [...metricKeys].sort().map((key) => ({
    id: `metric:${key}`,
    emptyText: "Waiting for sampled metrics.",
    group: chartGroupForMetricKey(key),
    title: metricTitle(key),
    buildPoints: (_run: ManagedRun, samples: ManagedRunMetricSample[]) =>
      metricPoints(samples, key),
  }));
}

function defaultSelectedRunIds(runs: ManagedRun[], focusedRunId: string | null) {
  if (runs.length === 0) {
    return [];
  }
  if (focusedRunId !== null && runs.some((run) => run.id === focusedRunId)) {
    return [focusedRunId];
  }
  const selected: string[] = [];
  for (const run of runs) {
    if (!selected.includes(run.id)) {
      selected.push(run.id);
    }
    if (selected.length >= Math.min(2, runs.length)) {
      break;
    }
  }
  return selected;
}

function latestPointValue(points: RunPlotPoint[]) {
  return points.at(-1)?.value ?? null;
}

function metricPoints(samples: ManagedRunMetricSample[], key: string) {
  return samples
    .map((sample) => {
      const value = metricValueFromSample(sample, key);
      return value === undefined ? null : { step: chartStep(sample), value };
    })
    .filter((point): point is RunPlotPoint => point !== null);
}

function metricValueFromSample(sample: ManagedRunMetricSample, key: string) {
  const metricValue = sample.metrics[key];
  if (metricValue !== undefined) {
    return metricValue;
  }
  const legacyField = LEGACY_SAMPLE_FIELD_BY_METRIC_KEY[key];
  if (legacyField === undefined) {
    return undefined;
  }
  const legacyValue = sample[legacyField];
  return typeof legacyValue === "number" ? legacyValue : undefined;
}

function buildEnvStepRatePoints(run: ManagedRun, samples: ManagedRunMetricSample[]) {
  return samples
    .map((sample, index) => {
      const value = sampleEnvStepRateValue(run, samples, index);
      return value === null ? null : { step: chartStep(sample), value };
    })
    .filter((point): point is RunPlotPoint => point !== null);
}

function chartSeriesColor(index: number) {
  return RUN_CHART_STYLE.seriesPalette[index % RUN_CHART_STYLE.seriesPalette.length];
}

function buildLineageInfoById(runs: readonly ManagedRun[]) {
  const runsByLineageId = new Map<string, ManagedRun[]>();
  for (const run of runs) {
    const lineageRuns = runsByLineageId.get(run.lineage_id);
    if (lineageRuns === undefined) {
      runsByLineageId.set(run.lineage_id, [run]);
    } else {
      lineageRuns.push(run);
    }
  }
  return new Map(
    [...runsByLineageId.entries()].map(([lineageId, lineageRuns]) => [
      lineageId,
      {
        label: lineageLabel(lineageRuns),
        lineageId,
        totalRunCount: lineageRuns.length,
      } satisfies LineageInfo,
    ]),
  );
}

function buildLineageRunGroups(
  orderedRuns: readonly ManagedRun[],
  lineageInfoById: ReadonlyMap<string, LineageInfo>,
) {
  const groupsByLineageId = new Map<string, LineageRunGroup>();
  for (const run of orderedRuns) {
    const lineageInfo = lineageInfoById.get(run.lineage_id) ?? {
      label: run.name,
      lineageId: run.lineage_id,
      totalRunCount: 1,
    };
    const existing = groupsByLineageId.get(run.lineage_id);
    if (existing === undefined) {
      groupsByLineageId.set(run.lineage_id, { ...lineageInfo, runs: [run] });
      continue;
    }
    existing.runs.push(run);
  }
  return [...groupsByLineageId.values()];
}

function lineageLabel(runs: readonly ManagedRun[]) {
  const rootCandidates = runs.filter(
    (run) => run.parent_run_id === null && run.source_run_id === null,
  );
  const rootRun =
    [...(rootCandidates.length > 0 ? rootCandidates : runs)].sort(compareRunsAscending).at(0) ??
    null;
  return rootRun?.name ?? "Lineage";
}

function compareRunsAscending(left: ManagedRun, right: ManagedRun) {
  if (left.created_at !== right.created_at) {
    return left.created_at.localeCompare(right.created_at);
  }
  return left.id.localeCompare(right.id);
}

function lineageSelectionState(runs: readonly ManagedRun[], selectedRunIds: readonly string[]) {
  const selected = runs.filter((run) => selectedRunIds.includes(run.id)).length;
  if (selected === 0) {
    return "none";
  }
  if (selected === runs.length) {
    return "all";
  }
  return "partial";
}

function readStoredSelectedRunIds(runs: ManagedRun[], focusedRunId: string | null) {
  const defaults = defaultSelectedRunIds(runs, focusedRunId);
  if (typeof window === "undefined") {
    return defaults;
  }
  try {
    const raw = window.localStorage.getItem(CHART_SELECTION_STORAGE_KEY);
    if (raw === null) {
      return defaults;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return defaults;
    }
    const allowed = new Set(runs.map((run) => run.id));
    const filtered = parsed.filter(
      (value): value is string => typeof value === "string" && allowed.has(value),
    );
    return filtered.length === 0 ? defaults : filtered;
  } catch {
    return defaults;
  }
}

function writeStoredSelectedRunIds(runIds: readonly string[]) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(CHART_SELECTION_STORAGE_KEY, JSON.stringify(runIds));
}

function sampleEnvStepRateValue(run: ManagedRun, samples: ManagedRunMetricSample[], index: number) {
  const sample = samples[index];
  if (sample === undefined) {
    return null;
  }
  const loggedFps = sample.metrics["time/fps"];
  if (loggedFps !== undefined) {
    return loggedFps;
  }
  if (sample.fps !== null && sample.fps !== undefined) {
    return sample.fps;
  }
  const currentMs = Date.parse(sample.created_at);
  if (Number.isNaN(currentMs)) {
    return null;
  }
  const previous = index > 0 ? samples[index - 1] : null;
  if (previous !== null) {
    const previousMs = Date.parse(previous.created_at);
    const elapsedSeconds = (currentMs - previousMs) / 1000;
    if (!Number.isNaN(previousMs) && elapsedSeconds > 0) {
      return Math.max(0, sample.num_timesteps - previous.num_timesteps) / elapsedSeconds;
    }
  }
  const startedAt = run.started_at ?? run.created_at;
  const startedMs = Date.parse(startedAt);
  const elapsedSeconds = (currentMs - startedMs) / 1000;
  if (Number.isNaN(startedMs) || elapsedSeconds <= 0) {
    return null;
  }
  return sample.num_timesteps / elapsedSeconds;
}

function chartStep(sample: ManagedRunMetricSample) {
  return sample.lineage_num_timesteps;
}

function chartGroupForMetricKey(key: string): RunChartGroupId {
  if (key.startsWith("rollout/")) {
    return "rollout";
  }
  if (key.startsWith("time/")) {
    return "timing";
  }
  if (key.startsWith("train/")) {
    return "optimization";
  }
  if (key.startsWith("reward/") || key.startsWith("reward_clip/")) {
    return "reward";
  }
  if (key.startsWith("action/")) {
    return "action";
  }
  if (key.startsWith("state/")) {
    return "state";
  }
  if (key.startsWith("episode/")) {
    return "episode";
  }
  if (key.startsWith("curriculum/")) {
    return "curriculum";
  }
  if (key.startsWith("track_sampling/")) {
    return "sampling";
  }
  return "other";
}

function metricTitle(key: string) {
  const override = METRIC_TITLE_OVERRIDES[key];
  if (override !== undefined) {
    return override;
  }
  const leaf = key.includes("/") ? key.slice(key.lastIndexOf("/") + 1) : key;
  return humanizeMetricToken(leaf);
}

function humanizeMetricToken(token: string) {
  return token
    .replaceAll("_", " ")
    .replaceAll(" kph", " (kph)")
    .replace(/\bep\b/giu, "episode")
    .replace(/\bkl\b/giu, "KL")
    .replace(/\bfps\b/giu, "FPS")
    .replace(/\bs\b/giu, "s")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function formatChartValue(value: number | null, title: string) {
  if (value === null) {
    return "n/a";
  }
  if (title.includes("rate") || title.includes("fraction")) {
    return value.toFixed(3);
  }
  if (title === "Sim / wall") {
    return `${value.toFixed(2)}x`;
  }
  if (Math.abs(value) >= 1_000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}
