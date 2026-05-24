// src/rl_fzerox/apps/run_manager/web/src/features/runs/ChartsPanel.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { SegmentedChoiceStrip } from "@/features/configurator/fields/choices";
import {
  buildChartColorByRunId,
  buildChartGroups,
  buildLineageInfoById,
  buildLineageRunGroups,
  CHART_RANGE_OPTIONS,
  DEFAULT_CHART_RANGE_MODE,
  defaultSelectedRunIds,
  INITIAL_GROUP_OPEN,
  RUN_CHART_GROUPS,
  type RunChartGroupId,
} from "@/features/runs/charts_panel/model";
import { RunChartLegend } from "@/features/runs/charts_panel/RunChartLegend";
import { RunChartSelectionPanel } from "@/features/runs/charts_panel/RunChartSelectionPanel";
import { RunComparisonChart } from "@/features/runs/charts_panel/RunComparisonChart";
import {
  readStoredSelectedRunIds,
  writeStoredSelectedRunIds,
} from "@/features/runs/charts_panel/storage";
import { useRunChartMetrics } from "@/features/runs/charts_panel/useRunChartMetrics";
import type { RunMetricRangeMode } from "@/shared/api/client";
import type { ManagedRun } from "@/shared/api/contract";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

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
  const previousFocusedGroupRunIdRef = useRef<string | null>(focusedRunId);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>(() =>
    readStoredSelectedRunIds(runs, focusedRunId),
  );
  const [rangeMode, setRangeMode] = useState<RunMetricRangeMode>(DEFAULT_CHART_RANGE_MODE);
  const [groupFilter, setGroupFilter] = useState(() => defaultChartGroupFilter(runs, focusedRunId));
  const groupOptions = useMemo(() => chartGroupOptions(runs), [runs]);
  const fallbackGroupFilter = useMemo(
    () => defaultChartGroupFilter(runs, focusedRunId),
    [focusedRunId, runs],
  );
  const visibleRuns = useMemo(
    () => runs.filter((run) => chartGroupValues(run.lineage_groups).includes(groupFilter)),
    [groupFilter, runs],
  );

  const setSelectedRuns = useCallback((nextValue: string[] | ((current: string[]) => string[])) => {
    setSelectedRunIds((current) => {
      const next = typeof nextValue === "function" ? nextValue(current) : nextValue;
      if (sameRunIdList(current, next)) {
        return current;
      }
      writeStoredSelectedRunIds(next);
      return next;
    });
  }, []);

  useEffect(() => {
    const available = new Set(visibleRuns.map((run) => run.id));
    setSelectedRuns((current) => {
      const filtered = current.filter((runId) => available.has(runId));
      const next =
        filtered.length === 0 ? defaultSelectedRunIds(visibleRuns, focusedRunId) : filtered;
      return sameRunIdList(current, next) ? current : next;
    });
  }, [focusedRunId, setSelectedRuns, visibleRuns]);

  useEffect(() => {
    if (groupOptions.some((option) => option.value === groupFilter)) {
      return;
    }
    setGroupFilter(fallbackGroupFilter);
  }, [fallbackGroupFilter, groupFilter, groupOptions]);

  useEffect(() => {
    const previousFocusedRunId = previousFocusedGroupRunIdRef.current;
    if (focusedRunId === null || focusedRunId === previousFocusedRunId) {
      return;
    }
    const focusedRun = runs.find((run) => run.id === focusedRunId);
    if (focusedRun === undefined) {
      return;
    }
    previousFocusedGroupRunIdRef.current = focusedRunId;
    setGroupFilter(chartGroupValues(focusedRun.lineage_groups)[0] ?? fallbackGroupFilter);
  }, [fallbackGroupFilter, focusedRunId, runs]);

  useEffect(() => {
    const previousFocusedRunId = previousFocusedRunIdRef.current;
    previousFocusedRunIdRef.current = focusedRunId;
    if (focusedRunId === null || focusedRunId === previousFocusedRunId) {
      return;
    }
    if (!visibleRuns.some((run) => run.id === focusedRunId)) {
      return;
    }
    setSelectedRuns((current) => [
      focusedRunId,
      ...current.filter((runId) => runId !== focusedRunId),
    ]);
  }, [focusedRunId, setSelectedRuns, visibleRuns]);

  const runsById = useMemo(
    () => new Map(visibleRuns.map((run) => [run.id, run] as const)),
    [visibleRuns],
  );
  const lineageInfoById = useMemo(() => buildLineageInfoById(visibleRuns), [visibleRuns]);
  const selectionLineageGroups = useMemo(
    () => buildLineageRunGroups(visibleRuns, lineageInfoById),
    [lineageInfoById, visibleRuns],
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
  const refreshingRunIds = useMemo(
    () => selectedRuns.filter((run) => run.status === "running").map((run) => run.id),
    [selectedRuns],
  );
  const { loadError, metricsByRun } = useRunChartMetrics(
    selectedRunIds,
    rangeMode,
    refreshingRunIds,
  );
  const colorByRunId = useMemo(
    () => buildChartColorByRunId(visibleRuns, selectedRuns),
    [selectedRuns, visibleRuns],
  );
  const selectedLineageGroups = useMemo(
    () => buildLineageRunGroups(selectedRuns, lineageInfoById),
    [lineageInfoById, selectedRuns],
  );
  const chartColorMode = selectedLineageGroups.length > 1 ? "lineage" : "run";
  const chartGroups = useMemo(
    () => buildChartGroups(selectedRuns, metricsByRun),
    [metricsByRun, selectedRuns],
  );

  const toggleSelectedRun = useCallback(
    (runId: string) => {
      setSelectedRuns((current) =>
        current.includes(runId) ? current.filter((value) => value !== runId) : [...current, runId],
      );
    },
    [setSelectedRuns],
  );

  const toggleSelectedLineage = useCallback(
    (lineageRuns: readonly ManagedRun[]) => {
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
    },
    [setSelectedRuns],
  );

  const setAllGroupsOpen = useCallback(
    (open: boolean) => {
      const nextState = RUN_CHART_GROUPS.reduce<Record<RunChartGroupId, boolean>>(
        (state, group) => {
          state[group.id] = open;
          return state;
        },
        { ...INITIAL_GROUP_OPEN },
      );
      setGroupOpen(nextState);
    },
    [setGroupOpen],
  );

  const toggleCollapsedLineage = useCallback(
    (lineageId: string) => {
      setOpenLineageById((current) => ({
        ...current,
        [lineageId]: !(current[lineageId] ?? true),
      }));
    },
    [setOpenLineageById],
  );

  return (
    <Panel>
      <div className="panel-header-row">
        <PanelHeader title="Charts" subtitle="Sampled training metrics." />
        <div className="section-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={() => setSelectedRuns(defaultSelectedRunIds(visibleRuns, focusedRunId))}
          >
            Select latest
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => setSelectedRuns(visibleRuns.map((run) => run.id))}
          >
            Select all
          </button>
          <button className="secondary-button" type="button" onClick={() => setSelectedRuns([])}>
            Clear
          </button>
          <label className="run-chart-group-filter">
            <span>Group</span>
            <select
              aria-label="Chart lineage group"
              value={groupFilter}
              onChange={(event) => setGroupFilter(event.target.value)}
            >
              {groupOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
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

      <RunChartSelectionPanel
        colorByRunId={colorByRunId}
        openLineageById={openLineageById}
        selectedRunIds={selectedRunIds}
        selectionLineageGroups={selectionLineageGroups}
        totalRuns={visibleRuns.length}
        onToggleCollapsedLineage={toggleCollapsedLineage}
        onToggleLineage={toggleSelectedLineage}
        onToggleRun={toggleSelectedRun}
      />

      {loadError !== null ? <Notice tone="error">{loadError}</Notice> : null}

      {selectedRuns.length === 0 ? (
        <Notice>Select at least one run to render comparison plots.</Notice>
      ) : (
        <div className="run-chart-content">
          <RunChartLegend
            colorMode={chartColorMode}
            colorByRunId={colorByRunId}
            groups={selectedLineageGroups}
            onOpenRun={onOpenRun}
          />
          <div className="run-chart-group-stack">
            {chartGroups.map((group) => (
              <ConfigDisclosure
                key={group.id}
                open={groupOpen[group.id]}
                title={group.title}
                onToggle={(open) => setGroupOpen((current) => ({ ...current, [group.id]: open }))}
              >
                {groupOpen[group.id] ? (
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
                ) : null}
              </ConfigDisclosure>
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}

function chartGroupOptions(runs: readonly ManagedRun[]) {
  return [
    ...new Map(
      runs.flatMap((run) =>
        chartGroupLabels(run.lineage_groups).map((label) => [chartGroupFilterValue(label), label]),
      ),
    ).entries(),
  ]
    .map(([value, label]) => ({ label, value }))
    .sort((left, right) => {
      if (left.value === "ungrouped" || right.value === "ungrouped") {
        return left.value === "ungrouped" && right.value === "ungrouped"
          ? 0
          : left.value === "ungrouped"
            ? 1
            : -1;
      }
      return left.label.localeCompare(right.label);
    });
}

function sameRunIdList(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((runId, index) => runId === right[index]);
}

function defaultChartGroupFilter(runs: readonly ManagedRun[], focusedRunId: string | null) {
  const focusedRun =
    focusedRunId === null ? undefined : runs.find((run) => run.id === focusedRunId);
  return chartGroupValues(focusedRun?.lineage_groups ?? runs[0]?.lineage_groups ?? [])[0] ?? "";
}

function chartGroupLabels(groupNames: readonly string[]) {
  return groupNames.length === 0 ? ["Ungrouped"] : groupNames;
}

function chartGroupValues(groupNames: readonly string[]) {
  return chartGroupLabels(groupNames).map(chartGroupFilterValue);
}

function chartGroupFilterValue(groupName: string) {
  return (
    groupName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "unnamed"
  );
}
