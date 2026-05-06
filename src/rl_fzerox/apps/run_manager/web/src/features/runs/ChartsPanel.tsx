// src/rl_fzerox/apps/run_manager/web/src/features/runs/ChartsPanel.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { SegmentedChoiceStrip } from "@/features/configurator/fields/choices";
import {
  buildChartGroups,
  buildLineageInfoById,
  buildLineageRunGroups,
  CHART_RANGE_OPTIONS,
  chartSeriesColor,
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
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>(() =>
    readStoredSelectedRunIds(runs, focusedRunId),
  );
  const [rangeMode, setRangeMode] = useState<RunMetricRangeMode>(DEFAULT_CHART_RANGE_MODE);
  const { loadError, metricsByRun } = useRunChartMetrics(selectedRunIds, rangeMode);

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

      <RunChartSelectionPanel
        colorByRunId={colorByRunId}
        openLineageById={openLineageById}
        selectedRunIds={selectedRunIds}
        selectionLineageGroups={selectionLineageGroups}
        totalRuns={runs.length}
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
