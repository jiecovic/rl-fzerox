// web/run-manager/src/widgets/runCharts/ChartsPanel.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  buildBranchRunGroups,
  buildChartColorByRunId,
  buildChartGroups,
  buildLineageInfoById,
  buildLineageRunGroups,
  CHART_RANGE_OPTIONS,
  type ChartColorMode,
  DEFAULT_CHART_RANGE_MODE,
  defaultSelectedRunIds,
  INITIAL_GROUP_OPEN,
  RUN_CHART_GROUPS,
  type RunChartGroupId,
} from "@/entities/runChart/model";
import {
  readStoredSelectedRunIds,
  writeStoredSelectedRunIds,
} from "@/features/runChartMetrics/model/storage";
import { useRunChartMetrics } from "@/features/runChartMetrics/model/useRunChartMetrics";
import type { RunMetricRangeMode } from "@/shared/api/client";
import type { ManagedRun } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/shared/ui/config/disclosureState";
import { SegmentedChoiceStrip } from "@/shared/ui/configFields/choices";
import { FieldSelect } from "@/shared/ui/Field";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RunChartLegend } from "@/widgets/runCharts/chartsPanel/RunChartLegend";
import { RunChartSelectionPanel } from "@/widgets/runCharts/chartsPanel/RunChartSelectionPanel";
import {
  RunComparisonChart,
  type RunComparisonSeriesGroup,
} from "@/widgets/runCharts/chartsPanel/RunComparisonChart";

interface ChartsPanelProps {
  focusedRunId?: string | null;
  onGlobalError: (message: string | null) => void;
  onOpenRun?: (run: ManagedRun) => void;
  runs: ManagedRun[];
}

export function ChartsPanel({
  focusedRunId = null,
  onGlobalError,
  onOpenRun,
  runs,
}: ChartsPanelProps) {
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
  const [colorMode, setColorMode] = useState<ChartColorMode>("branch");
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
  useEffect(() => {
    if (loadError !== null) {
      onGlobalError(loadError);
    }
  }, [loadError, onGlobalError]);
  const colorByRunId = useMemo(
    () => buildChartColorByRunId(visibleRuns, selectedRuns, colorMode),
    [colorMode, selectedRuns, visibleRuns],
  );
  const selectedLineageGroups = useMemo(
    () => buildLineageRunGroups(selectedRuns, lineageInfoById),
    [lineageInfoById, selectedRuns],
  );
  const selectedBranchGroups = useMemo(
    () => buildBranchRunGroups(selectedRuns, colorByRunId),
    [colorByRunId, selectedRuns],
  );
  const comparisonSeriesGroups = useMemo(
    () =>
      comparisonGroupsForMode({
        branchGroups: selectedBranchGroups,
        colorByRunId,
        colorMode,
        lineageGroups: selectedLineageGroups,
        selectedRuns,
      }),
    [colorByRunId, colorMode, selectedBranchGroups, selectedLineageGroups, selectedRuns],
  );
  const comparisonSeriesUnit =
    colorMode === "branch" ? "branches" : colorMode === "lineage" ? "lineages" : "runs";
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
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Button
            type="button"
            onClick={() => setSelectedRuns(defaultSelectedRunIds(visibleRuns, focusedRunId))}
          >
            Select latest
          </Button>
          <Button type="button" onClick={() => setSelectedRuns(visibleRuns.map((run) => run.id))}>
            Select all
          </Button>
          <Button type="button" onClick={() => setSelectedRuns([])}>
            Clear
          </Button>
          <div className="inline-flex items-center gap-1.5 text-xs text-app-muted">
            <span>Group</span>
            <FieldSelect
              aria-label="Chart lineage group"
              className="h-8 w-auto rounded-md px-2 text-xs"
              value={groupFilter}
              onChange={(event) => setGroupFilter(event.target.value)}
            >
              {groupOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </FieldSelect>
          </div>
          <SegmentedChoiceStrip
            ariaLabel="Chart range"
            options={CHART_RANGE_OPTIONS.map((option) => ({
              active: rangeMode === option.value,
              key: option.value,
              label: option.label,
              onClick: () => setRangeMode(option.value),
            }))}
          />
          <SegmentedChoiceStrip
            ariaLabel="Chart color mode"
            options={[
              {
                active: colorMode === "branch",
                key: "branch",
                label: "Branches",
                onClick: () => setColorMode("branch"),
              },
              {
                active: colorMode === "run",
                key: "run",
                label: "Runs",
                onClick: () => setColorMode("run"),
              },
              {
                active: colorMode === "lineage",
                key: "lineage",
                label: "Lineages",
                onClick: () => setColorMode("lineage"),
              },
            ]}
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

      {selectedRuns.length === 0 ? (
        <Notice>Select at least one run to render comparison plots.</Notice>
      ) : (
        <div className="grid gap-3.5">
          <RunChartLegend
            branchGroups={selectedBranchGroups}
            colorMode={colorMode}
            colorByRunId={colorByRunId}
            groups={selectedLineageGroups}
            onOpenRun={onOpenRun}
          />
          <div className="grid gap-3.5 [&_.config-disclosure-body]:pt-3.5">
            {chartGroups.map((group) => (
              <ConfigDisclosure
                key={group.id}
                open={groupOpen[group.id]}
                title={group.title}
                onToggle={(open) => setGroupOpen((current) => ({ ...current, [group.id]: open }))}
              >
                {groupOpen[group.id] ? (
                  <div className="grid grid-cols-2 gap-3.5 max-[1100px]:grid-cols-1">
                    {group.charts.map((chart) => (
                      <RunComparisonChart
                        key={chart.id}
                        buildPoints={chart.buildPoints}
                        emptyText={chart.emptyText}
                        metricsByRun={metricsByRun}
                        runs={selectedRuns}
                        seriesGroups={comparisonSeriesGroups}
                        seriesUnit={comparisonSeriesUnit}
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

function comparisonGroupsForMode({
  branchGroups,
  colorByRunId,
  colorMode,
  lineageGroups,
  selectedRuns,
}: {
  branchGroups: ReturnType<typeof buildBranchRunGroups>;
  colorByRunId: ReadonlyMap<string, string>;
  colorMode: ChartColorMode;
  lineageGroups: ReturnType<typeof buildLineageRunGroups>;
  selectedRuns: readonly ManagedRun[];
}): RunComparisonSeriesGroup[] {
  if (colorMode === "branch") {
    return branchGroups.map((group) => ({
      color: group.color,
      id: group.id,
      label: group.label,
      runIds: group.runs.map((run) => run.id),
    }));
  }
  if (colorMode === "lineage") {
    return lineageGroups.map((group) => ({
      color: colorByRunId.get(group.runs[0]?.id ?? "") ?? "var(--accent)",
      id: group.lineageId,
      label: group.label,
      runIds: group.runs.map((run) => run.id),
    }));
  }
  return selectedRuns.map((run) => ({
    color: colorByRunId.get(run.id) ?? "var(--accent)",
    id: run.id,
    label: run.name,
    runIds: [run.id],
  }));
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
