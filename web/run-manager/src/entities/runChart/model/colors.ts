// web/run-manager/src/entities/runChart/model/colors.ts
import type { ManagedRun } from "@/shared/api/contract";

export type ChartColorMode = "branch" | "lineage" | "run";

const RUN_CHART_SERIES_PALETTE = [
  "var(--accent)",
  "var(--run-accent)",
  "#b7791f",
  "#7c3aed",
  "#c2410c",
  "#0f766e",
] as const;

export function chartSeriesColor(index: number) {
  return RUN_CHART_SERIES_PALETTE[index % RUN_CHART_SERIES_PALETTE.length];
}

export function buildChartColorByRunId(
  allRuns: readonly ManagedRun[],
  selectedRuns: readonly ManagedRun[],
  mode: ChartColorMode = "run",
) {
  if (mode === "lineage") {
    return lineageColorByRunId(allRuns);
  }
  if (mode === "branch") {
    return branchColorByRunId(allRuns);
  }
  return selectedRunColorByRunId(allRuns, selectedRuns);
}

export function buildBranchGroupKeyByRunId(runs: readonly ManagedRun[]) {
  const runById = new Map(runs.map((run) => [run.id, run]));
  const childrenByParentId = new Map<string, ManagedRun[]>();
  for (const run of runs) {
    const parentId = lineageParentId(run, runById);
    if (parentId === null) {
      continue;
    }
    const children = childrenByParentId.get(parentId);
    if (children === undefined) {
      childrenByParentId.set(parentId, [run]);
    } else {
      children.push(run);
    }
  }
  const groupKeyByRunId = new Map<string, string>();

  function groupKeyFor(run: ManagedRun, visiting: ReadonlySet<string>): string {
    const existing = groupKeyByRunId.get(run.id);
    if (existing !== undefined) {
      return existing;
    }
    const parentId = lineageParentId(run, runById);
    if (parentId === null || visiting.has(run.id)) {
      const rootKey = `lineage:${run.lineage_id}`;
      groupKeyByRunId.set(run.id, rootKey);
      return rootKey;
    }
    const parent = runById.get(parentId);
    if (parent === undefined) {
      const rootKey = `lineage:${run.lineage_id}`;
      groupKeyByRunId.set(run.id, rootKey);
      return rootKey;
    }
    const siblings = childrenByParentId.get(parentId) ?? [];
    const groupKey =
      siblings.length > 1
        ? `branch:${run.id}`
        : groupKeyFor(parent, new Set([...visiting, run.id]));
    groupKeyByRunId.set(run.id, groupKey);
    return groupKey;
  }

  for (const run of runs) {
    groupKeyFor(run, new Set());
  }
  return groupKeyByRunId;
}

function selectedRunColorByRunId(
  allRuns: readonly ManagedRun[],
  selectedRuns: readonly ManagedRun[],
) {
  const colorByRunId = new Map<string, string>();
  for (const [index, run] of allRuns.entries()) {
    colorByRunId.set(run.id, chartSeriesColor(index));
  }
  for (const [index, run] of selectedRuns.entries()) {
    colorByRunId.set(run.id, chartSeriesColor(index));
  }
  return colorByRunId;
}

function lineageColorByRunId(runs: readonly ManagedRun[]) {
  const colorByLineageId = new Map<string, string>();
  for (const [index, lineageId] of orderedUniqueLineageIds(runs).entries()) {
    colorByLineageId.set(lineageId, chartSeriesColor(index));
  }
  return new Map(
    runs.map((run) => [run.id, colorByLineageId.get(run.lineage_id) ?? chartSeriesColor(0)]),
  );
}

function branchColorByRunId(runs: readonly ManagedRun[]) {
  const groupKeyByRunId = buildBranchGroupKeyByRunId(runs);
  const orderedGroupKeys = orderedUniqueBranchGroupKeys(runs, groupKeyByRunId);
  const colorByGroupKey = new Map(
    orderedGroupKeys.map((groupKey, index) => [groupKey, chartSeriesColor(index)] as const),
  );
  return new Map(
    runs.map((run) => [
      run.id,
      colorByGroupKey.get(groupKeyByRunId.get(run.id) ?? "") ?? chartSeriesColor(0),
    ]),
  );
}

function orderedUniqueLineageIds(runs: readonly ManagedRun[]) {
  return [...new Set(runs.map((run) => run.lineage_id))].sort((left, right) =>
    left.localeCompare(right),
  );
}

function orderedUniqueBranchGroupKeys(
  runs: readonly ManagedRun[],
  groupKeyByRunId: ReadonlyMap<string, string>,
) {
  return [
    ...new Set(
      [...runs]
        .sort(compareRunsAscending)
        .map((run) => groupKeyByRunId.get(run.id))
        .filter((groupKey): groupKey is string => groupKey !== undefined),
    ),
  ];
}

function lineageParentId(run: ManagedRun, runById: ReadonlyMap<string, ManagedRun>) {
  const parentRunId = run.parent_run_id ?? run.source_run_id;
  if (parentRunId === null) {
    return null;
  }
  const parent = runById.get(parentRunId);
  return parent?.lineage_id === run.lineage_id ? parentRunId : null;
}

function compareRunsAscending(left: ManagedRun, right: ManagedRun) {
  if (left.created_at !== right.created_at) {
    return left.created_at.localeCompare(right.created_at);
  }
  return left.id.localeCompare(right.id);
}
