// web/run-manager/src/entities/runChart/model/lineages.ts
import type { ManagedRun } from "@/shared/api/contract";

export type LineageInfo = {
  label: string;
  lineageId: string;
  totalRunCount: number;
};

export type LineageRunGroup = LineageInfo & {
  runs: ManagedRun[];
};

export type LineageSelectionState = "none" | "partial" | "all";

export function buildLineageInfoById(runs: readonly ManagedRun[]) {
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

export function buildLineageRunGroups(
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

export function lineageSelectionState(
  runs: readonly ManagedRun[],
  selectedRunIds: readonly string[],
) {
  const selected = runs.filter((run) => selectedRunIds.includes(run.id)).length;
  if (selected === 0) {
    return "none";
  }
  if (selected === runs.length) {
    return "all";
  }
  return "partial";
}

export function defaultSelectedRunIds(runs: ManagedRun[], focusedRunId: string | null) {
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
