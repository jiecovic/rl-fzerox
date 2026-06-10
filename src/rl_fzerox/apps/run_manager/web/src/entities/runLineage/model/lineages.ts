// src/rl_fzerox/apps/run_manager/web/src/entities/runLineage/model/lineages.ts
import type {
  PendingDelete,
  RunLineageBucket,
  RunLineageGroup,
  RunLineageRun,
  RunSource,
} from "@/entities/runLineage/model/types";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";

export function disclosureDefaults(lineages: readonly RunLineageGroup[]) {
  const defaults: Record<string, boolean> = {};
  for (const lineage of lineages) {
    defaults[lineage.id] = true;
  }
  return defaults;
}

export function disclosureStateFor(lineages: readonly RunLineageGroup[], open: boolean) {
  const defaults: Record<string, boolean> = {};
  for (const lineage of lineages) {
    defaults[lineage.id] = open;
  }
  return defaults;
}

export function buildLineageGroups(
  runs: readonly ManagedRun[],
  drafts: readonly ManagedDraft[],
): RunLineageGroup[] {
  const dependentDraftCountByRunId = countDependentDrafts(drafts);
  const runsByLineage = new Map<string, ManagedRun[]>();
  for (const run of runs) {
    const lineageRuns = runsByLineage.get(run.lineage_id);
    if (lineageRuns === undefined) {
      runsByLineage.set(run.lineage_id, [run]);
    } else {
      lineageRuns.push(run);
    }
  }
  return [...runsByLineage.entries()]
    .map(([lineageId, lineageRuns]) =>
      buildLineageGroup(lineageId, lineageRuns, dependentDraftCountByRunId),
    )
    .sort((left, right) => compareIso(right.latestUpdatedAt, left.latestUpdatedAt));
}

export function buildLineageBuckets(lineages: readonly RunLineageGroup[]): RunLineageBucket[] {
  const bucketsById = new Map<string, RunLineageBucket>();
  for (const lineage of lineages) {
    const groupNames = lineage.groupNames.length === 0 ? [null] : lineage.groupNames;
    for (const groupName of groupNames) {
      const id = groupName === null ? "ungrouped" : slugifyGroupName(groupName);
      const existing = bucketsById.get(id);
      if (existing === undefined) {
        bucketsById.set(id, {
          groupName,
          id,
          label: groupName ?? "Ungrouped",
          latestUpdatedAt: lineage.latestUpdatedAt,
          lineages: [lineage],
          slug: id,
        });
      } else {
        if (compareIso(lineage.latestUpdatedAt, existing.latestUpdatedAt) > 0) {
          existing.latestUpdatedAt = lineage.latestUpdatedAt;
        }
        existing.lineages.push(lineage);
      }
    }
  }
  return [...bucketsById.values()].sort((left, right) => {
    const activityDelta = compareIso(right.latestUpdatedAt, left.latestUpdatedAt);
    if (activityDelta !== 0) {
      return activityDelta;
    }
    if (left.groupName === null && right.groupName !== null) {
      return 1;
    }
    if (left.groupName !== null && right.groupName === null) {
      return -1;
    }
    return left.label.localeCompare(right.label);
  });
}

export function progressLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "no samples";
  }
  return `${(runtime.progress_fraction * 100).toFixed(1)}% complete`;
}

export function stepLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "0 steps";
  }
  return `${runtime.num_timesteps.toLocaleString()} / ${runtime.total_timesteps.toLocaleString()} steps`;
}

export function runtimePrimaryLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "n/a";
  }
  const reward = runtime.episode_reward_mean;
  return reward === undefined || reward === null ? "n/a" : reward.toFixed(2);
}

export function runtimeSecondaryLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "n/a";
  }
  const fps = runtime.fps;
  return fps === undefined || fps === null ? "n/a" : `${fps.toFixed(0)} fps`;
}

export function statusLabel(run: ManagedRun) {
  return run.status;
}

export function deleteDisabledReason(entry: RunLineageRun, busy: boolean, isDeleting: boolean) {
  if (busy || isDeleting) {
    return "Wait for the current action to finish";
  }
  if (entry.run.status === "running") {
    return "Stop the run before deleting it";
  }
  if (entry.run.pending_command !== null) {
    return `${entry.run.pending_command} requested`;
  }
  if (entry.childCount > 0) {
    return "Only leaf runs can be deleted individually";
  }
  if (entry.dependentDraftCount > 0) {
    return "Delete or retarget fork drafts that still depend on this run";
  }
  return "Delete run";
}

export function deleteDescription(pendingDelete: PendingDelete | null) {
  if (pendingDelete === null) {
    return "";
  }
  if (pendingDelete.kind === "lineage") {
    return `Delete lineage "${pendingDelete.lineage.label}"? All runs, dependent fork drafts, and lineage files on disk will be removed.`;
  }
  return `Delete run "${pendingDelete.run.name}"? Its manager record and run directory will be removed from disk.`;
}

function buildLineageGroup(
  lineageId: string,
  lineageRuns: readonly ManagedRun[],
  dependentDraftCountByRunId: ReadonlyMap<string, number>,
): RunLineageGroup {
  const childCountByRunId = countChildRuns(lineageRuns);
  const depthByRunId = lineageDepthByRunId(lineageRuns);
  const runById = new Map(lineageRuns.map((run) => [run.id, run]));
  const orderedRuns = lineageOrder(lineageRuns);
  const rootRun =
    orderedRuns.find(
      (run) =>
        run.parent_run_id === null ||
        !lineageRuns.some((candidate) => candidate.id === run.parent_run_id),
    ) ?? orderedRuns[0];
  const label = rootRun?.name ?? lineageId;
  const createdAt = rootRun?.created_at ?? orderedRuns[0]?.created_at ?? "";
  const latestUpdatedAt = orderedRuns.reduce(
    (latest, run) =>
      compareIso(effectiveUpdatedAt(run), latest) > 0 ? effectiveUpdatedAt(run) : latest,
    createdAt,
  );
  const runs = orderedRuns.map((run) => ({
    childCount: childCountByRunId.get(run.id) ?? 0,
    dependentDraftCount: dependentDraftCountByRunId.get(run.id) ?? 0,
    depth: depthByRunId.get(run.id) ?? 0,
    isRoot: (depthByRunId.get(run.id) ?? 0) === 0,
    run,
    source: runSource(run, runById),
    stageLabel: runStageLabel(depthByRunId.get(run.id) ?? 0),
  }));
  const canDeleteLineage = runs.every(
    (entry) => entry.run.status !== "running" && entry.run.pending_command === null,
  );
  return {
    canDeleteLineage,
    createdAt,
    groupNames: rootRun?.lineage_groups ?? [],
    id: lineageId,
    label,
    latestUpdatedAt,
    runs,
  };
}

function slugifyGroupName(groupName: string) {
  return (
    groupName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "unnamed"
  );
}

function parentLineageRun(run: ManagedRun, runById: ReadonlyMap<string, ManagedRun>) {
  if (run.parent_run_id !== null) {
    const parent = runById.get(run.parent_run_id);
    if (parent !== undefined) {
      return parent;
    }
  }
  if (run.source_run_id !== null) {
    const sourceRun = runById.get(run.source_run_id);
    if (sourceRun !== undefined) {
      return sourceRun;
    }
  }
  return null;
}

function runSource(run: ManagedRun, runById: ReadonlyMap<string, ManagedRun>): RunSource {
  const parent = parentLineageRun(run, runById);
  if (parent === null) {
    return { kind: "root" };
  }
  return {
    kind: "fork",
    artifactLabel: run.source_artifact === null ? null : `${run.source_artifact} checkpoint`,
    parentName: parent.name,
    stepLabel:
      run.source_num_timesteps === null
        ? null
        : `${run.source_num_timesteps.toLocaleString()} steps`,
  };
}

function runStageLabel(depth: number) {
  return depth === 0 ? "root" : `fork ${depth}`;
}

function countChildRuns(runs: readonly ManagedRun[]) {
  const counts = new Map<string, number>();
  const lineageRunIds = new Set(runs.map((run) => run.id));
  for (const run of runs) {
    const parentId =
      run.parent_run_id !== null && lineageRunIds.has(run.parent_run_id)
        ? run.parent_run_id
        : run.source_run_id !== null && lineageRunIds.has(run.source_run_id)
          ? run.source_run_id
          : null;
    if (parentId !== null) {
      counts.set(parentId, (counts.get(parentId) ?? 0) + 1);
    }
  }
  return counts;
}

function countDependentDrafts(drafts: readonly ManagedDraft[]) {
  const counts = new Map<string, number>();
  for (const draft of drafts) {
    if (draft.source_run_id === null) {
      continue;
    }
    counts.set(draft.source_run_id, (counts.get(draft.source_run_id) ?? 0) + 1);
  }
  return counts;
}

function lineageDepthByRunId(lineageRuns: readonly ManagedRun[]) {
  const runById = new Map(lineageRuns.map((entry) => [entry.id, entry]));
  const depthByRunId = new Map<string, number>();

  function resolveDepth(runId: string): number {
    const existing = depthByRunId.get(runId);
    if (existing !== undefined) {
      return existing;
    }
    const run = runById.get(runId);
    if (run === undefined) {
      depthByRunId.set(runId, 0);
      return 0;
    }
    const parent = parentLineageRun(run, runById);
    if (parent === null) {
      depthByRunId.set(runId, 0);
      return 0;
    }
    const depth = resolveDepth(parent.id) + 1;
    depthByRunId.set(runId, depth);
    return depth;
  }

  for (const run of lineageRuns) {
    resolveDepth(run.id);
  }
  return depthByRunId;
}

function lineageOrder(lineageRuns: readonly ManagedRun[]) {
  const runById = new Map(lineageRuns.map((run) => [run.id, run]));
  const relationChildren = new Map<string, ManagedRun[]>();
  const roots: ManagedRun[] = [];

  for (const run of lineageRuns) {
    const parent = parentLineageRun(run, runById);
    if (parent === null) {
      roots.push(run);
      continue;
    }
    const children = relationChildren.get(parent.id);
    if (children === undefined) {
      relationChildren.set(parent.id, [run]);
    } else {
      children.push(run);
    }
  }

  const sortRuns = (left: ManagedRun, right: ManagedRun) => {
    const createdDelta = compareIso(left.created_at, right.created_at);
    if (createdDelta !== 0) {
      return createdDelta;
    }
    return left.name.localeCompare(right.name);
  };

  roots.sort(sortRuns);
  for (const children of relationChildren.values()) {
    children.sort(sortRuns);
  }

  const ordered: ManagedRun[] = [];
  const visited = new Set<string>();

  function visit(run: ManagedRun) {
    if (visited.has(run.id)) {
      return;
    }
    visited.add(run.id);
    ordered.push(run);
    for (const child of relationChildren.get(run.id) ?? []) {
      visit(child);
    }
  }

  for (const root of roots) {
    visit(root);
  }
  for (const run of lineageRuns) {
    visit(run);
  }
  return ordered;
}

function effectiveUpdatedAt(run: ManagedRun) {
  return run.runtime?.updated_at ?? run.stopped_at ?? run.started_at ?? run.created_at;
}

function compareIso(left: string, right: string) {
  return Date.parse(left) - Date.parse(right);
}
