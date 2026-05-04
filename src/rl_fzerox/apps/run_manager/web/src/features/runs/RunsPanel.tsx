import { useMemo, useState } from "react";

import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { RunActivityIndicator } from "@/features/runs/RunActivityIndicator";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { formatDate, formatRelativeTime } from "@/shared/ui/format";
import { BranchSourceIcon, ChevronIcon, ResumeIcon, StopIcon, TrashIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface RunsPanelProps {
  drafts: ManagedDraft[];
  onDeleteLineage: (lineageId: string) => Promise<void>;
  onDeleteRun: (run: ManagedRun) => Promise<void>;
  onOpenRun: (run: ManagedRun) => void;
  onResumeRun: (run: ManagedRun) => Promise<void>;
  onStopRun: (run: ManagedRun) => Promise<void>;
  runs: ManagedRun[];
}

type PendingDelete =
  | { kind: "lineage"; lineage: RunLineageGroup }
  | { kind: "run"; run: ManagedRun };

type RunLineageGroup = {
  canDeleteLineage: boolean;
  createdAt: string;
  id: string;
  label: string;
  latestUpdatedAt: string;
  runs: RunLineageRun[];
};

type RunLineageRun = {
  childCount: number;
  dependentDraftCount: number;
  depth: number;
  isRoot: boolean;
  source: RunSource;
  run: ManagedRun;
  stageLabel: string;
};

type RunSource =
  | { kind: "root" }
  | {
      kind: "fork";
      artifactLabel: string | null;
      parentName: string;
      stepLabel: string | null;
    };

export function RunsPanel({
  drafts,
  onDeleteLineage,
  onDeleteRun,
  onOpenRun,
  onResumeRun,
  onStopRun,
  runs,
}: RunsPanelProps) {
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [busyActionRunId, setBusyActionRunId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const lineageGroups = useMemo(() => buildLineageGroups(runs, drafts), [drafts, runs]);
  const lineageDisclosureDefaults = useMemo(
    () => disclosureDefaults(lineageGroups),
    [lineageGroups],
  );
  const [lineageOpen, setLineageOpen] = usePersistentDisclosureMap(
    "run-manager-lineage-open",
    lineageDisclosureDefaults,
  );

  if (runs.length === 0) {
    return <Notice>No launched runs yet.</Notice>;
  }

  return (
    <>
      <Panel>
        <div className="panel-header-row">
          <PanelHeader title="Runs" subtitle="Fork chains grouped by lineage." />
          <DisclosureToolbar
            collapseLabel="Collapse all lineages"
            expandLabel="Expand all lineages"
            onCollapseAll={() => setLineageOpen(disclosureStateFor(lineageGroups, false))}
            onExpandAll={() => setLineageOpen(disclosureStateFor(lineageGroups, true))}
          />
        </div>
        {actionError !== null ? <Notice tone="error">{actionError}</Notice> : null}
        <div className="run-lineage-list">
          {lineageGroups.map((lineage) => (
            <section className="run-lineage-card" key={lineage.id}>
              <div className="run-lineage-summary">
                <button
                  aria-expanded={lineageOpen[lineage.id] ?? true}
                  aria-label={`${(lineageOpen[lineage.id] ?? true) ? "Collapse" : "Expand"} lineage ${lineage.label}`}
                  className="run-lineage-toggle"
                  type="button"
                  onClick={() =>
                    setLineageOpen((current) => ({
                      ...current,
                      [lineage.id]: !(current[lineage.id] ?? true),
                    }))
                  }
                >
                  <span
                    aria-hidden="true"
                    className={
                      (lineageOpen[lineage.id] ?? true)
                        ? "run-lineage-chevron is-open"
                        : "run-lineage-chevron"
                    }
                  >
                    <ChevronIcon />
                  </span>
                  <span className="run-lineage-copy">
                    <strong>{lineage.label}</strong>
                    <span className="run-record-subtle">
                      {lineage.runs.length} runs · created {formatDate(lineage.createdAt)} · updated{" "}
                      {formatRelativeTime(lineage.latestUpdatedAt)}
                    </span>
                  </span>
                </button>
                <div className="run-lineage-actions">
                  <button
                    aria-label={`Delete lineage ${lineage.label}`}
                    className="icon-button compact-icon-button danger"
                    title={
                      lineage.canDeleteLineage
                        ? "Delete lineage"
                        : "Stop all runs and clear pending commands before deleting lineage"
                    }
                    type="button"
                    disabled={!lineage.canDeleteLineage || isDeleting || busyActionRunId !== null}
                    onClick={() => {
                      setActionError(null);
                      setPendingDelete({ kind: "lineage", lineage });
                    }}
                  >
                    <TrashIcon />
                  </button>
                </div>
              </div>
              {(lineageOpen[lineage.id] ?? true) ? (
                <div className="run-lineage-body">
                  <div className="run-list-head run-lineage-head" role="presentation">
                    <div className="run-list-head-main">
                      <span>Run</span>
                      <span>Progress</span>
                      <span>Reward · FPS</span>
                      <span>Status</span>
                      <span>Created at</span>
                    </div>
                    <span className="run-list-head-actions">Actions</span>
                  </div>
                  <div className="run-lineage-runs">
                    {lineage.runs.map((entry) => {
                      const { run } = entry;
                      const pendingCommand = run.pending_command;
                      const busy = busyActionRunId === run.id;
                      const canStop =
                        run.status === "running" && pendingCommand === null && !busy && !isDeleting;
                      const canResume =
                        (run.status === "paused" ||
                          run.status === "stopped" ||
                          run.status === "failed") &&
                        !busy &&
                        !isDeleting;
                      const canDelete =
                        entry.childCount === 0 &&
                        entry.dependentDraftCount === 0 &&
                        run.status !== "running" &&
                        pendingCommand === null &&
                        !busy &&
                        !isDeleting;
                      const deleteReason = deleteDisabledReason(entry, busy, isDeleting);

                      return (
                        <div className="run-list-row run-lineage-row" key={run.id}>
                          <button
                            aria-label={`Open run ${run.name}`}
                            className="run-list-main"
                            type="button"
                            onClick={() => onOpenRun(run)}
                          >
                            <span className="run-list-cell run-list-cell-name">
                              <span className="run-branch-name">
                                <span
                                  aria-hidden="true"
                                  className={
                                    entry.isRoot
                                      ? "run-branch-graph is-root"
                                      : entry.childCount === 0
                                        ? "run-branch-graph is-leaf"
                                        : "run-branch-graph"
                                  }
                                >
                                  <span className="run-branch-line run-branch-line-top" />
                                  <span className="run-branch-node">
                                    {entry.isRoot ? "R" : entry.depth}
                                  </span>
                                  <span className="run-branch-line run-branch-line-bottom" />
                                </span>
                                <span className="run-branch-copy">
                                  <span className="run-branch-header">
                                    <span className="record-name">{run.name}</span>
                                    <span
                                      className={
                                        entry.isRoot
                                          ? "run-branch-stage run-branch-stage-root"
                                          : "run-branch-stage"
                                      }
                                    >
                                      {entry.stageLabel}
                                    </span>
                                  </span>
                                  {entry.source.kind === "root" ? (
                                    <span className="run-branch-source-row run-branch-source-root">
                                      <span className="run-branch-source-kicker">origin</span>
                                      <span className="run-branch-source-text">lineage root</span>
                                    </span>
                                  ) : (
                                    <span className="run-branch-source-row">
                                      <span aria-hidden="true" className="run-branch-source-icon">
                                        <BranchSourceIcon />
                                      </span>
                                      <span className="run-branch-source-kicker">forked from</span>
                                      <span className="run-branch-source-text">
                                        {entry.source.parentName}
                                      </span>
                                      {entry.source.artifactLabel !== null ? (
                                        <span className="run-branch-source-chip">
                                          {entry.source.artifactLabel}
                                        </span>
                                      ) : null}
                                      {entry.source.stepLabel !== null ? (
                                        <span className="run-branch-source-chip">
                                          {entry.source.stepLabel}
                                        </span>
                                      ) : null}
                                    </span>
                                  )}
                                </span>
                              </span>
                            </span>
                            <span className="run-list-cell run-list-progress">
                              <div className="run-list-progress-top">
                                <span className="run-record-progress-label">
                                  {progressLabel(run)}
                                </span>
                              </div>
                              <span aria-hidden="true" className="run-record-progress-track">
                                <span
                                  className="run-record-progress-fill"
                                  style={{
                                    width: `${(run.runtime?.progress_fraction ?? 0) * 100}%`,
                                  }}
                                />
                              </span>
                              <span className="run-record-subtle">{stepLabel(run)}</span>
                            </span>
                            <span className="run-list-cell run-list-live">
                              <span>{runtimePrimaryLabel(run)}</span>
                              <span className="run-record-subtle">
                                {runtimeSecondaryLabel(run)}
                              </span>
                            </span>
                            <span className="run-list-cell run-list-status">
                              <span className="run-status-chip">{statusLabel(run)}</span>
                              <span className="run-record-subtle">
                                <RunActivityIndicator run={run} />
                              </span>
                            </span>
                            <span className="run-list-cell run-list-created">
                              <span className="run-record-subtle run-created-at">
                                {formatDate(run.created_at)}
                              </span>
                            </span>
                          </button>
                          <div className="run-list-actions">
                            <button
                              aria-label={`Stop run ${run.name}`}
                              className="icon-button compact-icon-button run-list-action-button"
                              title={
                                pendingCommand !== null ? `${pendingCommand} requested` : "Stop run"
                              }
                              type="button"
                              disabled={!canStop}
                              onClick={() => void runAction(run.id, () => onStopRun(run))}
                            >
                              <StopIcon />
                            </button>
                            <button
                              aria-label={`Resume run ${run.name}`}
                              className="icon-button compact-icon-button run-list-action-button"
                              title="Resume run"
                              type="button"
                              disabled={!canResume}
                              onClick={() => void runAction(run.id, () => onResumeRun(run))}
                            >
                              <ResumeIcon />
                            </button>
                            <button
                              aria-label={`Delete run ${run.name}`}
                              className="icon-button compact-icon-button danger run-list-action-button"
                              title={deleteReason}
                              type="button"
                              disabled={!canDelete}
                              onClick={() => {
                                setActionError(null);
                                setPendingDelete({ kind: "run", run });
                              }}
                            >
                              <TrashIcon />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </section>
          ))}
        </div>
      </Panel>
      <ConfirmDialog
        busy={isDeleting}
        confirmLabel={pendingDelete?.kind === "lineage" ? "Delete lineage" : "Delete run"}
        description={deleteDescription(pendingDelete)}
        open={pendingDelete !== null}
        title={pendingDelete?.kind === "lineage" ? "Delete lineage" : "Delete run"}
        onClose={() => {
          if (!isDeleting) {
            setPendingDelete(null);
          }
        }}
        onConfirm={() => void confirmDelete()}
      />
    </>
  );

  async function runAction(runId: string, callback: () => Promise<void>) {
    setActionError(null);
    setBusyActionRunId(runId);
    try {
      await callback();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "run action failed");
    } finally {
      setBusyActionRunId((current) => (current === runId ? null : current));
    }
  }

  async function confirmDelete() {
    if (pendingDelete === null) {
      return;
    }
    setActionError(null);
    setIsDeleting(true);
    try {
      if (pendingDelete.kind === "lineage") {
        await onDeleteLineage(pendingDelete.lineage.id);
      } else {
        await onDeleteRun(pendingDelete.run);
      }
      setPendingDelete(null);
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "delete failed");
    } finally {
      setIsDeleting(false);
    }
  }
}

function disclosureDefaults(lineages: readonly RunLineageGroup[]) {
  const defaults: Record<string, boolean> = {};
  for (const lineage of lineages) {
    defaults[lineage.id] = true;
  }
  return defaults;
}

function disclosureStateFor(lineages: readonly RunLineageGroup[], open: boolean) {
  const defaults: Record<string, boolean> = {};
  for (const lineage of lineages) {
    defaults[lineage.id] = open;
  }
  return defaults;
}

function buildLineageGroups(
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
    id: lineageId,
    label,
    latestUpdatedAt,
    runs,
  };
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

function progressLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "no samples";
  }
  return `${(runtime.progress_fraction * 100).toFixed(1)}% complete`;
}

function stepLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "0 steps";
  }
  return `${runtime.num_timesteps.toLocaleString()} / ${runtime.total_timesteps.toLocaleString()} steps`;
}

function runtimePrimaryLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "n/a";
  }
  const reward = runtime.episode_reward_mean;
  return reward === undefined || reward === null ? "n/a" : reward.toFixed(2);
}

function runtimeSecondaryLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "n/a";
  }
  const fps = runtime.fps;
  return fps === undefined || fps === null ? "n/a" : `${fps.toFixed(0)} fps`;
}

function statusLabel(run: ManagedRun) {
  return run.status;
}

function deleteDisabledReason(entry: RunLineageRun, busy: boolean, isDeleting: boolean) {
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

function deleteDescription(pendingDelete: PendingDelete | null) {
  if (pendingDelete === null) {
    return "";
  }
  if (pendingDelete.kind === "lineage") {
    return `Delete lineage "${pendingDelete.lineage.label}"? All runs, dependent fork drafts, and lineage files on disk will be removed.`;
  }
  return `Delete run "${pendingDelete.run.name}"? Its manager record and run directory will be removed from disk.`;
}
