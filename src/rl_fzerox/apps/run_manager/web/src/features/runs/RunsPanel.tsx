// src/rl_fzerox/apps/run_manager/web/src/features/runs/RunsPanel.tsx
import { useMemo } from "react";

import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { useRunsPanelActions } from "@/features/runs/panel/actions";
import { LineageCard } from "@/features/runs/panel/LineageCard";
import {
  buildLineageGroups,
  deleteDescription,
  disclosureDefaults,
  disclosureStateFor,
} from "@/features/runs/panel/model";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
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

export function RunsPanel({
  drafts,
  onDeleteLineage,
  onDeleteRun,
  onOpenRun,
  onResumeRun,
  onStopRun,
  runs,
}: RunsPanelProps) {
  const lineageGroups = useMemo(() => buildLineageGroups(runs, drafts), [drafts, runs]);
  const lineageDisclosureDefaults = useMemo(
    () => disclosureDefaults(lineageGroups),
    [lineageGroups],
  );
  const [lineageOpen, setLineageOpen] = usePersistentDisclosureMap(
    "run-manager-lineage-open",
    lineageDisclosureDefaults,
  );
  const {
    actionError,
    busyActionRunId,
    closePendingDelete,
    confirmDelete,
    isDeleting,
    pendingDelete,
    requestLineageDelete,
    requestRunDelete,
    runAction,
  } = useRunsPanelActions({
    onDeleteLineage,
    onDeleteRun,
  });

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
            <LineageCard
              busyActionRunId={busyActionRunId}
              isDeleting={isDeleting}
              key={lineage.id}
              lineage={lineage}
              onDeleteLineage={() => requestLineageDelete(lineage)}
              onOpenRun={onOpenRun}
              onRequestRunDelete={requestRunDelete}
              onResumeRun={(run) => onResumeRun(run)}
              onRunAction={runAction}
              onStopRun={(run) => onStopRun(run)}
              onToggle={() =>
                setLineageOpen((current) => ({
                  ...current,
                  [lineage.id]: !(current[lineage.id] ?? true),
                }))
              }
              open={lineageOpen[lineage.id] ?? true}
            />
          ))}
        </div>
      </Panel>
      <ConfirmDialog
        busy={isDeleting}
        confirmLabel={pendingDelete?.kind === "lineage" ? "Delete lineage" : "Delete run"}
        description={deleteDescription(pendingDelete)}
        open={pendingDelete !== null}
        title={pendingDelete?.kind === "lineage" ? "Delete lineage" : "Delete run"}
        onClose={closePendingDelete}
        onConfirm={() => void confirmDelete()}
      />
    </>
  );
}
