// src/rl_fzerox/apps/run_manager/web/src/features/runs/RunsPanel.tsx
import { useMemo, useRef, useState } from "react";

import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { useRunsPanelActions } from "@/features/runs/panel/actions";
import { LineageCard } from "@/features/runs/panel/LineageCard";
import {
  buildLineageBuckets,
  buildLineageGroups,
  deleteDescription,
  disclosureDefaults,
  disclosureStateFor,
} from "@/features/runs/panel/model";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { ChevronIcon, ImportIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface RunsPanelProps {
  drafts: ManagedDraft[];
  onDeleteLineage: (lineageId: string) => Promise<void>;
  onDeleteRun: (run: ManagedRun) => Promise<void>;
  onExportRun: (run: ManagedRun) => Promise<void>;
  onImportRunBundle: (file: File) => Promise<void>;
  onOpenRun: (run: ManagedRun) => void;
  onResumeRun: (run: ManagedRun) => Promise<void>;
  onStopRun: (run: ManagedRun) => Promise<void>;
  onUpdateLineageGroups: (lineageId: string, groupNames: readonly string[]) => Promise<void>;
  runs: ManagedRun[];
}

export function RunsPanel({
  drafts,
  onDeleteLineage,
  onDeleteRun,
  onExportRun,
  onImportRunBundle,
  onOpenRun,
  onResumeRun,
  onStopRun,
  onUpdateLineageGroups,
  runs,
}: RunsPanelProps) {
  const importInputRef = useRef<HTMLInputElement>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const lineageGroups = useMemo(() => buildLineageGroups(runs, drafts), [drafts, runs]);
  const lineageBuckets = useMemo(() => buildLineageBuckets(lineageGroups), [lineageGroups]);
  const bucketDisclosureDefaults = useMemo<Record<string, boolean>>(
    () => Object.fromEntries(lineageBuckets.map((bucket) => [bucket.id, true] as const)),
    [lineageBuckets],
  );
  const lineageDisclosureDefaults = useMemo(
    () => disclosureDefaults(lineageGroups),
    [lineageGroups],
  );
  const [bucketOpen, setBucketOpen] = usePersistentDisclosureMap(
    "run-manager-lineage-group-open",
    bucketDisclosureDefaults,
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

  async function importSelectedBundle(file: File) {
    setImportError(null);
    setIsImporting(true);
    try {
      await onImportRunBundle(file);
    } catch (caught) {
      setImportError(caught instanceof Error ? caught.message : "run import failed");
    } finally {
      setIsImporting(false);
    }
  }

  return (
    <>
      <Panel>
        <div className="panel-header-row">
          <PanelHeader title="Runs" subtitle="Fork chains grouped by lineage." />
          <div className="run-panel-actions">
            <input
              ref={importInputRef}
              accept=".zip,application/zip"
              aria-label="Import run bundle file"
              style={{ display: "none" }}
              type="file"
              onChange={(event) => {
                const file = event.currentTarget.files?.[0] ?? null;
                event.currentTarget.value = "";
                if (file !== null) {
                  void importSelectedBundle(file);
                }
              }}
            />
            <button
              className="secondary-button button-with-icon"
              disabled={isImporting}
              type="button"
              onClick={() => importInputRef.current?.click()}
            >
              <ImportIcon />
              {isImporting ? "Importing" : "Import"}
            </button>
            {runs.length > 0 ? (
              <DisclosureToolbar
                collapseLabel="Collapse all lineages"
                expandLabel="Expand all lineages"
                onCollapseAll={() => setLineageOpen(disclosureStateFor(lineageGroups, false))}
                onExpandAll={() => setLineageOpen(disclosureStateFor(lineageGroups, true))}
              />
            ) : null}
          </div>
        </div>
        {actionError !== null ? <Notice tone="error">{actionError}</Notice> : null}
        {importError !== null ? <Notice tone="error">{importError}</Notice> : null}
        {runs.length === 0 ? (
          <Notice>No launched runs yet.</Notice>
        ) : (
          <div className="run-lineage-bucket-list">
            {lineageBuckets.map((bucket) => {
              const isBucketOpen = bucketOpen[bucket.id] ?? true;
              return (
                <section
                  aria-label={`${bucket.label} lineage group`}
                  className="run-lineage-bucket"
                  key={bucket.id}
                >
                  <button
                    aria-expanded={isBucketOpen}
                    aria-label={`${isBucketOpen ? "Collapse" : "Expand"} group ${bucket.label}`}
                    className="run-lineage-bucket-header"
                    type="button"
                    onClick={() =>
                      setBucketOpen((current) => ({
                        ...current,
                        [bucket.id]: !(current[bucket.id] ?? true),
                      }))
                    }
                  >
                    <span
                      aria-hidden="true"
                      className={
                        isBucketOpen ? "run-lineage-chevron is-open" : "run-lineage-chevron"
                      }
                    >
                      <ChevronIcon />
                    </span>
                    <strong>{bucket.label}</strong>
                    <span className="run-lineage-bucket-meta">
                      {bucket.lineages.length} lineages · tensorboard: local/tensorboard_views/
                      {bucket.slug}
                    </span>
                  </button>
                  {isBucketOpen ? (
                    <div className="run-lineage-list">
                      {bucket.lineages.map((lineage) => (
                        <LineageCard
                          busyActionRunId={busyActionRunId}
                          isDeleting={isDeleting}
                          key={lineage.id}
                          lineage={lineage}
                          onDeleteLineage={() => requestLineageDelete(lineage)}
                          onExportRun={(run) => onExportRun(run)}
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
                          onUpdateGroups={onUpdateLineageGroups}
                          open={lineageOpen[lineage.id] ?? true}
                        />
                      ))}
                    </div>
                  ) : null}
                </section>
              );
            })}
          </div>
        )}
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
