// web/run-manager/src/pages/runs/RunsPanel.tsx
import { useMemo, useRef, useState } from "react";
import { isPinnedRun } from "@/entities/run/model/runtime";
import {
  buildLineageBuckets,
  buildLineageGroups,
  deleteDescription,
  disclosureDefaults,
  disclosureStateFor,
} from "@/entities/runLineage/model/lineages";
import type { RunLineageRun } from "@/entities/runLineage/model/types";
import { LineageCard } from "@/entities/runLineage/ui/LineageCard";
import { runLineageMainGridClass, runLineageOuterGridClass } from "@/entities/runLineage/ui/layout";
import { RunRow } from "@/entities/runLineage/ui/RunRow";
import { useRunsPanelActions } from "@/features/runsPanelActions/useRunsPanelActions";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/shared/ui/config/disclosureState";
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
  const activeRunEntries = useMemo(
    () =>
      lineageGroups.flatMap((lineage) => lineage.runs.filter((entry) => isPinnedRun(entry.run))),
    [lineageGroups],
  );

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
          <div className="flex items-center gap-2">
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
            <Button
              className="gap-2"
              disabled={isImporting}
              type="button"
              onClick={() => importInputRef.current?.click()}
            >
              <ImportIcon />
              {isImporting ? "Importing" : "Import"}
            </Button>
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
          <div className="grid gap-3.5">
            {activeRunEntries.length > 0 ? (
              <ActiveRunsSection
                busyActionRunId={busyActionRunId}
                entries={activeRunEntries}
                isDeleting={isDeleting}
                onExportRun={onExportRun}
                onOpenRun={onOpenRun}
                onRequestRunDelete={requestRunDelete}
                onResumeRun={onResumeRun}
                onRunAction={runAction}
                onStopRun={onStopRun}
              />
            ) : null}
            {lineageBuckets.map((bucket) => {
              const isBucketOpen = bucketOpen[bucket.id] ?? true;
              return (
                <section
                  aria-label={`${bucket.label} lineage group`}
                  className="grid gap-2"
                  key={bucket.id}
                >
                  <button
                    aria-expanded={isBucketOpen}
                    aria-label={`${isBucketOpen ? "Collapse" : "Expand"} group ${bucket.label}`}
                    className="grid w-full grid-cols-[auto_auto_minmax(0,1fr)] items-center gap-2 border-0 bg-transparent px-0.5 text-left text-app-text"
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
                    <span className="min-w-0 justify-self-end overflow-hidden text-ellipsis whitespace-nowrap text-[11px] tabular-nums text-app-muted">
                      {bucket.lineages.length} lineages · tensorboard: local/tensorboard_views/
                      {bucket.slug}
                    </span>
                  </button>
                  {isBucketOpen ? (
                    <div className="grid gap-2.5">
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

function ActiveRunsSection({
  busyActionRunId,
  entries,
  isDeleting,
  onExportRun,
  onOpenRun,
  onRequestRunDelete,
  onResumeRun,
  onRunAction,
  onStopRun,
}: {
  busyActionRunId: string | null;
  entries: readonly RunLineageRun[];
  isDeleting: boolean;
  onExportRun: (run: ManagedRun) => Promise<void>;
  onOpenRun: (run: ManagedRun) => void;
  onRequestRunDelete: (run: ManagedRun) => void;
  onResumeRun: (run: ManagedRun) => Promise<void>;
  onRunAction: (runId: string, callback: () => Promise<void>) => Promise<void>;
  onStopRun: (run: ManagedRun) => Promise<void>;
}) {
  return (
    <section
      aria-label="Active runs"
      className="rounded-lg border border-app-accent bg-app-surface"
    >
      <div className="flex min-h-11 items-center justify-between gap-3 px-3">
        <strong>Active runs</strong>
        <span className="text-[11px] tabular-nums text-app-muted">
          {entries.length.toLocaleString()} running
        </span>
      </div>
      <div
        className={`${runLineageOuterGridClass} hidden border-t border-app-border px-3 py-1 text-[11px] font-bold tracking-[0.04em] text-app-muted uppercase min-[761px]:grid`}
        role="presentation"
      >
        <div className={runLineageMainGridClass}>
          <span>Run</span>
          <span>Progress</span>
          <span>FPS</span>
          <span>Status</span>
          <span>Created at</span>
        </div>
        <span className="w-[136px] justify-self-end text-right">Actions</span>
      </div>
      <div className="grid border-t border-app-border">
        {entries.map((entry) => (
          <RunRow
            busyActionRunId={busyActionRunId}
            entry={entry}
            isDeleting={isDeleting}
            key={entry.run.id}
            onExportRun={() => onExportRun(entry.run)}
            onOpenRun={() => onOpenRun(entry.run)}
            onRequestDelete={() => onRequestRunDelete(entry.run)}
            onResumeRun={() => onResumeRun(entry.run)}
            onRunAction={onRunAction}
            onStopRun={() => onStopRun(entry.run)}
          />
        ))}
      </div>
    </section>
  );
}
