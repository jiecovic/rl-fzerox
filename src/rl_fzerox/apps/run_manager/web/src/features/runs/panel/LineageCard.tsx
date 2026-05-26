// src/rl_fzerox/apps/run_manager/web/src/features/runs/panel/LineageCard.tsx
import { useEffect, useState } from "react";
import { runLineageMainGridClass, runLineageOuterGridClass } from "@/features/runs/panel/layout";
import { RunRow } from "@/features/runs/panel/RunRow";
import type { RunLineageGroup } from "@/features/runs/panel/types";
import type { ManagedRun } from "@/shared/api/contract";
import { Button, IconButton } from "@/shared/ui/Button";
import { FieldInput } from "@/shared/ui/Field";
import { formatDate, formatRelativeTime } from "@/shared/ui/format";
import { ChevronIcon, TrashIcon } from "@/shared/ui/icons";

interface LineageCardProps {
  busyActionRunId: string | null;
  isDeleting: boolean;
  lineage: RunLineageGroup;
  onDeleteLineage: () => void;
  onExportRun: (run: ManagedRun) => Promise<void>;
  onOpenRun: (run: ManagedRun) => void;
  onRequestRunDelete: (run: ManagedRun) => void;
  onResumeRun: (run: ManagedRun) => Promise<void>;
  onRunAction: (runId: string, callback: () => Promise<void>) => Promise<void>;
  onStopRun: (run: ManagedRun) => Promise<void>;
  onToggle: () => void;
  onUpdateGroups: (lineageId: string, groupNames: readonly string[]) => Promise<void>;
  open: boolean;
}

export function LineageCard({
  busyActionRunId,
  isDeleting,
  lineage,
  onDeleteLineage,
  onExportRun,
  onOpenRun,
  onRequestRunDelete,
  onResumeRun,
  onRunAction,
  onStopRun,
  onToggle,
  onUpdateGroups,
  open,
}: LineageCardProps) {
  const persistedGroupInput = formatGroupNames(lineage.groupNames);
  const persistedGroupKey = groupNameKey(lineage.groupNames);
  const [groupInput, setGroupInput] = useState(persistedGroupInput);
  const [syncedGroupKey, setSyncedGroupKey] = useState(persistedGroupKey);
  const [savingGroup, setSavingGroup] = useState(false);
  const [groupError, setGroupError] = useState<string | null>(null);
  const inputGroupNames = parseLineageGroupInput(groupInput);
  const groupChanged = groupNameKey(inputGroupNames) !== persistedGroupKey;

  useEffect(() => {
    if (persistedGroupKey === syncedGroupKey) {
      return;
    }
    setGroupInput(persistedGroupInput);
    setGroupError(null);
    setSyncedGroupKey(persistedGroupKey);
  }, [persistedGroupInput, persistedGroupKey, syncedGroupKey]);

  async function saveGroup() {
    if (!groupChanged || savingGroup) {
      return;
    }
    setSavingGroup(true);
    setGroupError(null);
    try {
      await onUpdateGroups(lineage.id, inputGroupNames);
    } catch (caught) {
      setGroupError(caught instanceof Error ? caught.message : "group update failed");
    } finally {
      setSavingGroup(false);
    }
  }

  return (
    <section className="rounded-lg border border-app-border bg-app-surface">
      <div className="grid min-h-[52px] grid-cols-[minmax(0,1fr)_auto] items-center gap-2 px-3">
        <button
          aria-expanded={open}
          aria-label={`${open ? "Collapse" : "Expand"} lineage ${lineage.label}`}
          className="grid min-h-[52px] min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-2.5 border-0 bg-transparent p-0 text-left text-app-text hover:bg-app-surface-muted"
          type="button"
          onClick={onToggle}
        >
          <span
            aria-hidden="true"
            className={open ? "run-lineage-chevron is-open" : "run-lineage-chevron"}
          >
            <ChevronIcon />
          </span>
          <span className="grid min-w-0 gap-1">
            <strong>{lineage.label}</strong>
            <span className="text-[11px] tabular-nums text-app-muted">
              {lineage.runs.length} runs · created {formatDate(lineage.createdAt)} · updated{" "}
              {formatRelativeTime(lineage.latestUpdatedAt)}
            </span>
          </span>
        </button>
        <div className="inline-flex items-center gap-1.5">
          <form
            className="inline-flex items-center gap-1.5"
            onSubmit={(event) => {
              event.preventDefault();
              void saveGroup();
            }}
          >
            <FieldInput
              aria-label={`Groups for lineage ${lineage.label}`}
              className="h-8 w-[220px] min-w-0 rounded-md bg-app-surface-muted px-2 text-xs"
              placeholder="Ungrouped"
              type="text"
              value={groupInput}
              onChange={(event) => setGroupInput(event.target.value)}
            />
            <Button
              className="h-8 px-2 text-xs"
              disabled={!groupChanged || savingGroup}
              type="submit"
            >
              {savingGroup ? "Saving" : "Save"}
            </Button>
          </form>
          <IconButton
            aria-label={`Delete lineage ${lineage.label}`}
            title={
              lineage.canDeleteLineage
                ? "Delete lineage"
                : "Stop all runs and clear pending commands before deleting lineage"
            }
            disabled={!lineage.canDeleteLineage || isDeleting || busyActionRunId !== null}
            size="compact"
            tone="danger"
            onClick={onDeleteLineage}
          >
            <TrashIcon />
          </IconButton>
        </div>
      </div>
      {groupError !== null ? (
        <div className="border-t border-app-border px-3 py-2 text-xs text-app-danger">
          {groupError}
        </div>
      ) : null}
      {open ? (
        <div className="border-t border-app-border pt-2 pb-1">
          <div
            className={`${runLineageOuterGridClass} hidden px-3 pb-1 text-[11px] font-bold tracking-[0.04em] text-app-muted uppercase min-[761px]:grid`}
            role="presentation"
          >
            <div className={runLineageMainGridClass}>
              <span>Run</span>
              <span>Progress</span>
              <span>Reward · FPS</span>
              <span>Status</span>
              <span>Created at</span>
            </div>
            <span className="w-[136px] justify-self-end text-right">Actions</span>
          </div>
          <div className="grid border-t border-app-border">
            {lineage.runs.map((entry) => (
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
        </div>
      ) : null}
    </section>
  );
}

function parseLineageGroupInput(value: string) {
  return [
    ...new Set(
      value
        .split(",")
        .map((part) => part.trim().replace(/\s+/g, " "))
        .filter((part) => part.length > 0),
    ),
  ].sort((left, right) => left.localeCompare(right));
}

function formatGroupNames(groupNames: readonly string[]) {
  return groupNames.join(", ");
}

function groupNameKey(groupNames: readonly string[]) {
  return [...groupNames].sort((left, right) => left.localeCompare(right)).join("\n");
}
