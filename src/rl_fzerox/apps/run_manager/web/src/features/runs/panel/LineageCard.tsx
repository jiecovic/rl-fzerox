// src/rl_fzerox/apps/run_manager/web/src/features/runs/panel/LineageCard.tsx
import { useEffect, useState } from "react";
import { RunRow } from "@/features/runs/panel/RunRow";
import type { RunLineageGroup } from "@/features/runs/panel/types";
import type { ManagedRun } from "@/shared/api/contract";
import { formatDate, formatRelativeTime } from "@/shared/ui/format";
import { ChevronIcon, TrashIcon } from "@/shared/ui/icons";

interface LineageCardProps {
  busyActionRunId: string | null;
  isDeleting: boolean;
  lineage: RunLineageGroup;
  onDeleteLineage: () => void;
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
    <section className="run-lineage-card">
      <div className="run-lineage-summary">
        <button
          aria-expanded={open}
          aria-label={`${open ? "Collapse" : "Expand"} lineage ${lineage.label}`}
          className="run-lineage-toggle"
          type="button"
          onClick={onToggle}
        >
          <span
            aria-hidden="true"
            className={open ? "run-lineage-chevron is-open" : "run-lineage-chevron"}
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
          <form
            className="run-lineage-group-form"
            onSubmit={(event) => {
              event.preventDefault();
              void saveGroup();
            }}
          >
            <input
              aria-label={`Groups for lineage ${lineage.label}`}
              placeholder="Ungrouped"
              type="text"
              value={groupInput}
              onChange={(event) => setGroupInput(event.target.value)}
            />
            <button
              className="secondary-button compact-lineage-group-button"
              disabled={!groupChanged || savingGroup}
              type="submit"
            >
              {savingGroup ? "Saving" : "Save"}
            </button>
          </form>
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
            onClick={onDeleteLineage}
          >
            <TrashIcon />
          </button>
        </div>
      </div>
      {groupError !== null ? <div className="run-lineage-group-error">{groupError}</div> : null}
      {open ? (
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
            {lineage.runs.map((entry) => (
              <RunRow
                busyActionRunId={busyActionRunId}
                entry={entry}
                isDeleting={isDeleting}
                key={entry.run.id}
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
