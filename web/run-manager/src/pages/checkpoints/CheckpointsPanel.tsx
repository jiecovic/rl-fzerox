// web/run-manager/src/pages/checkpoints/CheckpointsPanel.tsx
import { useEffect, useMemo, useState } from "react";

import type {
  CheckpointCatalogEntry,
  CheckpointCatalogResponse,
  PublishedCheckpoint,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { formatInteger } from "@/shared/ui/format";
import { ExportIcon, TrashIcon } from "@/shared/ui/icons";
import {
  ListActionsCell,
  ListActionsHeaderCell,
  ListRow,
  ListSelectAllHeaderCell,
  ListSelectionCell,
  ListTable,
  ListTableHead,
} from "@/shared/ui/ListTable";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface CheckpointsPanelProps {
  catalog: CheckpointCatalogResponse | null;
  error: string | null;
  onDeleteCheckpoint: (checkpoint: PublishedCheckpoint) => Promise<void>;
  onGlobalError: (message: string | null) => void;
  onInstallCheckpoint: (checkpointId: string, version: string) => Promise<void>;
  onOpenCheckpoint: (checkpoint: PublishedCheckpoint) => void;
}

export function CheckpointsPanel({
  catalog,
  error,
  onDeleteCheckpoint,
  onGlobalError,
  onInstallCheckpoint,
  onOpenCheckpoint,
}: CheckpointsPanelProps) {
  const [deleteRequested, setDeleteRequested] = useState<PublishedCheckpoint | null>(null);
  const [deleteSelectedRequested, setDeleteSelectedRequested] = useState(false);
  const [deletingCheckpointId, setDeletingCheckpointId] = useState<string | null>(null);
  const [installRequested, setInstallRequested] = useState<CheckpointCatalogEntry | null>(null);
  const [installingKey, setInstallingKey] = useState<string | null>(null);
  const [isDeletingSelected, setIsDeletingSelected] = useState(false);
  const [selectedCheckpointIds, setSelectedCheckpointIds] = useState<ReadonlySet<string>>(
    new Set(),
  );

  const installedCheckpoints = useMemo(
    () => catalog?.installed_checkpoints ?? [],
    [catalog?.installed_checkpoints],
  );
  const installedById = useMemo(
    () => new Map(installedCheckpoints.map((checkpoint) => [checkpoint.id, checkpoint])),
    [installedCheckpoints],
  );
  const installedCheckpointIds = useMemo(
    () => new Set(installedCheckpoints.map((checkpoint) => checkpoint.id)),
    [installedCheckpoints],
  );
  const selectedCheckpoints = useMemo(
    () => installedCheckpoints.filter((checkpoint) => selectedCheckpointIds.has(checkpoint.id)),
    [installedCheckpoints, selectedCheckpointIds],
  );
  const selectedCheckpointCount = selectedCheckpointIds.size;
  const allInstalledCheckpointsSelected =
    installedCheckpoints.length > 0 &&
    installedCheckpoints.every((checkpoint) => selectedCheckpointIds.has(checkpoint.id));
  const isDeletingCheckpoint = deletingCheckpointId !== null || isDeletingSelected;

  useEffect(() => {
    setSelectedCheckpointIds((current) => {
      const next = new Set(
        [...current].filter((checkpointId) => installedCheckpointIds.has(checkpointId)),
      );
      return next.size === current.size ? current : next;
    });
  }, [installedCheckpointIds]);

  async function installEntry(entry: CheckpointCatalogEntry): Promise<boolean> {
    const key = checkpointEntryKey(entry);
    setInstallingKey(key);
    onGlobalError(null);
    try {
      await onInstallCheckpoint(entry.id, entry.version);
      return true;
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "checkpoint install failed");
      return false;
    } finally {
      setInstallingKey(null);
    }
  }

  async function deleteCheckpoint(checkpoint: PublishedCheckpoint): Promise<boolean> {
    setDeletingCheckpointId(checkpoint.id);
    onGlobalError(null);
    try {
      await onDeleteCheckpoint(checkpoint);
      return true;
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "checkpoint delete failed");
      return false;
    } finally {
      setDeletingCheckpointId(null);
    }
  }

  async function confirmSingleCheckpointDelete() {
    if (deleteRequested === null) {
      return;
    }
    if (await deleteCheckpoint(deleteRequested)) {
      const deletedId = deleteRequested.id;
      setDeleteRequested(null);
      setSelectedCheckpointIds((current) => withoutSetValue(current, deletedId));
    }
  }

  async function confirmSelectedCheckpointDelete() {
    const targets = selectedCheckpoints;
    if (targets.length === 0) {
      setDeleteSelectedRequested(false);
      return;
    }
    setIsDeletingSelected(true);
    onGlobalError(null);
    try {
      for (const checkpoint of targets) {
        await onDeleteCheckpoint(checkpoint);
      }
      setSelectedCheckpointIds(new Set());
      setDeleteSelectedRequested(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "checkpoint delete failed");
    } finally {
      setIsDeletingSelected(false);
    }
  }

  async function confirmCheckpointInstall() {
    if (installRequested === null) {
      return;
    }
    if (await installEntry(installRequested)) {
      setInstallRequested(null);
    }
  }

  function setAllCheckpointsSelected(selected: boolean) {
    setSelectedCheckpointIds(
      selected ? new Set(installedCheckpoints.map((checkpoint) => checkpoint.id)) : new Set(),
    );
  }

  function toggleCheckpointSelection(checkpointId: string, selected: boolean) {
    setSelectedCheckpointIds((current) => {
      const next = new Set(current);
      if (selected) {
        next.add(checkpointId);
      } else {
        next.delete(checkpointId);
      }
      return next;
    });
  }

  return (
    <>
      <Panel>
        <div className="panel-header-row">
          <PanelHeader
            title="Checkpoints"
            subtitle="Official release bundles available for local install."
          />
          <Button
            className="gap-2"
            disabled={selectedCheckpointCount === 0 || isDeletingCheckpoint}
            tone="danger"
            type="button"
            onClick={() => setDeleteSelectedRequested(true)}
          >
            <TrashIcon />
            <span>
              {selectedCheckpointCount === 0
                ? "Delete selected"
                : `Delete selected (${selectedCheckpointCount})`}
            </span>
          </Button>
        </div>

        {error !== null ? <Notice tone="error">{error}</Notice> : null}
        {catalog === null ? (
          error === null ? (
            <Notice>Checkpoint catalog unavailable.</Notice>
          ) : null
        ) : catalog.entries.length === 0 ? (
          <Notice>No published checkpoints listed.</Notice>
        ) : (
          <ListTable minWidthClass="min-w-[860px]">
            <ListTableHead>
              <tr>
                <ListSelectAllHeaderCell
                  aria-label="Select all installed checkpoints"
                  checked={allInstalledCheckpointsSelected}
                  disabled={installedCheckpoints.length === 0 || isDeletingCheckpoint}
                  onChange={setAllCheckpointsSelected}
                />
                <th className="px-4 py-3">Checkpoint</th>
                <th className="px-4 py-3">Policy</th>
                <th className="px-4 py-3">Bundle</th>
                <th className="px-4 py-3">Status</th>
                <ListActionsHeaderCell />
              </tr>
            </ListTableHead>
            <tbody>
              {catalog.entries.map((entry) => (
                <CheckpointCatalogRow
                  entry={entry}
                  installedCheckpoint={
                    entry.installed_checkpoint_id === null
                      ? null
                      : (installedById.get(entry.installed_checkpoint_id) ?? null)
                  }
                  installing={installingKey === checkpointEntryKey(entry)}
                  isDeleting={isDeletingCheckpoint}
                  key={checkpointEntryKey(entry)}
                  selected={
                    entry.installed_checkpoint_id !== null &&
                    selectedCheckpointIds.has(entry.installed_checkpoint_id)
                  }
                  onDelete={(checkpoint) => setDeleteRequested(checkpoint)}
                  onInstall={() => setInstallRequested(entry)}
                  onOpen={onOpenCheckpoint}
                  onToggleSelection={toggleCheckpointSelection}
                />
              ))}
            </tbody>
          </ListTable>
        )}
      </Panel>
      <ConfirmDialog
        busy={installRequested !== null && installingKey === checkpointEntryKey(installRequested)}
        busyLabel="Downloading..."
        confirmLabel="Download"
        confirmTone="default"
        confirmVariant="primary"
        description={
          installRequested === null
            ? ""
            : `Download and install ${installRequested.name}? The checkpoint will be stored under local/manager/checkpoints and can then be opened, forked, watched, or evaluated.`
        }
        open={installRequested !== null}
        title="Download checkpoint"
        onClose={() => setInstallRequested(null)}
        onConfirm={() => void confirmCheckpointInstall()}
      />
      <ConfirmDialog
        busy={deletingCheckpointId !== null}
        confirmLabel="Delete checkpoint"
        description={
          deleteRequested === null
            ? ""
            : `Delete ${deleteRequested.name}? The imported checkpoint bundle will be removed locally.`
        }
        open={deleteRequested !== null}
        title="Delete checkpoint"
        onClose={() => setDeleteRequested(null)}
        onConfirm={() => void confirmSingleCheckpointDelete()}
      />
      <ConfirmDialog
        busy={isDeletingSelected}
        confirmLabel={`Delete ${selectedCheckpointCount}`}
        description={`Delete ${selectedCheckpointCount} selected checkpoint${
          selectedCheckpointCount === 1 ? "" : "s"
        }? Imported checkpoint bundles will be removed locally.`}
        open={deleteSelectedRequested}
        title="Delete selected checkpoints"
        onClose={() => setDeleteSelectedRequested(false)}
        onConfirm={() => void confirmSelectedCheckpointDelete()}
      />
    </>
  );
}

function CheckpointCatalogRow({
  entry,
  installedCheckpoint,
  installing,
  isDeleting,
  selected,
  onDelete,
  onInstall,
  onOpen,
  onToggleSelection,
}: {
  entry: CheckpointCatalogEntry;
  installedCheckpoint: PublishedCheckpoint | null;
  installing: boolean;
  isDeleting: boolean;
  selected: boolean;
  onDelete: (checkpoint: PublishedCheckpoint) => void;
  onInstall: () => void;
  onOpen: (checkpoint: PublishedCheckpoint) => void;
  onToggleSelection: (checkpointId: string, selected: boolean) => void;
}) {
  const checkpoint = entry.manifest.checkpoint;
  const openRow =
    installedCheckpoint === null
      ? onInstall
      : installedCheckpoint.run === null
        ? undefined
        : () => onOpen(installedCheckpoint);

  return (
    <ListRow selected={selected} onOpen={openRow}>
      <ListSelectionCell
        aria-label={`Select checkpoint: ${entry.name}`}
        checked={selected}
        disabled={installedCheckpoint === null || isDeleting}
        onChange={(checked) => {
          if (installedCheckpoint !== null) {
            onToggleSelection(installedCheckpoint.id, checked);
          }
        }}
      />
      <td className="px-4 py-3 align-top">
        <div className="font-semibold text-app-text">{entry.name}</div>
        <div className="mt-1 text-xs text-app-muted">
          {entry.id} · {entry.version}
        </div>
      </td>
      <td className="px-4 py-3 align-top text-app-muted">
        <div>{checkpoint.source_artifact}</div>
        <div className="mt-1 text-xs">
          {formatNullableInteger(checkpoint.lineage_num_timesteps)} lineage steps
        </div>
        <div className="mt-1 text-xs">
          {entry.manifest.compatibility.training_algorithm ?? "unknown algorithm"}
        </div>
      </td>
      <td className="px-4 py-3 align-top text-app-muted">
        <div>{entry.bundle.filename}</div>
        <div className="mt-1 text-xs">{formatBytes(entry.bundle.size_bytes)}</div>
      </td>
      <td className="px-4 py-3 align-top">
        <span className={installedCheckpoint !== null ? "text-app-accent" : "text-app-muted"}>
          {installedCheckpoint !== null ? "installed" : "available"}
        </span>
      </td>
      <ListActionsCell>
        <TooltipIconButton
          aria-label={`Download checkpoint: ${entry.name}`}
          disabled={installedCheckpoint !== null || installing}
          size="compact"
          tooltip={installedCheckpoint !== null ? "Installed" : "Download checkpoint"}
          onClick={onInstall}
        >
          <ExportIcon />
        </TooltipIconButton>
        <TooltipIconButton
          aria-label={`Delete checkpoint: ${entry.name}`}
          disabled={installedCheckpoint === null || isDeleting}
          size="compact"
          tone="danger"
          tooltip="Delete checkpoint"
          onClick={() => {
            if (installedCheckpoint !== null) {
              onDelete(installedCheckpoint);
            }
          }}
        >
          <TrashIcon />
        </TooltipIconButton>
      </ListActionsCell>
    </ListRow>
  );
}

function checkpointEntryKey(entry: CheckpointCatalogEntry) {
  return `${entry.id}:${entry.version}`;
}

function withoutSetValue<T>(values: ReadonlySet<T>, value: T): ReadonlySet<T> {
  const next = new Set(values);
  next.delete(value);
  return next;
}

function formatNullableInteger(value: number | null) {
  return value === null ? "unknown" : formatInteger(value);
}

function formatBytes(value: number) {
  if (value < 1024 * 1024) {
    return `${formatInteger(Math.round(value / 1024))} KiB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}
