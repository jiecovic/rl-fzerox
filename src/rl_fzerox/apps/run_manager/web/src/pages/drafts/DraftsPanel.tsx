// src/rl_fzerox/apps/run_manager/web/src/pages/drafts/DraftsPanel.tsx
import { useEffect, useMemo, useState } from "react";
import type { ManagedDraft } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { cn } from "@/shared/ui/cn";
import { formatDate } from "@/shared/ui/format";
import { PlusIcon, TrashIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface DraftsPanelProps {
  drafts: ManagedDraft[];
  onCreateDraft: () => void;
  onDeleteDraft: (draft: ManagedDraft) => Promise<void>;
  onOpenDraft: (draft: ManagedDraft) => void;
}

interface PendingDelete {
  drafts: ManagedDraft[];
  title: string;
}

export function DraftsPanel({
  drafts,
  onCreateDraft,
  onDeleteDraft,
  onOpenDraft,
}: DraftsPanelProps) {
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [selectedDraftIds, setSelectedDraftIds] = useState<ReadonlySet<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);

  const selectedDrafts = useMemo(
    () => drafts.filter((draft) => selectedDraftIds.has(draft.id)),
    [drafts, selectedDraftIds],
  );
  const selectedCount = selectedDrafts.length;
  const allDraftsSelected = drafts.length > 0 && selectedCount === drafts.length;

  useEffect(() => {
    const visibleDraftIds = new Set(drafts.map((draft) => draft.id));
    setSelectedDraftIds((current) => {
      const next = new Set([...current].filter((draftId) => visibleDraftIds.has(draftId)));
      return next.size === current.size ? current : next;
    });
  }, [drafts]);

  async function confirmDelete() {
    if (pendingDelete === null) {
      return;
    }
    const draftsToDelete = pendingDelete.drafts;
    setIsDeleting(true);
    try {
      await Promise.all(draftsToDelete.map((draft) => onDeleteDraft(draft)));
      setSelectedDraftIds((current) => {
        const deletedIds = new Set(draftsToDelete.map((draft) => draft.id));
        return new Set([...current].filter((draftId) => !deletedIds.has(draftId)));
      });
      setPendingDelete(null);
    } finally {
      setIsDeleting(false);
    }
  }

  function toggleDraftSelection(draftId: string, selected: boolean) {
    setSelectedDraftIds((current) => {
      const next = new Set(current);
      if (selected) {
        next.add(draftId);
      } else {
        next.delete(draftId);
      }
      return next;
    });
  }

  function setAllDraftsSelected(selected: boolean) {
    setSelectedDraftIds(selected ? new Set(drafts.map((draft) => draft.id)) : new Set());
  }

  function queueSingleDelete(draft: ManagedDraft) {
    setPendingDelete({
      drafts: [draft],
      title: "Delete draft",
    });
  }

  function queueSelectedDelete() {
    if (selectedDrafts.length === 0) {
      return;
    }
    setPendingDelete({
      drafts: selectedDrafts,
      title: selectedDrafts.length === 1 ? "Delete draft" : "Delete drafts",
    });
  }

  return (
    <>
      <Panel>
        <div className="panel-header-row">
          <PanelHeader title="Drafts" subtitle="Create or reopen a draft." />
          <div className="flex flex-wrap justify-end gap-2">
            {drafts.length > 0 ? (
              <Button
                className="gap-2"
                disabled={selectedCount === 0 || isDeleting}
                tone="danger"
                type="button"
                onClick={queueSelectedDelete}
              >
                <TrashIcon />
                <span>
                  {selectedCount === 0 ? "Delete selected" : `Delete selected (${selectedCount})`}
                </span>
              </Button>
            ) : null}
            <Button className="gap-2" type="button" onClick={onCreateDraft}>
              <PlusIcon />
              <span>Create draft</span>
            </Button>
          </div>
        </div>
        {drafts.length === 0 ? (
          <Notice>No drafts yet. Create one to open the configurator.</Notice>
        ) : (
          <div className="grid gap-2">
            <div className="grid grid-cols-[38px_minmax(0,1fr)_40px] items-center px-3.5 pb-1 text-[11px] font-bold tracking-[0.04em] text-app-muted uppercase">
              <label className={draftSelectCellClass}>
                <input
                  aria-label="Select all drafts"
                  className={draftCheckboxClass}
                  checked={allDraftsSelected}
                  disabled={isDeleting}
                  type="checkbox"
                  onChange={(event) => setAllDraftsSelected(event.currentTarget.checked)}
                />
              </label>
              <div className={draftRecordGridClass}>
                <span>Draft</span>
                <span>Envs</span>
                <span>Steps</span>
                <span>LR</span>
                <span>Policy</span>
                <span>Created at</span>
              </div>
              <span className="w-10" />
            </div>
            {drafts.map((draft) => (
              <div className={draftRowClass(selectedDraftIds.has(draft.id))} key={draft.id}>
                <label className={draftSelectCellClass}>
                  <input
                    aria-label={`Select draft ${draft.name}`}
                    className={draftCheckboxClass}
                    checked={selectedDraftIds.has(draft.id)}
                    disabled={isDeleting}
                    type="checkbox"
                    onChange={(event) =>
                      toggleDraftSelection(draft.id, event.currentTarget.checked)
                    }
                  />
                </label>
                <button
                  className={draftRecordButtonClass}
                  type="button"
                  onClick={() => onOpenDraft(draft)}
                >
                  <span className="font-semibold text-app-text">{draft.name}</span>
                  <span>{draft.config.train.num_envs} envs</span>
                  <span>{draft.config.train.total_timesteps.toLocaleString()} steps</span>
                  <span>{draft.config.train.learning_rate.toExponential(2)}</span>
                  <span>{draft.config.policy.conv_profile}</span>
                  <span className="whitespace-nowrap">{formatDate(draft.created_at)}</span>
                </button>
                <TooltipIconButton
                  aria-label={`Delete draft ${draft.name}`}
                  className="mr-2.5 justify-self-end"
                  disabled={isDeleting}
                  size="compact"
                  tone="danger"
                  tooltip="Delete draft"
                  onClick={() => queueSingleDelete(draft)}
                >
                  <TrashIcon />
                </TooltipIconButton>
              </div>
            ))}
          </div>
        )}
      </Panel>
      <ConfirmDialog
        busy={isDeleting}
        confirmLabel={pendingDelete?.drafts.length === 1 ? "Delete draft" : "Delete drafts"}
        description={pendingDelete === null ? "" : deleteDescription(pendingDelete.drafts)}
        open={pendingDelete !== null}
        title={pendingDelete?.title ?? "Delete draft"}
        onClose={() => {
          if (!isDeleting) {
            setPendingDelete(null);
          }
        }}
        onConfirm={() => void confirmDelete()}
      />
    </>
  );
}

function deleteDescription(drafts: ManagedDraft[]): string {
  if (drafts.length === 1) {
    return `Delete draft "${drafts[0].name}"? This cannot be undone.`;
  }
  return `Delete ${drafts.length} selected drafts? This cannot be undone.`;
}

const draftRecordGridClass =
  "grid grid-cols-[2fr_0.8fr_1fr_0.9fr_1.1fr_1.45fr] items-center gap-3 [&>span]:whitespace-nowrap";

const draftRecordButtonClass = cn(
  draftRecordGridClass,
  "min-h-12 min-w-0 border-0 bg-transparent px-3.5 text-left text-app-muted",
);

const draftSelectCellClass = "grid min-h-full place-items-center";
const draftCheckboxClass = "h-4 w-4 accent-app-accent";

function draftRowClass(selected: boolean) {
  return cn(
    "grid min-h-12 grid-cols-[38px_minmax(0,1fr)_40px] items-center overflow-hidden rounded-lg border border-app-border bg-app-surface hover:border-app-border-strong hover:bg-app-surface-muted",
    selected ? "border-app-border-strong bg-app-surface-muted" : undefined,
  );
}
