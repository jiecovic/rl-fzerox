// src/rl_fzerox/apps/run_manager/web/src/features/drafts/DraftsPanel.tsx
import { useEffect, useMemo, useState } from "react";
import type { ManagedDraft } from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { formatDate } from "@/shared/ui/format";
import { PlusIcon, TrashIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

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
          <div className="drafts-panel-actions">
            {drafts.length > 0 ? (
              <button
                className="secondary-button button-with-icon danger"
                disabled={selectedCount === 0 || isDeleting}
                type="button"
                onClick={queueSelectedDelete}
              >
                <TrashIcon />
                <span>
                  {selectedCount === 0 ? "Delete selected" : `Delete selected (${selectedCount})`}
                </span>
              </button>
            ) : null}
            <button
              className="secondary-button button-with-icon"
              type="button"
              onClick={onCreateDraft}
            >
              <PlusIcon />
              <span>Create draft</span>
            </button>
          </div>
        </div>
        {drafts.length === 0 ? (
          <Notice>No drafts yet. Create one to open the configurator.</Notice>
        ) : (
          <div className="record-list">
            <div className="record-head draft-record-head">
              <label className="draft-select-cell">
                <input
                  aria-label="Select all drafts"
                  checked={allDraftsSelected}
                  disabled={isDeleting}
                  type="checkbox"
                  onChange={(event) => setAllDraftsSelected(event.currentTarget.checked)}
                />
              </label>
              <div className="record-head-main">
                <span>Draft</span>
                <span>Envs</span>
                <span>Steps</span>
                <span>LR</span>
                <span>Policy</span>
                <span>Created at</span>
              </div>
              <span className="record-head-actions" />
            </div>
            {drafts.map((draft) => (
              <div
                className={`record-row draft-record-row${
                  selectedDraftIds.has(draft.id) ? " is-selected" : ""
                }`}
                key={draft.id}
              >
                <label className="draft-select-cell">
                  <input
                    aria-label={`Select draft ${draft.name}`}
                    checked={selectedDraftIds.has(draft.id)}
                    disabled={isDeleting}
                    type="checkbox"
                    onChange={(event) =>
                      toggleDraftSelection(draft.id, event.currentTarget.checked)
                    }
                  />
                </label>
                <button
                  className="record-row-main"
                  type="button"
                  onClick={() => onOpenDraft(draft)}
                >
                  <span className="record-name">{draft.name}</span>
                  <span>{draft.config.train.num_envs} envs</span>
                  <span>{draft.config.train.total_timesteps.toLocaleString()} steps</span>
                  <span>{draft.config.train.learning_rate.toExponential(2)}</span>
                  <span>{draft.config.policy.conv_profile}</span>
                  <span className="record-created-at">{formatDate(draft.created_at)}</span>
                </button>
                <button
                  aria-label={`Delete draft ${draft.name}`}
                  className="icon-button compact-icon-button danger"
                  disabled={isDeleting}
                  type="button"
                  onClick={() => queueSingleDelete(draft)}
                >
                  <TrashIcon />
                </button>
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
