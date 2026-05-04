import { useState } from "react";
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

export function DraftsPanel({
  drafts,
  onCreateDraft,
  onDeleteDraft,
  onOpenDraft,
}: DraftsPanelProps) {
  const [pendingDelete, setPendingDelete] = useState<ManagedDraft | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  async function confirmDelete() {
    if (pendingDelete === null) {
      return;
    }
    setIsDeleting(true);
    try {
      await onDeleteDraft(pendingDelete);
      setPendingDelete(null);
    } finally {
      setIsDeleting(false);
    }
  }

  return (
    <>
      <Panel>
        <div className="panel-header-row">
          <PanelHeader title="Drafts" subtitle="Create or reopen a draft." />
          <button
            className="secondary-button button-with-icon"
            type="button"
            onClick={onCreateDraft}
          >
            <PlusIcon />
            <span>Create draft</span>
          </button>
        </div>
        {drafts.length === 0 ? (
          <Notice>No drafts yet. Create one to open the configurator.</Notice>
        ) : (
          <div className="record-list">
            <div className="record-head" role="presentation">
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
              <div className="record-row" key={draft.id}>
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
                  type="button"
                  onClick={() => setPendingDelete(draft)}
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
        confirmLabel="Delete draft"
        description={
          pendingDelete === null
            ? ""
            : `Delete draft "${pendingDelete.name}"? This cannot be undone.`
        }
        open={pendingDelete !== null}
        title="Delete draft"
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
