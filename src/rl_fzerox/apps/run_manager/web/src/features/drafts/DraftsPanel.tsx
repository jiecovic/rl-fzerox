import { useState } from "react";
import type { ManagedDraft } from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface DraftsPanelProps {
  drafts: ManagedDraft[];
  onDeleteDraft: (draft: ManagedDraft) => Promise<void>;
  onOpenDraft: (draft: ManagedDraft) => void;
}

export function DraftsPanel({ drafts, onDeleteDraft, onOpenDraft }: DraftsPanelProps) {
  const [pendingDelete, setPendingDelete] = useState<ManagedDraft | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  if (drafts.length === 0) {
    return <Notice>No drafts yet. Configure a run and save it first.</Notice>;
  }

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
        <PanelHeader title="Drafts" subtitle="Click a row to load it into the configurator." />
        <div className="record-list">
          {drafts.map((draft) => (
            <div className="record-row" key={draft.id}>
              <button className="record-row-main" type="button" onClick={() => onOpenDraft(draft)}>
                <span className="record-name">{draft.name}</span>
                <span>{draft.config.train.num_envs} envs</span>
                <span>{draft.config.train.total_timesteps.toLocaleString()} steps</span>
                <span>{draft.config.train.learning_rate.toExponential(2)}</span>
                <span>{draft.config.policy.conv_profile}</span>
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

function TrashIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M6.5 6.5v7M10 6.5v7M13.5 6.5v7M4.5 4.5h11M7.5 4.5l.8-1.5h3.4l.8 1.5M6 4.5v11h8v-11"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}
