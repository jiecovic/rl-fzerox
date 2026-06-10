// web/run-manager/src/pages/drafts/DraftsPanel.tsx
import { DraftList } from "@/entities/draft/ui/DraftList";
import { useDraftDeletion } from "@/features/draftDeletion/model/useDraftDeletion";
import type { ManagedDraft } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
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
  const draftDeletion = useDraftDeletion({ drafts, onDeleteDraft });

  return (
    <>
      <Panel>
        <div className="panel-header-row">
          <PanelHeader title="Drafts" subtitle="Create or reopen a draft." />
          <div className="flex flex-wrap justify-end gap-2">
            {drafts.length > 0 ? (
              <Button
                className="gap-2"
                disabled={draftDeletion.selectedCount === 0 || draftDeletion.isDeleting}
                tone="danger"
                type="button"
                onClick={draftDeletion.requestSelectedDelete}
              >
                <TrashIcon />
                <span>
                  {draftDeletion.selectedCount === 0
                    ? "Delete selected"
                    : `Delete selected (${draftDeletion.selectedCount})`}
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
          <DraftList
            allDraftsSelected={draftDeletion.allDraftsSelected}
            drafts={drafts}
            isDeleting={draftDeletion.isDeleting}
            selectedDraftIds={draftDeletion.selectedDraftIds}
            onOpenDraft={onOpenDraft}
            onRequestDelete={draftDeletion.requestSingleDelete}
            onSelectAll={draftDeletion.setAllDraftsSelected}
            onToggleDraftSelection={draftDeletion.toggleDraftSelection}
          />
        )}
      </Panel>
      <ConfirmDialog
        busy={draftDeletion.isDeleting}
        confirmLabel={draftDeletion.pendingDeleteConfirmLabel}
        description={draftDeletion.deleteDescription}
        open={draftDeletion.pendingDelete !== null}
        title={draftDeletion.deleteTitle}
        onClose={draftDeletion.closePendingDelete}
        onConfirm={() => void draftDeletion.confirmDelete()}
      />
    </>
  );
}
