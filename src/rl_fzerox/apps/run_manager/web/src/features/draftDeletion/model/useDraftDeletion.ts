// src/rl_fzerox/apps/run_manager/web/src/features/draftDeletion/model/useDraftDeletion.ts
import { useEffect, useMemo, useState } from "react";
import type { ManagedDraft } from "@/shared/api/contract";

interface PendingDraftDelete {
  drafts: ManagedDraft[];
  title: string;
}

interface DraftDeletionOptions {
  drafts: readonly ManagedDraft[];
  onDeleteDraft: (draft: ManagedDraft) => Promise<void>;
}

export function useDraftDeletion({ drafts, onDeleteDraft }: DraftDeletionOptions) {
  const [pendingDelete, setPendingDelete] = useState<PendingDraftDelete | null>(null);
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

  function requestSingleDelete(draft: ManagedDraft) {
    setPendingDelete({
      drafts: [draft],
      title: "Delete draft",
    });
  }

  function requestSelectedDelete() {
    if (selectedDrafts.length === 0) {
      return;
    }
    setPendingDelete({
      drafts: selectedDrafts,
      title: selectedDrafts.length === 1 ? "Delete draft" : "Delete drafts",
    });
  }

  function closePendingDelete() {
    if (!isDeleting) {
      setPendingDelete(null);
    }
  }

  return {
    allDraftsSelected,
    confirmDelete,
    deleteDescription: pendingDelete === null ? "" : draftDeleteDescription(pendingDelete.drafts),
    deleteTitle: pendingDelete?.title ?? "Delete draft",
    isDeleting,
    pendingDelete,
    pendingDeleteConfirmLabel:
      pendingDelete?.drafts.length === 1 ? "Delete draft" : "Delete drafts",
    closePendingDelete,
    requestSelectedDelete,
    requestSingleDelete,
    selectedCount,
    selectedDraftIds,
    setAllDraftsSelected,
    toggleDraftSelection,
  };
}

function draftDeleteDescription(drafts: readonly ManagedDraft[]): string {
  if (drafts.length === 1) {
    return `Delete draft "${drafts[0].name}"? This cannot be undone.`;
  }
  return `Delete ${drafts.length} selected drafts? This cannot be undone.`;
}
