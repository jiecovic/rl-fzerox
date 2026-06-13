// web/run-manager/src/features/saveGameDeletion/model/useSaveGameDeletion.ts
import { useEffect, useMemo, useState } from "react";

import type { ManagedSaveGame } from "@/shared/api/contract";

interface PendingSaveGameDelete {
  saveGames: ManagedSaveGame[];
  title: string;
}

interface SaveGameDeletionOptions {
  onDeleteSaveGame: (saveGame: ManagedSaveGame) => Promise<void>;
  saveGames: readonly ManagedSaveGame[];
}

export function useSaveGameDeletion({ onDeleteSaveGame, saveGames }: SaveGameDeletionOptions) {
  const [pendingDelete, setPendingDelete] = useState<PendingSaveGameDelete | null>(null);
  const [selectedSaveGameIds, setSelectedSaveGameIds] = useState<ReadonlySet<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const selectedSaveGames = useMemo(
    () => saveGames.filter((saveGame) => selectedSaveGameIds.has(saveGame.id)),
    [saveGames, selectedSaveGameIds],
  );
  const selectedCount = selectedSaveGames.length;
  const allSaveGamesSelected = saveGames.length > 0 && selectedCount === saveGames.length;

  useEffect(() => {
    const visibleSaveGameIds = new Set(saveGames.map((saveGame) => saveGame.id));
    setSelectedSaveGameIds((current) => {
      const next = new Set([...current].filter((saveGameId) => visibleSaveGameIds.has(saveGameId)));
      return next.size === current.size ? current : next;
    });
  }, [saveGames]);

  async function confirmDelete() {
    if (pendingDelete === null) {
      return;
    }
    const saveGamesToDelete = pendingDelete.saveGames;
    setDeleteError(null);
    setIsDeleting(true);
    try {
      await Promise.all(saveGamesToDelete.map((saveGame) => onDeleteSaveGame(saveGame)));
      setSelectedSaveGameIds((current) => {
        const deletedIds = new Set(saveGamesToDelete.map((saveGame) => saveGame.id));
        return new Set([...current].filter((saveGameId) => !deletedIds.has(saveGameId)));
      });
      setPendingDelete(null);
    } catch (caught) {
      setDeleteError(caught instanceof Error ? caught.message : "delete failed");
    } finally {
      setIsDeleting(false);
    }
  }

  function toggleSaveGameSelection(saveGameId: string, selected: boolean) {
    setSelectedSaveGameIds((current) => {
      const next = new Set(current);
      if (selected) {
        next.add(saveGameId);
      } else {
        next.delete(saveGameId);
      }
      return next;
    });
  }

  function setAllSaveGamesSelected(selected: boolean) {
    setSelectedSaveGameIds(
      selected ? new Set(saveGames.map((saveGame) => saveGame.id)) : new Set(),
    );
  }

  function requestSingleDelete(saveGame: ManagedSaveGame) {
    setDeleteError(null);
    setPendingDelete({
      saveGames: [saveGame],
      title: "Delete career save",
    });
  }

  function requestSelectedDelete() {
    if (selectedSaveGames.length === 0) {
      return;
    }
    setDeleteError(null);
    setPendingDelete({
      saveGames: selectedSaveGames,
      title: selectedSaveGames.length === 1 ? "Delete career save" : "Delete career saves",
    });
  }

  function closePendingDelete() {
    if (!isDeleting) {
      setDeleteError(null);
      setPendingDelete(null);
    }
  }

  return {
    allSaveGamesSelected,
    closePendingDelete,
    confirmDelete,
    deleteDescription:
      pendingDelete === null ? "" : saveGameDeleteDescription(pendingDelete.saveGames),
    deleteError,
    deleteTitle: pendingDelete?.title ?? "Delete career save",
    isDeleting,
    pendingDelete,
    pendingDeleteConfirmLabel:
      pendingDelete?.saveGames.length === 1 ? "Delete career save" : "Delete career saves",
    requestSelectedDelete,
    requestSingleDelete,
    selectedCount,
    selectedSaveGameIds,
    setAllSaveGamesSelected,
    toggleSaveGameSelection,
  };
}

function saveGameDeleteDescription(saveGames: readonly ManagedSaveGame[]): string {
  if (saveGames.length === 1) {
    return `Delete career save "${saveGames[0].name}"? This cannot be undone.`;
  }
  return `Delete ${saveGames.length} selected career saves? This cannot be undone.`;
}
