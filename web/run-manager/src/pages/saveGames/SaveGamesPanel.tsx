// web/run-manager/src/pages/saveGames/SaveGamesPanel.tsx
import {
  summarizeSaveGameTargets,
  titleizeIdentifier,
  unlockCompletionFraction,
} from "@/entities/saveGame/model";
import { ProgressMeter } from "@/entities/saveGame/ui/ProgressMeter";
import { useSaveGameDeletion } from "@/features/saveGameDeletion/model/useSaveGameDeletion";
import type { ManagedSaveGame } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { cn } from "@/shared/ui/cn";
import { formatDate } from "@/shared/ui/format";
import { PlusIcon, TrashIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface SaveGamesPanelProps {
  onCreateSaveGame: () => void;
  onDeleteSaveGame: (saveGame: ManagedSaveGame) => Promise<void>;
  onOpenSaveGame: (saveGame: ManagedSaveGame) => void;
  saveGames: ManagedSaveGame[];
}

export function SaveGamesPanel({
  onCreateSaveGame,
  onDeleteSaveGame,
  onOpenSaveGame,
  saveGames,
}: SaveGamesPanelProps) {
  const saveGameDeletion = useSaveGameDeletion({ saveGames, onDeleteSaveGame });

  return (
    <>
      <Panel>
        <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
          <PanelHeader title="Career Mode" subtitle="Local game saves and unlock progress." />
          <div className="flex flex-wrap justify-end gap-2">
            {saveGames.length > 0 ? (
              <Button
                className="gap-2"
                disabled={saveGameDeletion.selectedCount === 0 || saveGameDeletion.isDeleting}
                tone="danger"
                type="button"
                onClick={saveGameDeletion.requestSelectedDelete}
              >
                <TrashIcon />
                <span>
                  {saveGameDeletion.selectedCount === 0
                    ? "Delete selected"
                    : `Delete selected (${saveGameDeletion.selectedCount})`}
                </span>
              </Button>
            ) : null}
            <Button className="gap-2" type="button" variant="primary" onClick={onCreateSaveGame}>
              <PlusIcon />
              <span>Create career save</span>
            </Button>
          </div>
        </div>
        {saveGames.length === 0 ? (
          <Notice>No career saves yet. Create one to prepare an unlock path.</Notice>
        ) : (
          <div className="overflow-x-auto border border-app-border bg-app-surface">
            <table className="w-full min-w-[960px] border-collapse text-left text-sm">
              <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
                <tr>
                  <th className="w-10 px-4 py-3">
                    <label className="grid place-items-center">
                      <input
                        aria-label="Select all career saves"
                        checked={saveGameDeletion.allSaveGamesSelected}
                        className={saveGameCheckboxClass}
                        disabled={saveGameDeletion.isDeleting}
                        type="checkbox"
                        onChange={(event) =>
                          saveGameDeletion.setAllSaveGamesSelected(event.currentTarget.checked)
                        }
                      />
                    </label>
                  </th>
                  <th className="px-4 py-3">Career save</th>
                  <th className="px-4 py-3">Progress</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Updated</th>
                  <th className="px-4 py-3">Path</th>
                  <th className="w-12 px-4 py-3">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {saveGames.map((saveGame) => (
                  <SaveGameRow
                    isDeleting={saveGameDeletion.isDeleting}
                    key={saveGame.id}
                    saveGame={saveGame}
                    selected={saveGameDeletion.selectedSaveGameIds.has(saveGame.id)}
                    onOpen={onOpenSaveGame}
                    onRequestDelete={saveGameDeletion.requestSingleDelete}
                    onToggleSelection={saveGameDeletion.toggleSaveGameSelection}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <ConfirmDialog
        busy={saveGameDeletion.isDeleting}
        confirmLabel={saveGameDeletion.pendingDeleteConfirmLabel}
        description={saveGameDeletion.deleteDescription}
        error={saveGameDeletion.deleteError}
        open={saveGameDeletion.pendingDelete !== null}
        title={saveGameDeletion.deleteTitle}
        onClose={saveGameDeletion.closePendingDelete}
        onConfirm={() => void saveGameDeletion.confirmDelete()}
      />
    </>
  );
}

function SaveGameRow({
  isDeleting,
  onOpen,
  onRequestDelete,
  onToggleSelection,
  saveGame,
  selected,
}: {
  isDeleting: boolean;
  onOpen: (saveGame: ManagedSaveGame) => void;
  onRequestDelete: (saveGame: ManagedSaveGame) => void;
  onToggleSelection: (saveGameId: string, selected: boolean) => void;
  saveGame: ManagedSaveGame;
  selected: boolean;
}) {
  const targetSummary = summarizeSaveGameTargets(saveGame);
  const completion = unlockCompletionFraction(targetSummary);
  return (
    <tr
      className={saveGameRowClass(selected)}
      tabIndex={0}
      onClick={(event) => {
        if (isSaveGameRowInteractionTarget(event.target)) {
          return;
        }
        onOpen(saveGame);
      }}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) {
          return;
        }
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(saveGame);
        }
      }}
    >
      <td className="px-4 py-3 align-top" data-save-game-row-interaction>
        <label className="grid place-items-center">
          <input
            aria-label={`Select career save ${saveGame.name}`}
            checked={selected}
            className={saveGameCheckboxClass}
            disabled={isDeleting}
            type="checkbox"
            onChange={(event) => onToggleSelection(saveGame.id, event.currentTarget.checked)}
          />
        </label>
      </td>
      <td className="px-4 py-3 align-top">
        <div className="grid gap-1">
          <div className="font-semibold text-app-text">{saveGame.name}</div>
          <div className="text-xs capitalize text-app-muted">{saveGame.status}</div>
        </div>
      </td>
      <td className="w-[240px] px-4 py-3 align-top">
        <div className="grid gap-1.5">
          <div className="text-xs text-app-muted">
            {targetSummary.succeeded.toLocaleString()} / {targetSummary.total.toLocaleString()}{" "}
            targets
          </div>
          <ProgressMeter label={`${saveGame.name} progress`} value={completion} />
        </div>
      </td>
      <td className="px-4 py-3 align-top text-app-muted">{titleizeIdentifier(saveGame.status)}</td>
      <td className="px-4 py-3 align-top text-app-muted">{formatDate(saveGame.updated_at)}</td>
      <td className="max-w-[280px] overflow-hidden px-4 py-3 align-top font-mono text-xs text-ellipsis whitespace-nowrap text-app-muted">
        {saveGame.save_path}
      </td>
      <td className="px-4 py-3 align-top" data-save-game-row-interaction>
        <TooltipIconButton
          aria-label={`Delete career save ${saveGame.name}`}
          disabled={isDeleting}
          size="compact"
          tone="danger"
          tooltip="Delete career save"
          onClick={(event) => {
            event.stopPropagation();
            onRequestDelete(saveGame);
          }}
        >
          <TrashIcon />
        </TooltipIconButton>
      </td>
    </tr>
  );
}

const saveGameCheckboxClass = "h-4 w-4 accent-app-accent";

function saveGameRowClass(selected: boolean) {
  return cn(
    "cursor-pointer border-b border-app-border transition-colors last:border-b-0 hover:bg-app-surface-muted",
    selected ? "bg-app-surface-muted" : undefined,
  );
}

function isSaveGameRowInteractionTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest("[data-save-game-row-interaction],a,button,input,label,select,textarea") !== null
  );
}
