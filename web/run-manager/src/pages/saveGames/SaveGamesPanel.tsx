// web/run-manager/src/pages/saveGames/SaveGamesPanel.tsx
import {
  summarizeSaveGameTargets,
  titleizeIdentifier,
  unlockCompletionFraction,
} from "@/entities/saveGame/model";
import { ProgressMeter } from "@/entities/saveGame/ui/ProgressMeter";
import type { ManagedSaveGame } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { formatDate } from "@/shared/ui/format";
import { PlusIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface SaveGamesPanelProps {
  onCreateSaveGame: () => void;
  onOpenSaveGame: (saveGame: ManagedSaveGame) => void;
  saveGames: ManagedSaveGame[];
}

export function SaveGamesPanel({
  onCreateSaveGame,
  onOpenSaveGame,
  saveGames,
}: SaveGamesPanelProps) {
  return (
    <Panel>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <PanelHeader title="Career Mode" subtitle="Local game saves and unlock progress." />
        <Button className="gap-2" type="button" variant="primary" onClick={onCreateSaveGame}>
          <PlusIcon />
          <span>Create career save</span>
        </Button>
      </div>
      {saveGames.length === 0 ? (
        <Notice>No career saves yet. Create one to prepare an unlock path.</Notice>
      ) : (
        <div className="overflow-x-auto border border-app-border bg-app-surface">
          <table className="w-full min-w-[860px] border-collapse text-left text-sm">
            <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
              <tr>
                <th className="px-4 py-3">Career save</th>
                <th className="px-4 py-3">Progress</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Updated</th>
                <th className="px-4 py-3">Path</th>
              </tr>
            </thead>
            <tbody>
              {saveGames.map((saveGame) => (
                <SaveGameRow key={saveGame.id} saveGame={saveGame} onOpen={onOpenSaveGame} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function SaveGameRow({
  onOpen,
  saveGame,
}: {
  onOpen: (saveGame: ManagedSaveGame) => void;
  saveGame: ManagedSaveGame;
}) {
  const targetSummary = summarizeSaveGameTargets(saveGame);
  const completion = unlockCompletionFraction(targetSummary);
  return (
    <tr
      className="cursor-pointer border-b border-app-border transition-colors last:border-b-0 hover:bg-app-surface-muted"
      tabIndex={0}
      onClick={() => onOpen(saveGame)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen(saveGame);
        }
      }}
    >
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
    </tr>
  );
}
