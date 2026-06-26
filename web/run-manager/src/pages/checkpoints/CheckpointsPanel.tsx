// web/run-manager/src/pages/checkpoints/CheckpointsPanel.tsx
import { useState } from "react";

import type { CheckpointCatalogResponse } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { formatInteger } from "@/shared/ui/format";
import { ImportIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface CheckpointsPanelProps {
  catalog: CheckpointCatalogResponse | null;
  error: string | null;
  onGlobalError: (message: string | null) => void;
  onInstallCheckpoint: (checkpointId: string, version: string) => Promise<void>;
}

export function CheckpointsPanel({
  catalog,
  error,
  onGlobalError,
  onInstallCheckpoint,
}: CheckpointsPanelProps) {
  const [installingKey, setInstallingKey] = useState<string | null>(null);

  async function installEntry(entry: CheckpointCatalogResponse["entries"][number]) {
    const key = checkpointEntryKey(entry);
    setInstallingKey(key);
    onGlobalError(null);
    try {
      await onInstallCheckpoint(entry.id, entry.version);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "checkpoint install failed");
    } finally {
      setInstallingKey(null);
    }
  }

  return (
    <Panel>
      <PanelHeader
        title="Checkpoints"
        subtitle="Official release bundles available for local install."
      />

      {error !== null ? <Notice tone="error">{error}</Notice> : null}
      {catalog === null ? (
        error === null ? (
          <Notice>Checkpoint catalog unavailable.</Notice>
        ) : null
      ) : catalog.entries.length === 0 ? (
        <Notice>No published checkpoints listed.</Notice>
      ) : (
        <div className="overflow-hidden border border-app-border bg-app-surface">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="bg-app-surface-muted text-xs uppercase tracking-[0.08em] text-app-muted">
              <tr>
                <th className="border-b border-app-border px-4 py-3 font-semibold">Checkpoint</th>
                <th className="border-b border-app-border px-4 py-3 font-semibold">Policy</th>
                <th className="border-b border-app-border px-4 py-3 font-semibold">Bundle</th>
                <th className="border-b border-app-border px-4 py-3 font-semibold">Status</th>
                <th className="border-b border-app-border px-4 py-3 text-right font-semibold">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {catalog.entries.map((entry) => (
                <CheckpointCatalogRow
                  entry={entry}
                  installing={installingKey === checkpointEntryKey(entry)}
                  key={checkpointEntryKey(entry)}
                  onInstall={() => void installEntry(entry)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function CheckpointCatalogRow({
  entry,
  installing,
  onInstall,
}: {
  entry: CheckpointCatalogResponse["entries"][number];
  installing: boolean;
  onInstall: () => void;
}) {
  const checkpoint = entry.manifest.checkpoint;
  const installed = entry.installed_checkpoint_id !== null;

  return (
    <tr className="border-b border-app-border last:border-b-0">
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
        <span className={installed ? "text-app-accent" : "text-app-muted"}>
          {installed ? "installed" : "available"}
        </span>
      </td>
      <td className="px-4 py-3 text-right align-top">
        <Button
          className="gap-2"
          disabled={installed || installing}
          type="button"
          onClick={onInstall}
        >
          <ImportIcon />
          {installed ? "Installed" : installing ? "Installing" : "Install"}
        </Button>
      </td>
    </tr>
  );
}

function checkpointEntryKey(entry: CheckpointCatalogResponse["entries"][number]) {
  return `${entry.id}:${entry.version}`;
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
