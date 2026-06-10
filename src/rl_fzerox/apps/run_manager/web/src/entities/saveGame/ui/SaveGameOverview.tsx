// src/rl_fzerox/apps/run_manager/web/src/entities/saveGame/ui/SaveGameOverview.tsx
import type { UnlockTargetSummary } from "@/entities/saveGame/model";
import { titleizeIdentifier } from "@/entities/saveGame/model";
import { ProgressMeter } from "@/entities/saveGame/ui/ProgressMeter";
import type { ManagedSaveGame, ManagedSaveUnlockTarget } from "@/shared/api/contract";
import { formatDate } from "@/shared/ui/format";
import { CopyIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export function SaveGameOverview({
  completion,
  copiedSaveId,
  nextTarget,
  onCopySaveId,
  runnerLabel,
  saveGame,
  targetSummary,
}: {
  completion: number;
  copiedSaveId: boolean;
  nextTarget: ManagedSaveUnlockTarget | null;
  onCopySaveId: () => void;
  runnerLabel: string;
  saveGame: ManagedSaveGame;
  targetSummary: UnlockTargetSummary;
}) {
  return (
    <section className="grid gap-5 border border-app-border bg-app-surface p-5 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="grid content-start gap-4">
        <div className="grid gap-1">
          <h3 className="m-0 text-lg font-bold text-app-text">Unlock progress</h3>
          <p className="m-0 text-sm text-app-muted">
            {targetSummary.succeeded.toLocaleString()} of {targetSummary.total.toLocaleString()}{" "}
            targets complete
          </p>
        </div>
        <ProgressMeter label={`${saveGame.name} progress`} value={completion} />
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatusCard label="Targets" value={targetSummary.total.toLocaleString()} />
          <StatusCard label="Status" value={titleizeIdentifier(saveGame.status)} />
          <StatusCard label="Next target" value={nextTarget?.label ?? "none"} />
          <StatusCard label="Runner" value={runnerLabel} />
        </div>
      </div>
      <dl className="grid content-start gap-3 border border-app-border bg-app-surface-muted p-3 text-sm">
        <DetailRow
          copyLabel={copiedSaveId ? "Copied" : "Copy save id"}
          label="Save id"
          value={saveGame.id}
          monospace
          onCopy={onCopySaveId}
        />
        <DetailRow label="Save path" value={saveGame.save_path} monospace />
        <DetailRow label="Created" value={formatDate(saveGame.created_at)} />
        <DetailRow label="Updated" value={formatDate(saveGame.updated_at)} />
      </dl>
    </section>
  );
}

function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 border border-app-border bg-app-surface-muted px-3 py-2">
      <div className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</div>
      <div className="mt-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-sm font-semibold text-app-text">
        {value}
      </div>
    </div>
  );
}

function DetailRow({
  copyLabel,
  label,
  monospace = false,
  onCopy,
  value,
}: {
  copyLabel?: string;
  label: string;
  monospace?: boolean;
  onCopy?: () => void;
  value: string;
}) {
  return (
    <div className="grid gap-1">
      <dt className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</dt>
      <dd className="m-0 grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
        <span
          className={`min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-app-text ${
            monospace ? "font-mono text-xs" : ""
          }`}
          title={value}
        >
          {value}
        </span>
        {onCopy === undefined ? null : (
          <TooltipIconButton
            aria-label={copyLabel ?? `Copy ${label.toLowerCase()}`}
            size="compact"
            tooltip={copyLabel ?? `Copy ${label.toLowerCase()}`}
            onClick={onCopy}
          >
            <CopyIcon />
          </TooltipIconButton>
        )}
      </dd>
    </div>
  );
}
