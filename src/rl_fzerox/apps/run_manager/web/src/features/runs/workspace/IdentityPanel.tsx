// src/rl_fzerox/apps/run_manager/web/src/features/runs/workspace/IdentityPanel.tsx
import { FieldLabel } from "@/features/configurator/fields";
import type { ManagedRunDetail } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldShell } from "@/shared/ui/Field";

interface RunIdentityPanelProps {
  canRename: boolean;
  isRenaming: boolean;
  onRename: () => Promise<void>;
  onRunNameChange: (name: string) => void;
  run: ManagedRunDetail;
  runName: string;
}

export function RunIdentityPanel({
  canRename,
  isRenaming,
  onRename,
  onRunNameChange,
  run,
  runName,
}: RunIdentityPanelProps) {
  return (
    <div className="mb-[18px] grid grid-cols-[minmax(0,1fr)_240px] items-end gap-4">
      <FieldShell>
        <FieldLabel
          help="Manager label for this run. Renaming it does not mutate the frozen training config."
          label="Run name"
        />
        <FieldInput
          aria-label="Run name"
          value={runName}
          onChange={(event) => onRunNameChange(event.target.value)}
        />
      </FieldShell>
      <FieldShell>
        <FieldLabel help="Frozen seed stored with this launched run config." label="Seed" />
        <FieldInput aria-label="Seed" readOnly value={run.config.seed} />
      </FieldShell>
      <div className="flex items-end justify-end">
        <Button disabled={!canRename} onClick={() => void onRename()}>
          {isRenaming ? "Saving..." : "Save name"}
        </Button>
      </div>
    </div>
  );
}
