import { FieldLabel } from "@/features/configurator/fields";
import type { ManagedRun } from "@/shared/api/contract";

interface RunIdentityPanelProps {
  canRename: boolean;
  isRenaming: boolean;
  onRename: () => Promise<void>;
  onRunNameChange: (name: string) => void;
  run: ManagedRun;
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
    <div className="form-grid run-identity-grid run-identity-grid-readonly">
      <div className="field-shell">
        <FieldLabel
          help="Manager label for this run. Renaming it does not mutate the frozen training config."
          label="Run name"
        />
        <input
          aria-label="Run name"
          value={runName}
          onChange={(event) => onRunNameChange(event.target.value)}
        />
      </div>
      <div className="field-shell">
        <FieldLabel help="Frozen seed stored with this launched run config." label="Seed" />
        <input aria-label="Seed" readOnly value={run.config.seed} />
      </div>
      <div className="run-identity-actions">
        <button
          className="secondary-button"
          type="button"
          disabled={!canRename}
          onClick={() => void onRename()}
        >
          {isRenaming ? "Saving..." : "Save name"}
        </button>
      </div>
    </div>
  );
}
