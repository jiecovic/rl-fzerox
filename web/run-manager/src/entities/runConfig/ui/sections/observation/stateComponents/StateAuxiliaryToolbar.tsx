// web/run-manager/src/entities/runConfig/ui/sections/observation/stateComponents/StateAuxiliaryToolbar.tsx
import { ToggleSwitch } from "@/shared/ui/configFields";

interface StateAuxiliaryToolbarProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

export function StateAuxiliaryToolbar({ enabled, onChange }: StateAuxiliaryToolbarProps) {
  return (
    <div className="flex items-center justify-between gap-3 border border-app-border bg-app-surface-muted px-3 py-2.5">
      <span className="grid gap-0.5">
        <strong className="text-[13px] text-app-text">Auxiliary RAM supervision</strong>
        <small className="text-xs text-app-muted">
          Optional supervised targets over the shared policy latent.
        </small>
      </span>
      <ToggleSwitch
        checked={enabled}
        disabled={false}
        hideLabel
        label="auxiliary state supervision enabled"
        tooltip={
          enabled
            ? "Disable auxiliary RAM supervision and clear active aux losses"
            : "Enable auxiliary RAM supervision"
        }
        onChange={onChange}
      />
    </div>
  );
}
