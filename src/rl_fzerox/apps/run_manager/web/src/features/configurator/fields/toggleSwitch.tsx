// src/rl_fzerox/apps/run_manager/web/src/features/configurator/fields/toggleSwitch.tsx
interface ToggleSwitchProps {
  checked: boolean;
  disabled?: boolean;
  hideLabel?: boolean;
  label: string;
  tooltip?: string;
  onChange: (checked: boolean) => void;
}

export function ToggleSwitch({
  checked,
  disabled = false,
  hideLabel = false,
  label,
  tooltip,
  onChange,
}: ToggleSwitchProps) {
  return (
    <label
      className={disabled ? "state-switch disabled tooltip-anchor" : "state-switch tooltip-anchor"}
      data-tooltip={tooltip}
    >
      <input
        aria-label={label}
        checked={checked}
        disabled={disabled}
        type="checkbox"
        onChange={(event) => onChange(event.target.checked)}
      />
      <span aria-hidden="true" />
      {hideLabel ? null : <strong>{label}</strong>}
    </label>
  );
}
