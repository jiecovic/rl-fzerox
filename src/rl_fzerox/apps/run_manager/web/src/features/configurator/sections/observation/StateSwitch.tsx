interface StateSwitchProps {
  checked: boolean;
  disabled?: boolean;
  hideLabel?: boolean;
  label: string;
  tooltip?: string;
  onChange: (checked: boolean) => void;
}

export function StateSwitch({
  checked,
  disabled = false,
  hideLabel = false,
  label,
  tooltip,
  onChange,
}: StateSwitchProps) {
  return (
    <label
      className={disabled ? "state-switch disabled tooltip-anchor" : "state-switch tooltip-anchor"}
      data-tooltip={tooltip}
    >
      <input
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
