import { ResetIcon } from "@/features/configurator/fields/reset";
import type { FieldLabelProps } from "@/features/configurator/fields/types";

export function FieldLabel({ help, label, onReset }: FieldLabelProps) {
  return (
    <span className="field-label">
      <span>{label}</span>
      <button
        aria-label={`${label}: ${help}`}
        className="field-help"
        data-tooltip={help}
        type="button"
      >
        ?
      </button>
      {onReset !== undefined ? (
        <button
          aria-label={`Reset ${label} to default`}
          className="field-reset-button"
          title="Reset to default"
          type="button"
          onClick={(event) => {
            event.preventDefault();
            onReset();
          }}
        >
          <ResetIcon />
        </button>
      ) : null}
    </span>
  );
}
