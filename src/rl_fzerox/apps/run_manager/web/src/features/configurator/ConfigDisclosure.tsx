import type { ReactNode } from "react";

import { ResetIcon } from "@/shared/ui/icons";

export function ConfigDisclosure({
  children,
  defaultOpen = true,
  open,
  onReset,
  onToggle,
  title,
}: {
  children: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onReset?: () => void;
  onToggle?: (open: boolean) => void;
  title: string;
}) {
  return (
    <details
      className="config-disclosure"
      open={open ?? defaultOpen}
      onToggle={(event) => onToggle?.(event.currentTarget.open)}
    >
      <summary className="config-disclosure-summary">
        <span className="config-disclosure-title">
          <span className="config-disclosure-copy">
            <strong>{title}</strong>
          </span>
        </span>
        {onReset !== undefined ? (
          <button
            aria-label={`Reset ${title} defaults`}
            className="reset-button"
            type="button"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onReset();
            }}
          >
            <ResetIcon />
          </button>
        ) : null}
      </summary>
      <div className="config-disclosure-body">{children}</div>
    </details>
  );
}
