import type { ReactNode } from "react";

import { ResetIcon } from "@/shared/ui/icons";

export function ConfigPanel({
  children,
  onReset,
  title,
  wide = false,
}: {
  children: ReactNode;
  onReset?: () => void;
  title: string;
  wide?: boolean;
}) {
  return (
    <section className={wide ? "config-group wide" : "config-group"}>
      <div className="config-group-header">
        <h3>{title}</h3>
        {onReset !== undefined ? (
          <button
            aria-label={`Reset ${title} defaults`}
            className="reset-button tooltip-anchor"
            data-tooltip="Reset section defaults"
            type="button"
            onClick={onReset}
          >
            <ResetIcon />
          </button>
        ) : null}
      </div>
      {children}
    </section>
  );
}
