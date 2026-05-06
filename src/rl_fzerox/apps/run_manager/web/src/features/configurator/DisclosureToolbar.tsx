// src/rl_fzerox/apps/run_manager/web/src/features/configurator/DisclosureToolbar.tsx
import { CollapseAllIcon, ExpandAllIcon } from "@/shared/ui/icons";

interface DisclosureToolbarProps {
  expandLabel: string;
  collapseLabel: string;
  onExpandAll: () => void;
  onCollapseAll: () => void;
}

export function DisclosureToolbar({
  expandLabel,
  collapseLabel,
  onExpandAll,
  onCollapseAll,
}: DisclosureToolbarProps) {
  return (
    <div className="disclosure-toolbar">
      <button
        aria-label={expandLabel}
        className="icon-button compact-icon-button tooltip-anchor"
        data-tooltip="Expand all"
        type="button"
        onClick={onExpandAll}
      >
        <ExpandAllIcon />
      </button>
      <button
        aria-label={collapseLabel}
        className="icon-button compact-icon-button tooltip-anchor"
        data-tooltip="Collapse all"
        type="button"
        onClick={onCollapseAll}
      >
        <CollapseAllIcon />
      </button>
    </div>
  );
}
