// src/rl_fzerox/apps/run_manager/web/src/features/configurator/DisclosureToolbar.tsx
import { IconButton } from "@/shared/ui/Button";
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
    <div className="flex justify-end gap-2">
      <IconButton
        aria-label={expandLabel}
        className="tooltip-anchor"
        data-tooltip="Expand all"
        size="compact"
        onClick={onExpandAll}
      >
        <ExpandAllIcon />
      </IconButton>
      <IconButton
        aria-label={collapseLabel}
        className="tooltip-anchor"
        data-tooltip="Collapse all"
        size="compact"
        onClick={onCollapseAll}
      >
        <CollapseAllIcon />
      </IconButton>
    </div>
  );
}
