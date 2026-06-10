// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/DisclosureToolbar.tsx
import { CollapseAllIcon, ExpandAllIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

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
      <TooltipIconButton
        aria-label={expandLabel}
        size="compact"
        tooltip="Expand all"
        onClick={onExpandAll}
      >
        <ExpandAllIcon />
      </TooltipIconButton>
      <TooltipIconButton
        aria-label={collapseLabel}
        size="compact"
        tooltip="Collapse all"
        onClick={onCollapseAll}
      >
        <CollapseAllIcon />
      </TooltipIconButton>
    </div>
  );
}
