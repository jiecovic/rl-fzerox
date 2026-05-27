// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/ChoiceStrip.tsx
import { SegmentedChoiceStrip } from "@/features/configurator/fields";

export function ChoiceStrip({
  description,
  options,
}: {
  description: string;
  options: readonly {
    active: boolean;
    disabled?: boolean;
    key: string;
    label: string;
    tooltip?: string;
    onClick: () => void;
  }[];
}) {
  return (
    <div className="grid gap-2.5">
      <SegmentedChoiceStrip ariaLabel="Selection" options={options} />
      <p className="m-0 text-xs leading-snug text-app-muted">{description}</p>
    </div>
  );
}
