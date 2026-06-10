// web/run-manager/src/entities/runConfig/ui/sections/reward/RewardPanels.tsx
import { DamagePanel } from "@/entities/runConfig/ui/sections/reward/DamagePanel";
import { TimeProgressPanels } from "@/entities/runConfig/ui/sections/reward/TimeProgressPanels";
import { TrackActionPanels } from "@/entities/runConfig/ui/sections/reward/TrackActionPanels";
import type { RewardPanelProps } from "@/entities/runConfig/ui/sections/reward/types";

export function RewardPanels(props: RewardPanelProps) {
  return (
    <>
      <TimeProgressPanels {...props} />
      <TrackActionPanels {...props} />
      <DamagePanel {...props} />
    </>
  );
}
