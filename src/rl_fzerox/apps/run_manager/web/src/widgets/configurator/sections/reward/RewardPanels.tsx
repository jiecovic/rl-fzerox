// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/reward/RewardPanels.tsx
import { DamagePanel } from "@/widgets/configurator/sections/reward/DamagePanel";
import { TimeProgressPanels } from "@/widgets/configurator/sections/reward/TimeProgressPanels";
import { TrackActionPanels } from "@/widgets/configurator/sections/reward/TrackActionPanels";
import type { RewardPanelProps } from "@/widgets/configurator/sections/reward/types";

export function RewardPanels(props: RewardPanelProps) {
  return (
    <>
      <TimeProgressPanels {...props} />
      <TrackActionPanels {...props} />
      <DamagePanel {...props} />
    </>
  );
}
