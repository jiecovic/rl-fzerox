// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/RewardPanels.tsx
import { DamagePanel } from "@/features/configurator/sections/reward/DamagePanel";
import { TimeProgressPanels } from "@/features/configurator/sections/reward/TimeProgressPanels";
import { TrackActionPanels } from "@/features/configurator/sections/reward/TrackActionPanels";
import type { RewardPanelProps } from "@/features/configurator/sections/reward/types";

export function RewardPanels(props: RewardPanelProps) {
  return (
    <>
      <TimeProgressPanels {...props} />
      <TrackActionPanels {...props} />
      <DamagePanel {...props} />
    </>
  );
}
