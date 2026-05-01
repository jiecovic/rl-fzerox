import { DamagePanel } from "./DamagePanel";
import { TimeProgressPanels } from "./TimeProgressPanels";
import { TrackActionPanels } from "./TrackActionPanels";
import type { RewardPanelProps } from "./types";

export function RewardPanels(props: RewardPanelProps) {
  return (
    <>
      <TimeProgressPanels {...props} />
      <TrackActionPanels {...props} />
      <DamagePanel {...props} />
    </>
  );
}
