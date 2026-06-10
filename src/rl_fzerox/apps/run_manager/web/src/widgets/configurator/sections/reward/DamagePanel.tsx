// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/reward/DamagePanel.tsx
import { ConfigDisclosure } from "@/widgets/configurator/ConfigDisclosure";
import { NumberField, OptionalRangePairField } from "@/widgets/configurator/fields";
import type { RewardPanelProps } from "@/widgets/configurator/sections/reward/types";
import { damageDefaults } from "@/widgets/configurator/sections/rewardDefaults";

export function DamagePanel({
  config,
  defaultConfig,
  openSections,
  setSectionOpen,
  updateReward,
}: RewardPanelProps) {
  return (
    <ConfigDisclosure
      open={openSections.damage}
      title="Impact, energy, and terminal"
      onToggle={(open) => setSectionOpen("damage", open)}
      onReset={() => updateReward(damageDefaults(defaultConfig.reward))}
    >
      <div className="config-field-grid">
        <NumberField
          help="Penalty for each frame where the game reports damage or collision recoil."
          label="Impact frame penalty"
          resetValue={defaultConfig.reward.impact_frame_penalty}
          step="0.001"
          value={config.reward.impact_frame_penalty}
          onChange={(value) => updateReward({ impact_frame_penalty: value })}
        />
        <NumberField
          help="Penalty per energy point lost, including boost drain and collision damage."
          label="Energy loss penalty"
          resetValue={defaultConfig.reward.energy_loss_penalty}
          step="0.001"
          value={config.reward.energy_loss_penalty}
          onChange={(value) => updateReward({ energy_loss_penalty: value })}
        />
        <NumberField
          help="Reward per energy point gained, paid only when a progress bucket is also rewarded."
          label="Energy gain reward"
          resetValue={defaultConfig.reward.energy_gain_reward}
          step="0.001"
          value={config.reward.energy_gain_reward}
          onChange={(value) => updateReward({ energy_gain_reward: value })}
        />
        <NumberField
          help="Minimum repeated-step energy change counted as loss or gain, filtering tiny native noise."
          label="Energy loss epsilon"
          resetValue={defaultConfig.reward.energy_loss_epsilon}
          step="0.001"
          value={config.reward.energy_loss_epsilon}
          onChange={(value) => updateReward({ energy_loss_epsilon: value })}
        />
        <NumberField
          help="Penalty for terminal race failure."
          label="Failure penalty"
          resetValue={defaultConfig.reward.failure_penalty}
          step="1"
          value={config.reward.failure_penalty}
          onChange={(value) => updateReward({ failure_penalty: value })}
        />
        <NumberField
          help="Penalty for truncation."
          label="Truncation penalty"
          resetValue={defaultConfig.reward.truncation_penalty}
          step="1"
          value={config.reward.truncation_penalty}
          onChange={(value) => updateReward({ truncation_penalty: value })}
        />
        <OptionalRangePairField
          defaultMax={100}
          defaultMin={-100}
          help="Optional final per-step reward bounds after all terms are summed."
          label="Step reward clip"
          max={500}
          min={-500}
          resetMax={defaultConfig.reward.step_reward_clip_max}
          resetMin={defaultConfig.reward.step_reward_clip_min}
          step={5}
          valueMax={config.reward.step_reward_clip_max}
          valueMin={config.reward.step_reward_clip_min}
          onChange={(value) =>
            updateReward({
              step_reward_clip_max: value.max,
              step_reward_clip_min: value.min,
            })
          }
        />
      </div>
    </ConfigDisclosure>
  );
}
