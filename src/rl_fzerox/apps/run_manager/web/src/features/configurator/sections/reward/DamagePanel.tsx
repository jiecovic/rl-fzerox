// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/DamagePanel.tsx
import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { IntegerField, NumberField, OptionalRangePairField } from "@/features/configurator/fields";
import { damageDefaults } from "@/features/configurator/sections/rewardDefaults";

import type { RewardPanelProps } from "./types";

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
      title="Damage and terminal"
      onToggle={(open) => setSectionOpen("damage", open)}
      onReset={() => updateReward(damageDefaults(defaultConfig.reward))}
    >
      <div className="config-field-grid">
        <NumberField
          help="Penalty for frames that take damage."
          label="Damage frame penalty"
          resetValue={defaultConfig.reward.damage_taken_frame_penalty}
          step="0.001"
          value={config.reward.damage_taken_frame_penalty}
          onChange={(value) => updateReward({ damage_taken_frame_penalty: value })}
        />
        <NumberField
          help="Ramp penalty added during consecutive damage streaks."
          label="Damage streak ramp"
          resetValue={defaultConfig.reward.damage_taken_streak_ramp_penalty}
          step="0.001"
          value={config.reward.damage_taken_streak_ramp_penalty}
          onChange={(value) => updateReward({ damage_taken_streak_ramp_penalty: value })}
        />
        <IntegerField
          help="Maximum damage-streak frame count used by the ramp."
          label="Damage streak cap"
          resetValue={defaultConfig.reward.damage_taken_streak_cap_frames}
          value={config.reward.damage_taken_streak_cap_frames}
          onChange={(value) => updateReward({ damage_taken_streak_cap_frames: value })}
        />
        <NumberField
          help="Minimum repeated-step energy drop treated as real damage, filtering tiny native noise."
          label="Energy loss epsilon"
          resetValue={defaultConfig.reward.energy_loss_epsilon}
          step="0.001"
          value={config.reward.energy_loss_epsilon}
          onChange={(value) => updateReward({ energy_loss_epsilon: value })}
        />
        <NumberField
          help="Penalty when collision recoil is entered."
          label="Collision recoil penalty"
          resetValue={defaultConfig.reward.collision_recoil_penalty}
          step="0.5"
          value={config.reward.collision_recoil_penalty}
          onChange={(value) => updateReward({ collision_recoil_penalty: value })}
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
