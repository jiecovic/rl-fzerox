// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/TrackActionPanels.tsx
import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { IntegerField, NumberField } from "@/features/configurator/fields";
import {
  actionDefaults,
  energyDefaults,
  trackEventDefaults,
} from "@/features/configurator/sections/rewardDefaults";

import type { RewardPanelProps } from "./types";

export function TrackActionPanels({
  config,
  defaultConfig,
  openSections,
  setSectionOpen,
  updateReward,
}: RewardPanelProps) {
  return (
    <>
      <ConfigDisclosure
        open={openSections.track}
        title="Track events"
        onToggle={(open) => setSectionOpen("track", open)}
        onReset={() => updateReward(trackEventDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Reward for completing a lap."
            label="Lap completion bonus"
            resetValue={defaultConfig.reward.lap_completion_bonus}
            step="0.5"
            value={config.reward.lap_completion_bonus}
            onChange={(value) => updateReward({ lap_completion_bonus: value })}
          />
          <NumberField
            help="Multiplier applied to lap completion reward by final position."
            label="Lap position multiplier"
            resetValue={defaultConfig.reward.lap_position_scale}
            step="0.1"
            value={config.reward.lap_position_scale}
            onChange={(value) => updateReward({ lap_position_scale: value })}
          />
          <NumberField
            help="Minimum energy loss treated as meaningful damage."
            label="Energy loss epsilon"
            resetValue={defaultConfig.reward.energy_loss_epsilon}
            step="0.001"
            value={config.reward.energy_loss_epsilon}
            onChange={(value) => updateReward({ energy_loss_epsilon: value })}
          />
          <NumberField
            help="Progress reward multiplier while on dirt."
            label="Dirt progress multiplier"
            resetValue={defaultConfig.reward.dirt_progress_multiplier}
            step="0.1"
            value={config.reward.dirt_progress_multiplier}
            onChange={(value) => updateReward({ dirt_progress_multiplier: value })}
          />
          <NumberField
            help="Progress reward multiplier while on ice."
            label="Ice progress multiplier"
            resetValue={defaultConfig.reward.ice_progress_multiplier}
            step="0.1"
            value={config.reward.ice_progress_multiplier}
            onChange={(value) => updateReward({ ice_progress_multiplier: value })}
          />
          <NumberField
            help="Penalty for entering dirt."
            label="Dirt entry penalty"
            resetValue={defaultConfig.reward.dirt_entry_penalty}
            step="0.01"
            value={config.reward.dirt_entry_penalty}
            onChange={(value) => updateReward({ dirt_entry_penalty: value })}
          />
          <NumberField
            help="Penalty for entering ice."
            label="Ice entry penalty"
            resetValue={defaultConfig.reward.ice_entry_penalty}
            step="0.01"
            value={config.reward.ice_entry_penalty}
            onChange={(value) => updateReward({ ice_entry_penalty: value })}
          />
        </div>
      </ConfigDisclosure>

      <ConfigDisclosure
        open={openSections.energy}
        title="Energy refill"
        onToggle={(open) => setSectionOpen("energy", open)}
        onReset={() => updateReward(energyDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Progress reward multiplier while on refill surface."
            label="Refill progress multiplier"
            resetValue={defaultConfig.reward.energy_refill_progress_multiplier}
            step="0.1"
            value={config.reward.energy_refill_progress_multiplier}
            onChange={(value) => updateReward({ energy_refill_progress_multiplier: value })}
          />
          <IntegerField
            help="Frames after collision during which refill surface reward is suppressed."
            label="Refill collision cooldown"
            resetValue={defaultConfig.reward.energy_refill_collision_cooldown_frames}
            value={config.reward.energy_refill_collision_cooldown_frames}
            onChange={(value) => updateReward({ energy_refill_collision_cooldown_frames: value })}
          />
        </div>
      </ConfigDisclosure>

      <ConfigDisclosure
        open={openSections.actions}
        title="Actions"
        onToggle={(open) => setSectionOpen("actions", open)}
        onReset={() => updateReward(actionDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Reward added when the policy requests manual boost."
            label="Boost use reward"
            resetValue={defaultConfig.reward.manual_boost_reward}
            step="0.01"
            value={config.reward.manual_boost_reward}
            onChange={(value) => updateReward({ manual_boost_reward: value })}
          />
          <NumberField
            help="Reward for entering a detected boost pad."
            label="Boost pad reward"
            resetValue={defaultConfig.reward.boost_pad_reward}
            step="0.5"
            value={config.reward.boost_pad_reward}
            onChange={(value) => updateReward({ boost_pad_reward: value })}
          />
          <NumberField
            help="Progress window used to make boost-pad rewards one-way and non-farmable."
            label="Boost pad progress window"
            resetValue={defaultConfig.reward.boost_pad_reward_progress_window}
            step="50"
            value={config.reward.boost_pad_reward_progress_window}
            onChange={(value) => updateReward({ boost_pad_reward_progress_window: value })}
          />
          <NumberField
            help="Per-step penalty for requesting air brake."
            label="Air brake request penalty"
            resetValue={defaultConfig.reward.air_brake_request_penalty}
            step="0.001"
            value={config.reward.air_brake_request_penalty}
            onChange={(value) => updateReward({ air_brake_request_penalty: value })}
          />
          <NumberField
            help="Per-step penalty for requesting lean."
            label="Lean request penalty"
            resetValue={defaultConfig.reward.lean_request_penalty}
            step="0.001"
            value={config.reward.lean_request_penalty}
            onChange={(value) => updateReward({ lean_request_penalty: value })}
          />
          <NumberField
            help="Penalty when requesting pitch-up while airborne."
            label="Airborne pitch-up penalty"
            resetValue={defaultConfig.reward.airborne_pitch_up_penalty}
            step="0.01"
            value={config.reward.airborne_pitch_up_penalty}
            onChange={(value) => updateReward({ airborne_pitch_up_penalty: value })}
          />
        </div>
      </ConfigDisclosure>
    </>
  );
}
