import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import {
  BooleanField,
  IntegerField,
  NumberField,
  OptionalNumberField,
} from "@/features/configurator/fields";
import {
  airborneDefaults,
  progressDefaults,
  timePressureDefaults,
} from "@/features/configurator/sections/rewardDefaults";

import type { RewardPanelProps } from "./types";

export function TimeProgressPanels({
  config,
  defaultConfig,
  openSections,
  setSectionOpen,
  updateReward,
}: RewardPanelProps) {
  return (
    <>
      <ConfigDisclosure
        open={openSections.time}
        title="Time pressure"
        onToggle={(open) => setSectionOpen("time", open)}
        onReset={() => updateReward(timePressureDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Base reward added every emulator frame. Usually negative to apply time pressure."
            label="Time penalty / frame"
            resetValue={defaultConfig.reward.time_penalty_per_frame}
            step="0.001"
            value={config.reward.time_penalty_per_frame}
            onChange={(value) => updateReward({ time_penalty_per_frame: value })}
          />
          <NumberField
            help="Multiplier applied to time pressure while reversing."
            label="Reverse penalty multiplier"
            resetValue={defaultConfig.reward.reverse_time_penalty_scale}
            step="0.1"
            value={config.reward.reverse_time_penalty_scale}
            onChange={(value) => updateReward({ reverse_time_penalty_scale: value })}
          />
          <NumberField
            help="Multiplier applied to time pressure while below the native low-speed threshold."
            label="Low-speed multiplier"
            resetValue={defaultConfig.reward.low_speed_time_penalty_scale}
            step="0.1"
            value={config.reward.low_speed_time_penalty_scale}
            onChange={(value) => updateReward({ low_speed_time_penalty_scale: value })}
          />
          <NumberField
            help="Additional multiplier applied near very low speed."
            label="Slow-speed multiplier"
            resetValue={defaultConfig.reward.slow_speed_time_penalty_scale}
            step="0.1"
            value={config.reward.slow_speed_time_penalty_scale}
            onChange={(value) => updateReward({ slow_speed_time_penalty_scale: value })}
          />
          <NumberField
            help="Speed below which the slow-speed time-pressure ramp starts."
            label="Slow-speed start"
            resetValue={defaultConfig.reward.slow_speed_time_penalty_start_kph}
            step="10"
            value={config.reward.slow_speed_time_penalty_start_kph}
            onChange={(value) => updateReward({ slow_speed_time_penalty_start_kph: value })}
          />
          <NumberField
            help="Exponent for the slow-speed time-pressure ramp."
            label="Slow-speed power"
            resetValue={defaultConfig.reward.slow_speed_time_penalty_power}
            step="0.1"
            value={config.reward.slow_speed_time_penalty_power}
            onChange={(value) => updateReward({ slow_speed_time_penalty_power: value })}
          />
        </div>
      </ConfigDisclosure>

      <ConfigDisclosure
        open={openSections.progress}
        title="Frontier progress"
        onToggle={(open) => setSectionOpen("progress", open)}
        onReset={() => updateReward(progressDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Spline distance represented by one progress reward bucket."
            label="Bucket distance"
            resetValue={defaultConfig.reward.progress_bucket_distance}
            step="25"
            value={config.reward.progress_bucket_distance}
            onChange={(value) => updateReward({ progress_bucket_distance: value })}
          />
          <NumberField
            help="Reward paid per newly covered progress bucket."
            label="Bucket reward"
            resetValue={defaultConfig.reward.progress_bucket_reward}
            step="0.01"
            value={config.reward.progress_bucket_reward}
            onChange={(value) => updateReward({ progress_bucket_reward: value })}
          />
          <IntegerField
            help="Minimum frame interval between progress reward payouts."
            label="Progress interval frames"
            min={1}
            resetValue={defaultConfig.reward.progress_reward_interval_frames}
            value={config.reward.progress_reward_interval_frames}
            onChange={(value) => updateReward({ progress_reward_interval_frames: value })}
          />
          <OptionalNumberField
            defaultValue={100}
            help="Optional larger bucket distance used while airborne."
            label="Airborne bucket distance"
            max={5_000}
            min={1}
            resetValue={defaultConfig.reward.airborne_progress_bucket_distance}
            step="25"
            value={config.reward.airborne_progress_bucket_distance}
            onChange={(value) => updateReward({ airborne_progress_bucket_distance: value })}
          />
          <OptionalNumberField
            defaultValue={10_000}
            help="Caps deferred outside-track re-entry progress distance."
            label="Re-entry distance cap"
            max={50_000}
            min={0}
            resetValue={defaultConfig.reward.outside_bounds_reentry_progress_distance_cap}
            step="100"
            value={config.reward.outside_bounds_reentry_progress_distance_cap}
            onChange={(value) =>
              updateReward({ outside_bounds_reentry_progress_distance_cap: value })
            }
          />
        </div>
      </ConfigDisclosure>

      <ConfigDisclosure
        open={openSections.airborne}
        title="Airborne off-track"
        onToggle={(open) => setSectionOpen("airborne", open)}
        onReset={() => updateReward(airborneDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <BooleanField
            help="Only pay airborne off-track recovery reward while height is descending."
            label="Require descending"
            resetValue={defaultConfig.reward.airborne_offtrack_recovery_requires_descending}
            value={config.reward.airborne_offtrack_recovery_requires_descending}
            onChange={(value) =>
              updateReward({ airborne_offtrack_recovery_requires_descending: value })
            }
          />
          <NumberField
            help="Minimum height drop required when descending-gated recovery is enabled."
            label="Descend epsilon"
            resetValue={defaultConfig.reward.airborne_offtrack_recovery_descend_epsilon}
            step="0.1"
            value={config.reward.airborne_offtrack_recovery_descend_epsilon}
            onChange={(value) =>
              updateReward({ airborne_offtrack_recovery_descend_epsilon: value })
            }
          />
          <NumberField
            help="Penalty multiplier while airborne and outside track bounds."
            label="Off-track penalty multiplier"
            resetValue={defaultConfig.reward.airborne_offtrack_penalty_scale}
            step="0.01"
            value={config.reward.airborne_offtrack_penalty_scale}
            onChange={(value) => updateReward({ airborne_offtrack_penalty_scale: value })}
          />
          <NumberField
            help="Recovery reward multiplier for reducing off-track distance while airborne."
            label="Recovery reward multiplier"
            resetValue={defaultConfig.reward.airborne_offtrack_recovery_reward_scale}
            step="0.01"
            value={config.reward.airborne_offtrack_recovery_reward_scale}
            onChange={(value) => updateReward({ airborne_offtrack_recovery_reward_scale: value })}
          />
          <NumberField
            help="Reward paid on landing after airborne time."
            label="Landing reward"
            resetValue={defaultConfig.reward.airborne_landing_reward}
            step="0.5"
            value={config.reward.airborne_landing_reward}
            onChange={(value) => updateReward({ airborne_landing_reward: value })}
          />
        </div>
      </ConfigDisclosure>
    </>
  );
}
