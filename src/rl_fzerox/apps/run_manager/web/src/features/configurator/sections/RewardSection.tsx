import { useState } from "react";

import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import {
  BooleanField,
  IntegerField,
  NumberField,
  OptionalNumberField,
  OptionalRangePairField,
} from "@/features/configurator/fields";
import {
  actionDefaults,
  airborneDefaults,
  damageDefaults,
  energyDefaults,
  leanDefaults,
  progressDefaults,
  timePressureDefaults,
  trackEventDefaults,
} from "@/features/configurator/sections/rewardDefaults";
import type { ManagedRunConfig } from "@/shared/api/contract";

interface ConfigSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: (config: ManagedRunConfig) => void;
}

type RewardDisclosureId =
  | "time"
  | "progress"
  | "airborne"
  | "track"
  | "energy"
  | "actions"
  | "lean"
  | "damage";

type RewardDisclosureState = Record<RewardDisclosureId, boolean>;

export function RewardSection({ config, defaultConfig, setConfig }: ConfigSectionProps) {
  const [openSections, setOpenSections] = useState<RewardDisclosureState>(
    initialRewardDisclosureState,
  );
  const updateReward = (patch: Partial<ManagedRunConfig["reward"]>) => {
    setConfig({ ...config, reward: { ...config.reward, ...patch } });
  };

  const setSectionOpen = (id: RewardDisclosureId, open: boolean) => {
    setOpenSections((current) => ({ ...current, [id]: open }));
  };

  return (
    <div className="reward-accordion-stack">
      <div className="reward-accordion-toolbar">
        <button
          aria-label="Expand all reward sections"
          className="icon-button compact-icon-button"
          title="Expand all"
          type="button"
          onClick={() => setOpenSections(allRewardSectionsOpen(true))}
        >
          <ExpandAllIcon />
        </button>
        <button
          aria-label="Collapse all reward sections"
          className="icon-button compact-icon-button"
          title="Collapse all"
          type="button"
          onClick={() => setOpenSections(allRewardSectionsOpen(false))}
        >
          <CollapseAllIcon />
        </button>
      </div>
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
            help="Progress reward multiplier while on refill surface."
            label="Refill progress multiplier"
            resetValue={defaultConfig.reward.energy_refill_progress_multiplier}
            step="0.1"
            value={config.reward.energy_refill_progress_multiplier}
            onChange={(value) => updateReward({ energy_refill_progress_multiplier: value })}
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
          <IntegerField
            help="Frames after collision during which refill surface reward is suppressed."
            label="Refill collision cooldown"
            resetValue={defaultConfig.reward.energy_refill_collision_cooldown_frames}
            value={config.reward.energy_refill_collision_cooldown_frames}
            onChange={(value) => updateReward({ energy_refill_collision_cooldown_frames: value })}
          />
          <NumberField
            help="Lap bonus for sufficiently refilling energy."
            label="Full refill lap bonus"
            resetValue={defaultConfig.reward.energy_full_refill_lap_bonus}
            step="0.5"
            value={config.reward.energy_full_refill_lap_bonus}
            onChange={(value) => updateReward({ energy_full_refill_lap_bonus: value })}
          />
          <NumberField
            help="Minimum gained fraction required for the full-refill lap bonus."
            label="Full refill min fraction"
            resetValue={defaultConfig.reward.energy_full_refill_min_gain_fraction}
            step="0.05"
            value={config.reward.energy_full_refill_min_gain_fraction}
            onChange={(value) => updateReward({ energy_full_refill_min_gain_fraction: value })}
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
            help="Penalty when the agent underuses gas below the threshold."
            label="Gas underuse penalty"
            resetValue={defaultConfig.reward.gas_underuse_penalty}
            step="0.001"
            value={config.reward.gas_underuse_penalty}
            onChange={(value) => updateReward({ gas_underuse_penalty: value })}
          />
          <NumberField
            help="Gas request threshold below which underuse penalty can apply."
            label="Gas underuse threshold"
            resetValue={defaultConfig.reward.gas_underuse_threshold}
            step="0.05"
            value={config.reward.gas_underuse_threshold}
            onChange={(value) => updateReward({ gas_underuse_threshold: value })}
          />
          <NumberField
            help="Penalty for rapid steering oscillation."
            label="Steer oscillation penalty"
            resetValue={defaultConfig.reward.steer_oscillation_penalty}
            step="0.001"
            value={config.reward.steer_oscillation_penalty}
            onChange={(value) => updateReward({ steer_oscillation_penalty: value })}
          />
          <NumberField
            help="Steering magnitude ignored by oscillation detection."
            label="Steer oscillation deadzone"
            resetValue={defaultConfig.reward.steer_oscillation_deadzone}
            step="0.01"
            value={config.reward.steer_oscillation_deadzone}
            onChange={(value) => updateReward({ steer_oscillation_deadzone: value })}
          />
          <NumberField
            help="Maximum oscillation penalty magnitude before power shaping."
            label="Steer oscillation cap"
            resetValue={defaultConfig.reward.steer_oscillation_cap}
            step="0.1"
            value={config.reward.steer_oscillation_cap}
            onChange={(value) => updateReward({ steer_oscillation_cap: value })}
          />
          <NumberField
            help="Exponent for steering oscillation shaping."
            label="Steer oscillation power"
            resetValue={defaultConfig.reward.steer_oscillation_power}
            step="0.1"
            value={config.reward.steer_oscillation_power}
            onChange={(value) => updateReward({ steer_oscillation_power: value })}
          />
        </div>
      </ConfigDisclosure>

      <ConfigDisclosure
        open={openSections.lean}
        title="Lean and pitch"
        onToggle={(open) => setSectionOpen("lean", open)}
        onReset={() => updateReward(leanDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Per-step penalty for requesting lean."
            label="Lean request penalty"
            resetValue={defaultConfig.reward.lean_request_penalty}
            step="0.001"
            value={config.reward.lean_request_penalty}
            onChange={(value) => updateReward({ lean_request_penalty: value })}
          />
          <NumberField
            help="Additional lean penalty below the configured speed cutoff."
            label="Low-speed lean penalty"
            resetValue={defaultConfig.reward.lean_low_speed_penalty}
            step="0.001"
            value={config.reward.lean_low_speed_penalty}
            onChange={(value) => updateReward({ lean_low_speed_penalty: value })}
          />
          <NumberField
            help="Speed below which the low-speed lean penalty applies."
            label="Low-speed lean cutoff"
            resetValue={defaultConfig.reward.lean_low_speed_penalty_max_speed_kph}
            step="10"
            value={config.reward.lean_low_speed_penalty_max_speed_kph}
            onChange={(value) => updateReward({ lean_low_speed_penalty_max_speed_kph: value })}
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
    </div>
  );
}

function initialRewardDisclosureState(): RewardDisclosureState {
  return allRewardSectionsOpen(true);
}

function allRewardSectionsOpen(open: boolean): RewardDisclosureState {
  return {
    actions: open,
    airborne: open,
    damage: open,
    energy: open,
    lean: open,
    progress: open,
    time: open,
    track: open,
  };
}

function ExpandAllIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="16" viewBox="0 0 20 20" width="16">
      <path
        d="M5 8l5-5 5 5M5 12l5 5 5-5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

function CollapseAllIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="16" viewBox="0 0 20 20" width="16">
      <path
        d="M3 5l5 5-5 5M17 5l-5 5 5 5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}
