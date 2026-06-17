// web/run-manager/src/entities/runConfig/ui/sections/reward/TrackActionPanels.tsx

import { boostRequestRewardPreviewPoints } from "@/entities/runConfig/ui/sections/reward/boostDerived";
import { RewardCurvePreview } from "@/entities/runConfig/ui/sections/reward/RewardCurvePreview";
import type { RewardPanelProps } from "@/entities/runConfig/ui/sections/reward/types";
import {
  actionDefaults,
  energyDefaults,
  trackEventDefaults,
} from "@/entities/runConfig/ui/sections/rewardDefaults";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import {
  BooleanField,
  IntegerField,
  NumberField,
  RangeNumberField,
  SelectField,
} from "@/shared/ui/configFields";

export function TrackActionPanels({
  config,
  defaultConfig,
  openSections,
  setSectionOpen,
  updateAction,
  updateReward,
}: RewardPanelProps) {
  const boostRewardPreviewPoints = boostRequestRewardPreviewPoints(config);

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
            help="Progress reward multiplier at last place. Single-racer modes stay at 1x."
            label="Position progress min"
            resetValue={defaultConfig.reward.position_progress_min_multiplier}
            step="0.01"
            value={config.reward.position_progress_min_multiplier}
            onChange={(value) => updateReward({ position_progress_min_multiplier: value })}
          />
          <NumberField
            help="Progress reward multiplier at first place. Intermediate positions are linearly interpolated."
            label="Position progress max"
            resetValue={defaultConfig.reward.position_progress_max_multiplier}
            step="0.01"
            value={config.reward.position_progress_max_multiplier}
            onChange={(value) => updateReward({ position_progress_max_multiplier: value })}
          />
          <NumberField
            help="Reward for each newly gained KO star in GP race. This is ignored outside GP race."
            label="KO star reward"
            resetValue={defaultConfig.reward.ko_star_reward}
            step="0.1"
            value={config.reward.ko_star_reward}
            onChange={(value) => updateReward({ ko_star_reward: value })}
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
        onReset={() => {
          updateReward(actionDefaults(defaultConfig.reward));
          updateAction({ pitch_deadzone: defaultConfig.action.pitch_deadzone });
        }}
      >
        <div className="grid gap-4">
          <section className={rewardActionGroupClass}>
            <h4 className={rewardActionTitleClass}>Manual boost</h4>
            <div className={rewardActionBoostLayoutClass}>
              <div className={rewardActionFieldsClass}>
                <NumberField
                  help="Reward added when the policy requests manual boost. With energy shaping enabled, this is the high-energy reward."
                  label="Boost request reward"
                  resetValue={defaultConfig.reward.manual_boost_reward}
                  step="0.01"
                  value={config.reward.manual_boost_reward}
                  onChange={(value) => updateReward({ manual_boost_reward: value })}
                />
                <BooleanField
                  help="Scale the boost request reward by current energy. Off keeps the request reward constant."
                  label="Energy-shaped boost reward"
                  resetValue={defaultConfig.reward.manual_boost_reward_energy_shaping}
                  value={config.reward.manual_boost_reward_energy_shaping}
                  onChange={(value) => updateReward({ manual_boost_reward_energy_shaping: value })}
                />
                {config.reward.manual_boost_reward_energy_shaping ? (
                  <>
                    <NumberField
                      help="Reward used at and below the unsafe-energy threshold. Negative values make low-energy boost requests a penalty."
                      label="Low energy reward"
                      resetValue={defaultConfig.reward.manual_boost_reward_min_energy_value}
                      step="0.1"
                      value={config.reward.manual_boost_reward_min_energy_value}
                      onChange={(value) =>
                        updateReward({ manual_boost_reward_min_energy_value: value })
                      }
                    />
                    <NumberField
                      help="Energy fraction at and below which boost request reward stays at the minimum multiplier."
                      label="Unsafe energy below"
                      resetValue={defaultConfig.reward.manual_boost_reward_min_energy_fraction}
                      step="0.05"
                      value={config.reward.manual_boost_reward_min_energy_fraction}
                      onChange={(value) =>
                        updateReward({ manual_boost_reward_min_energy_fraction: value })
                      }
                    />
                    <NumberField
                      help="Energy fraction at and above which boost request reward reaches the full configured value."
                      label="Full reward above"
                      resetValue={defaultConfig.reward.manual_boost_reward_full_energy_fraction}
                      step="0.05"
                      value={config.reward.manual_boost_reward_full_energy_fraction}
                      onChange={(value) =>
                        updateReward({ manual_boost_reward_full_energy_fraction: value })
                      }
                    />
                    <SelectField
                      help="Curve used only between the unsafe and full-reward energy thresholds."
                      label="Energy reward curve"
                      optionLabels={{ linear: "Linear", smoothstep: "Smoothstep" }}
                      options={["linear", "smoothstep"]}
                      resetValue={defaultConfig.reward.manual_boost_reward_energy_curve}
                      value={config.reward.manual_boost_reward_energy_curve}
                      onChange={(value) =>
                        updateReward({ manual_boost_reward_energy_curve: value })
                      }
                    />
                  </>
                ) : null}
              </div>
              {config.reward.manual_boost_reward_energy_shaping ? (
                <RewardCurvePreview
                  points={boostRewardPreviewPoints}
                  title="Boost request reward preview"
                  xAxisLabel="energy (%)"
                  ySuffix=""
                />
              ) : null}
            </div>
          </section>

          <section className={rewardActionGroupClass}>
            <h4 className={rewardActionTitleClass}>Boost pads</h4>
            <div className={rewardActionFieldsClass}>
              <NumberField
                help="Reward for entering a detected boost pad when manual boost cannot be used."
                label="Cannot boost"
                resetValue={defaultConfig.reward.boost_pad_reward_cannot_boost}
                step="0.5"
                value={config.reward.boost_pad_reward_cannot_boost}
                onChange={(value) => updateReward({ boost_pad_reward_cannot_boost: value })}
              />
              <NumberField
                help="Reward for entering a detected boost pad when manual boost can be used."
                label="Can boost"
                resetValue={defaultConfig.reward.boost_pad_reward_can_boost}
                step="0.5"
                value={config.reward.boost_pad_reward_can_boost}
                onChange={(value) => updateReward({ boost_pad_reward_can_boost: value })}
              />
              <NumberField
                help="Progress window used to make boost-pad rewards one-way and non-farmable."
                label="Boost pad progress window"
                resetValue={defaultConfig.reward.boost_pad_reward_progress_window}
                step="50"
                value={config.reward.boost_pad_reward_progress_window}
                onChange={(value) => updateReward({ boost_pad_reward_progress_window: value })}
              />
            </div>
          </section>

          <section className={rewardActionGroupClass}>
            <h4 className={rewardActionTitleClass}>Action penalties</h4>
            <div className={rewardActionFieldsClass}>
              <NumberField
                help="Per-step penalty for requesting air brake."
                label="Air brake request penalty"
                resetValue={defaultConfig.reward.air_brake_request_penalty}
                step="0.001"
                value={config.reward.air_brake_request_penalty}
                onChange={(value) => updateReward({ air_brake_request_penalty: value })}
              />
              <NumberField
                help="One-time penalty when a spin macro is requested."
                label="Spin request penalty"
                resetValue={defaultConfig.reward.spin_request_penalty}
                step="0.001"
                value={config.reward.spin_request_penalty}
                onChange={(value) => updateReward({ spin_request_penalty: value })}
              />
              <NumberField
                help="Per-frame penalty while lean remains active after startup."
                label="Lean hold penalty"
                resetValue={defaultConfig.reward.lean_request_penalty}
                step="0.001"
                value={config.reward.lean_request_penalty}
                onChange={(value) => updateReward({ lean_request_penalty: value })}
              />
              <NumberField
                help="One-env-step startup penalty when lean changes from idle to active, scaled by action-repeat native frames."
                label="Lean activation penalty"
                resetValue={defaultConfig.reward.lean_activation_penalty}
                step="0.001"
                value={config.reward.lean_activation_penalty}
                onChange={(value) => updateReward({ lean_activation_penalty: value })}
              />
              <NumberField
                help="Small penalty for grounded pitch requests above the penalty threshold. This does not clamp controller pitch."
                label="Grounded pitch penalty"
                resetValue={defaultConfig.reward.grounded_pitch_penalty}
                step="0.001"
                value={config.reward.grounded_pitch_penalty}
                onChange={(value) => updateReward({ grounded_pitch_penalty: value })}
              />
              <RangeNumberField
                help="Threshold used only by grounded pitch penalty. It does not alter the pitch sent to the controller."
                label="Pitch penalty threshold"
                max={0.5}
                min={0}
                rangeStep={0.01}
                resetValue={defaultConfig.action.pitch_deadzone}
                ticks={[
                  { label: "0", value: 0 },
                  { label: "0.1", value: 0.1 },
                  { label: "0.25", value: 0.25 },
                  { label: "0.5", value: 0.5 },
                ]}
                value={config.action.pitch_deadzone}
                onChange={(value) => updateAction({ pitch_deadzone: value })}
              />
            </div>
          </section>
        </div>
      </ConfigDisclosure>
    </>
  );
}

const rewardActionGroupClass =
  "grid min-w-0 gap-2.5 border-app-border border-t pt-3.5 first:border-t-0 first:pt-0";
const rewardActionTitleClass = "m-0 text-[13px] font-bold text-app-text";
const rewardActionFieldsClass =
  "grid items-end gap-x-4 gap-y-3 grid-cols-[repeat(auto-fill,minmax(190px,240px))]";
const rewardActionBoostLayoutClass =
  "grid items-start gap-x-5 gap-y-4 grid-cols-[minmax(460px,0.9fr)_minmax(560px,1.1fr)] max-[1260px]:grid-cols-1";
