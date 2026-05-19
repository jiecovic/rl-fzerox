// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/TimeProgressPanels.tsx
import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import {
  BooleanField,
  IntegerField,
  NumberField,
  OptionalNumberField,
} from "@/features/configurator/fields";
import {
  boundsDefaults,
  progressDefaults,
  timePressureDefaults,
} from "@/features/configurator/sections/rewardDefaults";

import {
  progressBucketRewardFromDensity,
  progressRewardDensityPerThousand,
  progressSummaryRows,
} from "./progressDerived";
import type { RewardPanelProps } from "./types";

export function TimeProgressPanels({
  config,
  defaultConfig,
  openSections,
  setSectionOpen,
  updateReward,
}: RewardPanelProps) {
  const progressDensityPerThousand = progressRewardDensityPerThousand(config);
  const progressRows = progressSummaryRows(config);

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
            help="Additional per-frame time pressure. Discounting already pushes faster progress, so this is best treated as an extra experiment rather than a canonical default."
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
            help="Extra multiplier for the smooth slow-speed ramp. The added term is time_penalty_per_frame × slow_speed_multiplier × ((max(start_kph - speed_kph, 0) / start_kph) ^ power)."
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
            help="Exponent in the slow-speed term ((max(start_kph - speed_kph, 0) / start_kph) ^ power). Higher values keep the penalty flatter until speed drops further below the threshold."
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
            help="Spline distance represented by one progress reward bucket. Changing this keeps the current reward density per 1k spline units and derives a new per-bucket reward automatically."
            label="Bucket distance"
            resetValue={defaultConfig.reward.progress_bucket_distance}
            step="5"
            value={config.reward.progress_bucket_distance}
            onChange={(value) =>
              updateReward({
                progress_bucket_distance: value,
                progress_bucket_reward: progressBucketRewardFromDensity(
                  value,
                  progressDensityPerThousand,
                ),
              })
            }
          />
          <NumberField
            help="Overall frontier reward density. The per-bucket reward is derived as (reward per 1k units × bucket distance / 1000), so you can change granularity without changing total lap shaping."
            label="Reward / 1k spline units"
            resetValue={progressRewardDensityPerThousand(defaultConfig)}
            step="0.01"
            value={progressDensityPerThousand}
            onChange={(value) =>
              updateReward({
                progress_bucket_reward: progressBucketRewardFromDensity(
                  config.reward.progress_bucket_distance,
                  value,
                ),
              })
            }
          />
          <IntegerField
            help="Native-frame payout cooldown for frontier buckets. Crossed buckets are accumulated and paid together once this many internal frames have elapsed, or earlier if the episode ends."
            label="Progress interval frames"
            min={1}
            resetValue={defaultConfig.reward.progress_reward_interval_frames}
            value={config.reward.progress_reward_interval_frames}
            onChange={(value) => updateReward({ progress_reward_interval_frames: value })}
          />
          <div className="reward-wide-field">
            <BooleanField
              help="Pause frontier bucket progress whenever the machine is outside track bounds. Airborne movement that stays over the track still counts normally. If progress gets deferred out of bounds, payout stays paused until the machine is back inside bounds and grounded again, then resumes from the visible grounded re-entry position rather than the off-track trajectory."
              label="Suspend progress while outside track bounds"
              resetValue={defaultConfig.reward.suspend_progress_while_outside_track_bounds}
              value={config.reward.suspend_progress_while_outside_track_bounds}
              onChange={(value) =>
                updateReward({ suspend_progress_while_outside_track_bounds: value })
              }
            />
          </div>
          <OptionalNumberField
            defaultValue={10_000}
            enabledLabel="Cap"
            help="Caps deferred outside-track re-entry progress distance. Drag fully right for unlimited."
            label="Re-entry distance cap"
            max={50_000}
            min={0}
            nullLabel="Unlimited"
            resetValue={defaultConfig.reward.outside_bounds_reentry_progress_distance_cap}
            sliderNullPosition="max"
            sliderNullTickLabel="∞"
            step="100"
            value={config.reward.outside_bounds_reentry_progress_distance_cap}
            onChange={(value) =>
              updateReward({ outside_bounds_reentry_progress_distance_cap: value })
            }
          />
        </div>
        <table className="derived-table">
          <tbody>
            {progressRows.map((row) => (
              <tr key={row.label}>
                <th>{row.label}</th>
                <td>{row.detail}</td>
                <td>{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ConfigDisclosure>

      <ConfigDisclosure
        open={openSections.airborne}
        title="Outside bounds"
        onToggle={(open) => setSectionOpen("airborne", open)}
        onReset={() => updateReward(boundsDefaults(defaultConfig.reward))}
      >
        <div className="config-field-grid">
          <NumberField
            help="Positive shaping weight for outside-track recovery. Once armed, this adds weight × the decrease in future-local 3D segment distance in raw game units. The final reward is clamped by the recovery cap. Set weight to 0 to disable."
            label="Outside-track recovery weight"
            resetValue={defaultConfig.reward.outside_track_recovery_reward}
            step="0.0001"
            value={config.reward.outside_track_recovery_reward}
            onChange={(value) => updateReward({ outside_track_recovery_reward: value })}
          />
          <NumberField
            help="Maximum absolute reward this recovery shaping term can add per env step after applying the recovery weight. Default 0.1 means this term is clamped to [-0.1, +0.1]."
            label="Recovery reward cap"
            resetValue={defaultConfig.reward.outside_track_recovery_reward_cap}
            step="0.01"
            value={config.reward.outside_track_recovery_reward_cap}
            onChange={(value) => updateReward({ outside_track_recovery_reward_cap: value })}
          />
          <IntegerField
            help="Keep outside-track recovery shaping off until the current outside-track excursion has accumulated at least this many airborne internal frames. Short off-track jumps stay ungated if they land earlier. Set to 0 to arm shaping from the first airborne frame."
            label="Recovery airborne grace frames"
            min={0}
            resetValue={defaultConfig.reward.outside_track_recovery_airborne_grace_frames}
            value={config.reward.outside_track_recovery_airborne_grace_frames}
            onChange={(value) =>
              updateReward({ outside_track_recovery_airborne_grace_frames: value })
            }
          />
          <NumberField
            help="Reward paid when a jump lands after meeting the landing airborne grace."
            label="Landing reward"
            resetValue={defaultConfig.reward.airborne_landing_reward}
            step="0.5"
            value={config.reward.airborne_landing_reward}
            onChange={(value) => updateReward({ airborne_landing_reward: value })}
          />
          <IntegerField
            help="Require at least this many airborne internal frames in the jump before a landing reward pays out. Set to 0 to reward any landing."
            label="Landing airborne grace frames"
            min={0}
            resetValue={defaultConfig.reward.airborne_landing_grace_frames}
            value={config.reward.airborne_landing_grace_frames}
            onChange={(value) => updateReward({ airborne_landing_grace_frames: value })}
          />
        </div>
      </ConfigDisclosure>
    </>
  );
}
