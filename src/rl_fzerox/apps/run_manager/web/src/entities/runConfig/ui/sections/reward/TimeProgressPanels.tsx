// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/reward/TimeProgressPanels.tsx

import {
  progressBucketRewardFromDensity,
  progressRewardDensityPerThousand,
  progressSpeedPreviewPoints,
  progressSummaryRows,
} from "@/entities/runConfig/ui/sections/reward/progressDerived";
import { RewardCurvePreview } from "@/entities/runConfig/ui/sections/reward/RewardCurvePreview";
import type { RewardPanelProps } from "@/entities/runConfig/ui/sections/reward/types";
import {
  boundsDefaults,
  progressDefaults,
  timePressureDefaults,
} from "@/entities/runConfig/ui/sections/rewardDefaults";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { BooleanField, IntegerField, NumberField } from "@/shared/ui/configFields";

export function TimeProgressPanels({
  config,
  defaultConfig,
  openSections,
  setSectionOpen,
  updateReward,
}: RewardPanelProps) {
  const progressDensityPerThousand = progressRewardDensityPerThousand(config);
  const progressRows = progressSummaryRows(config);
  const speedPreviewPoints = progressSpeedPreviewPoints(config);

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
            help="Spline distance represented by one progress reward bucket. Set to 0 for continuous proportional progress. Changing this keeps the current reward density per 1k spline units."
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
              help="Skip frontier bucket payout only when the machine is outside track bounds and farther than the configured future-local 3D distance. Skipped buckets still advance the frontier, so they are not repaid on re-entry."
              label="Suspend progress while outside track bounds"
              resetValue={defaultConfig.reward.suspend_progress_while_outside_track_bounds}
              value={config.reward.suspend_progress_while_outside_track_bounds}
              onChange={(value) =>
                updateReward({ suspend_progress_while_outside_track_bounds: value })
              }
            />
          </div>
          <NumberField
            help="Nearest future-local center-spline distance allowed before outside-bounds progress is skipped. Values within this distance still pay progress; farther excursions advance the frontier without reward."
            label="Center spline tolerance"
            resetValue={defaultConfig.reward.progress_track_distance_tolerance}
            step="50"
            value={config.reward.progress_track_distance_tolerance}
            onChange={(value) => updateReward({ progress_track_distance_tolerance: value })}
          />
          <NumberField
            help="Speed at and below which the low-speed progress multiplier applies."
            label="Min speed (kph)"
            resetValue={defaultConfig.reward.progress_speed_min_kph}
            step="10"
            value={config.reward.progress_speed_min_kph}
            onChange={(value) => updateReward({ progress_speed_min_kph: value })}
          />
          <NumberField
            help="Progress multiplier at min speed and below. Set to 0 to make slow progress worthless; keep at 1 for no low-speed shaping."
            label="Min speed multiplier"
            resetValue={defaultConfig.reward.progress_speed_min_multiplier}
            step="0.05"
            value={config.reward.progress_speed_min_multiplier}
            onChange={(value) => updateReward({ progress_speed_min_multiplier: value })}
          />
          <NumberField
            help="Speed where frontier progress uses exactly 1x reward."
            label="Reference speed (kph)"
            resetValue={defaultConfig.reward.progress_speed_reference_kph}
            step="10"
            value={config.reward.progress_speed_reference_kph}
            onChange={(value) => updateReward({ progress_speed_reference_kph: value })}
          />
          <NumberField
            help="Speed where the configured high-speed multiplier is reached."
            label="Max speed (kph)"
            resetValue={defaultConfig.reward.progress_speed_max_kph}
            step="10"
            value={config.reward.progress_speed_max_kph}
            onChange={(value) => updateReward({ progress_speed_max_kph: value })}
          />
          <NumberField
            help="Progress multiplier at max speed and above."
            label="Max speed multiplier"
            resetValue={defaultConfig.reward.progress_speed_max_multiplier}
            step="0.05"
            value={config.reward.progress_speed_max_multiplier}
            onChange={(value) => updateReward({ progress_speed_max_multiplier: value })}
          />
          <NumberField
            help="Speed curve exponent. Higher values punish low speeds until near the reference speed, but pay high-speed bonus earlier and taper toward the max speed."
            label="Speed curve power"
            resetValue={defaultConfig.reward.progress_speed_curve_power}
            step="0.1"
            value={config.reward.progress_speed_curve_power}
            onChange={(value) => updateReward({ progress_speed_curve_power: value })}
          />
        </div>
        <RewardCurvePreview
          className="my-3"
          points={speedPreviewPoints.map((point) => ({
            label: point.label,
            xValue: point.speedKph,
            yValue: point.multiplier,
          }))}
          title="Speed multiplier preview"
          xAxisLabel="speed (kph)"
          yDomain="tight"
        />
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
          <NumberField
            help="Minimum peak height above ground reached during the airborne segment before landing reward can pay. This filters shallow border flicker while preserving real jumps. Set to 0 for the old behavior."
            label="Landing min peak height"
            resetValue={defaultConfig.reward.airborne_landing_min_peak_height}
            step="10"
            value={config.reward.airborne_landing_min_peak_height}
            onChange={(value) => updateReward({ airborne_landing_min_peak_height: value })}
          />
        </div>
      </ConfigDisclosure>
    </>
  );
}
