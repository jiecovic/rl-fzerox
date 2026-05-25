// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/TimeProgressPanels.tsx
import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { BooleanField, IntegerField, NumberField } from "@/features/configurator/fields";
import {
  type ProgressSpeedPreviewPoint,
  progressBucketRewardFromDensity,
  progressRewardDensityPerThousand,
  progressSpeedPreviewPoints,
  progressSummaryRows,
} from "@/features/configurator/sections/reward/progressDerived";
import type { RewardPanelProps } from "@/features/configurator/sections/reward/types";
import {
  boundsDefaults,
  progressDefaults,
  timePressureDefaults,
} from "@/features/configurator/sections/rewardDefaults";

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
        <ProgressSpeedPreview points={speedPreviewPoints} />
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

function ProgressSpeedPreview({ points }: { points: readonly ProgressSpeedPreviewPoint[] }) {
  if (points.length === 0) {
    return null;
  }
  const maxSpeed = Math.max(...points.map((point) => point.speedKph), 1);
  const multipliers = points.map((point) => point.multiplier);
  const minMultiplier = Math.min(...multipliers);
  const maxMultiplier = Math.max(...multipliers);
  const axisMinMultiplier = Math.min(minMultiplier, 0);
  const axisMaxMultiplier = Math.max(maxMultiplier, 1);
  const multiplierSpan = Math.max(axisMaxMultiplier - axisMinMultiplier, 1e-9);
  const viewBox = {
    height: 190,
    width: 900,
  };
  const plot = {
    bottom: 138,
    left: 70,
    right: 830,
    top: 24,
  };
  const plotWidth = plot.right - plot.left;
  const plotHeight = plot.bottom - plot.top;
  const plottedPoints = points.map((point) => ({
    ...point,
    x: plot.left + (point.speedKph / maxSpeed) * plotWidth,
    y: plot.bottom - ((point.multiplier - axisMinMultiplier) / multiplierSpan) * plotHeight,
  }));
  const svgPoints = plottedPoints
    .map((point) => {
      return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
    })
    .join(" ");
  const plottedTickPoints = plottedPoints.filter((point) => point.label !== undefined);
  const yTickPoints = visibleYTickPoints(plottedTickPoints, {
    maxMultiplier,
    minMultiplier,
  });

  return (
    <div className="progress-speed-preview">
      <div className="progress-speed-preview__header">
        <span>Speed multiplier preview</span>
        <span>
          {formatPreviewNumber(minMultiplier)}x - {formatPreviewNumber(maxMultiplier)}x
        </span>
      </div>
      <svg aria-hidden="true" viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}>
        {plottedTickPoints.map((point) => (
          <g key={`guide-${point.speedKph}`}>
            <line
              className="progress-speed-preview__guide"
              x1={point.x}
              x2={point.x}
              y1={point.y}
              y2={plot.bottom}
            />
            <line
              className="progress-speed-preview__guide"
              x1={plot.left}
              x2={point.x}
              y1={point.y}
              y2={point.y}
            />
          </g>
        ))}
        <line x1={plot.left} x2={plot.right} y1={plot.bottom} y2={plot.bottom} />
        <line x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.bottom} />
        {yTickPoints.map((point) => (
          <g key={`multiplier-${point.multiplier}`}>
            <line
              className="progress-speed-preview__tick"
              x1={plot.left - 7}
              x2={plot.left}
              y1={point.y}
              y2={point.y}
            />
            <text textAnchor="end" x={plot.left - 12} y={point.y + 3}>
              {formatPreviewNumber(point.multiplier)}x
            </text>
          </g>
        ))}
        {plottedTickPoints.map((point, index) => (
          <g key={`tick-${point.speedKph}`}>
            <line
              className="progress-speed-preview__tick"
              x1={point.x}
              x2={point.x}
              y1={plot.bottom}
              y2={plot.bottom + 6}
            />
            <text
              textAnchor={axisTickAnchor(index, plottedTickPoints.length)}
              x={point.x}
              y={plot.bottom + 28}
            >
              {point.label}
            </text>
          </g>
        ))}
        <text
          className="progress-speed-preview__axis-label"
          textAnchor="middle"
          x={(plot.left + plot.right) / 2}
          y={viewBox.height - 8}
        >
          speed (kph)
        </text>
        <polyline points={svgPoints} />
        {plottedTickPoints.map((point) => (
          <circle key={`${point.speedKph}-${point.multiplier}`} cx={point.x} cy={point.y} r="3" />
        ))}
      </svg>
    </div>
  );
}

interface PlottedProgressSpeedPreviewPoint extends ProgressSpeedPreviewPoint {
  x: number;
  y: number;
}

function visibleYTickPoints(
  points: readonly PlottedProgressSpeedPreviewPoint[],
  limits: { maxMultiplier: number; minMultiplier: number },
) {
  const minSpacing = 15;
  const uniquePoints = points.filter((point, index, values) => {
    return (
      values.findIndex((otherPoint) => {
        return Math.abs(otherPoint.multiplier - point.multiplier) < 1e-9;
      }) === index
    );
  });
  const selectedPoints: PlottedProgressSpeedPreviewPoint[] = [];
  for (const point of [...uniquePoints].sort((left, right) => {
    return yTickPriority(left, limits) - yTickPriority(right, limits);
  })) {
    if (
      selectedPoints.every((selectedPoint) => Math.abs(selectedPoint.y - point.y) >= minSpacing)
    ) {
      selectedPoints.push(point);
    }
  }
  return selectedPoints.sort((left, right) => left.y - right.y);
}

function yTickPriority(
  point: PlottedProgressSpeedPreviewPoint,
  limits: { maxMultiplier: number; minMultiplier: number },
) {
  if (
    Math.abs(point.multiplier - limits.minMultiplier) < 1e-9 ||
    Math.abs(point.multiplier - limits.maxMultiplier) < 1e-9
  ) {
    return 0;
  }
  if (Math.abs(point.multiplier - 1) < 1e-9) {
    return 1;
  }
  return 2;
}

function axisTickAnchor(index: number, count: number) {
  if (index === 0) {
    return "start";
  }
  if (index === count - 1) {
    return "end";
  }
  return "middle";
}

function formatPreviewNumber(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
