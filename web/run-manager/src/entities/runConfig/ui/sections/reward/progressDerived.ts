// web/run-manager/src/entities/runConfig/ui/sections/reward/progressDerived.ts
import type { ManagedRunConfig } from "@/shared/api/contract";
import {
  formatDecimal,
  formatDurationSeconds,
  formatInteger,
  formatSignedDecimal,
} from "@/shared/ui/format";

export interface ProgressSummaryRow {
  detail: string;
  label: string;
  value: string;
}

export interface ProgressSpeedPreviewPoint {
  label?: string;
  multiplier: number;
  speedKph: number;
}

export function progressRewardDensityPerThousand(config: ManagedRunConfig): number {
  return progressRewardDensityFromValues(
    config.reward.progress_bucket_distance,
    config.reward.progress_bucket_reward,
  );
}

export function progressRewardDensityFromValues(
  bucketDistance: number,
  bucketReward: number,
): number {
  if (bucketDistance <= 0) {
    return roundProgressValue(bucketReward);
  }
  return roundProgressValue((bucketReward / bucketDistance) * 1_000);
}

export function progressBucketRewardFromDensity(
  bucketDistance: number,
  rewardPerThousandUnits: number,
): number {
  if (bucketDistance <= 0) {
    return roundProgressValue(rewardPerThousandUnits);
  }
  return roundProgressValue((rewardPerThousandUnits * bucketDistance) / 1_000);
}

export function progressSummaryRows(config: ManagedRunConfig): readonly ProgressSummaryRow[] {
  const bucketDistance = config.reward.progress_bucket_distance;
  const bucketReward = config.reward.progress_bucket_reward;
  const intervalFrames = Math.max(config.reward.progress_reward_interval_frames, 1);
  const actionRepeat = Math.max(config.action.action_repeat, 1);
  const rewardPerThousandUnits = progressRewardDensityPerThousand(config);
  const minSpeed = config.reward.progress_speed_min_kph;
  const referenceSpeed = config.reward.progress_speed_reference_kph;
  const maxSpeed = config.reward.progress_speed_max_kph;

  return [
    {
      label: bucketDistance <= 0 ? "Continuous progress" : "One bucket",
      detail:
        bucketDistance <= 0
          ? "Pays proportional reward for each newly covered spline unit"
          : `${formatNumber(bucketDistance)} spline units of new frontier progress`,
      value:
        bucketDistance <= 0
          ? `${formatSignedNumber(bucketReward)} per 1k units`
          : `${formatSignedNumber(bucketReward)} reward`,
    },
    {
      label: "Reward density",
      detail:
        bucketDistance <= 0
          ? "Continuous reward density"
          : `${formatSignedNumber(bucketReward)} / ${formatNumber(bucketDistance)} units`,
      value: `${formatSignedNumber(rewardPerThousandUnits)} per 1k units`,
    },
    {
      label: "Payout cadence",
      detail: `${formatInteger(intervalFrames)} internal frames ≈ ${formatDurationSeconds(
        intervalFrames / 60,
        { secondsFractionDigits: 2 },
      )}`,
      value: policyCadenceSummary(intervalFrames, actionRepeat),
    },
    {
      label: "Outside-bounds handling",
      detail: config.reward.suspend_progress_while_outside_track_bounds
        ? `Outside-bounds progress is skipped beyond ${formatNumber(config.reward.progress_track_distance_tolerance)} units from the nearest center spline.`
        : "Outside-bounds movement can still cross and pay buckets.",
      value: config.reward.suspend_progress_while_outside_track_bounds
        ? "Distance gated"
        : "Live outside bounds",
    },
    {
      label: "Speed shaping",
      detail: `${formatNumber(minSpeed)} kph uses the low-speed multiplier; ${formatNumber(referenceSpeed)} kph is neutral; ${formatNumber(maxSpeed)} kph reaches the high-speed multiplier.`,
      value: `${formatNumber(config.reward.progress_speed_min_multiplier)}x → 1x → ${formatNumber(config.reward.progress_speed_max_multiplier)}x`,
    },
  ];
}

export function progressSpeedPreviewPoints(
  config: ManagedRunConfig,
): readonly ProgressSpeedPreviewPoint[] {
  const minKph = Math.max(config.reward.progress_speed_min_kph, 0);
  const referenceKph = Math.max(config.reward.progress_speed_reference_kph, minKph + 1);
  const maxKph = Math.max(config.reward.progress_speed_max_kph, referenceKph + 1);
  const sampleCount = 48;
  const lowMidKph = minKph + (referenceKph - minKph) / 2;
  const highMidKph = referenceKph + (maxKph - referenceKph) / 2;
  const speeds = [
    ...Array.from({ length: sampleCount + 1 }, (_, index) => (maxKph * index) / sampleCount),
    minKph,
    lowMidKph,
    referenceKph,
    highMidKph,
  ]
    .map(roundProgressValue)
    .filter((speedKph, index, values) => values.indexOf(speedKph) === index)
    .sort((left, right) => left - right);
  return speeds.map((speedKph) => ({
    label: speedTickLabel(speedKph, minKph, referenceKph, maxKph),
    speedKph: roundProgressValue(speedKph),
    multiplier: roundProgressValue(progressSpeedMultiplierAtKph(config, speedKph)),
  }));
}

export function progressSpeedMultiplierAtKph(config: ManagedRunConfig, speedKph: number): number {
  const minKph = Math.max(config.reward.progress_speed_min_kph, 0);
  const referenceKph = Math.max(config.reward.progress_speed_reference_kph, minKph + 1e-9);
  const maxKph = Math.max(config.reward.progress_speed_max_kph, referenceKph + 1e-9);
  const curvePower = Math.max(config.reward.progress_speed_curve_power, 1e-9);
  const clampedSpeed = Math.max(speedKph, 0);

  if (clampedSpeed <= minKph) {
    return config.reward.progress_speed_min_multiplier;
  }
  if (clampedSpeed <= referenceKph) {
    const ratio = ((clampedSpeed - minKph) / (referenceKph - minKph)) ** curvePower;
    return (
      config.reward.progress_speed_min_multiplier +
      (1 - config.reward.progress_speed_min_multiplier) * ratio
    );
  }

  const ratio = Math.min((clampedSpeed - referenceKph) / (maxKph - referenceKph), 1);
  const shapedRatio = 1 - (1 - ratio) ** curvePower;
  return 1 + (config.reward.progress_speed_max_multiplier - 1) * shapedRatio;
}

function speedTickLabel(speedKph: number, minKph: number, referenceKph: number, maxKph: number) {
  const lowMidKph = roundProgressValue(minKph + (referenceKph - minKph) / 2);
  const highMidKph = roundProgressValue(referenceKph + (maxKph - referenceKph) / 2);
  if (
    speedKph === 0 ||
    speedKph === minKph ||
    speedKph === lowMidKph ||
    speedKph === referenceKph ||
    speedKph === highMidKph ||
    speedKph === maxKph
  ) {
    return formatNumber(speedKph);
  }
  return undefined;
}

function policyCadenceSummary(intervalFrames: number, actionRepeat: number) {
  if (intervalFrames <= actionRepeat) {
    return `Immediate within env step at repeat x${actionRepeat}`;
  }
  const policySteps = intervalFrames / actionRepeat;
  return `Batched across about ${formatNumber(policySteps)} policy steps at repeat x${actionRepeat}`;
}

function formatNumber(value: number) {
  return formatDecimal(value, { maximumFractionDigits: 2 });
}

function formatSignedNumber(value: number) {
  return formatSignedDecimal(value, { maximumFractionDigits: 2 });
}

function roundProgressValue(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}
