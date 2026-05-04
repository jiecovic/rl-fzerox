import type { ManagedRunConfig } from "@/shared/api/contract";

export interface ProgressSummaryRow {
  detail: string;
  label: string;
  value: string;
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
  return roundProgressValue((bucketReward / bucketDistance) * 1_000);
}

export function progressBucketRewardFromDensity(
  bucketDistance: number,
  rewardPerThousandUnits: number,
): number {
  return roundProgressValue((rewardPerThousandUnits * bucketDistance) / 1_000);
}

export function progressSummaryRows(config: ManagedRunConfig): readonly ProgressSummaryRow[] {
  const bucketDistance = config.reward.progress_bucket_distance;
  const bucketReward = config.reward.progress_bucket_reward;
  const intervalFrames = Math.max(config.reward.progress_reward_interval_frames, 1);
  const actionRepeat = Math.max(config.action.action_repeat, 1);
  const reentryCap = config.reward.outside_bounds_reentry_progress_distance_cap;
  const rewardPerThousandUnits = progressRewardDensityPerThousand(config);

  return [
    {
      label: "One bucket",
      detail: `${formatNumber(bucketDistance)} spline units of new frontier progress`,
      value: `${formatSignedNumber(bucketReward)} reward`,
    },
    {
      label: "Reward density",
      detail: `${formatSignedNumber(bucketReward)} / ${formatNumber(bucketDistance)} units`,
      value: `${formatSignedNumber(rewardPerThousandUnits)} per 1k units`,
    },
    {
      label: "Payout cadence",
      detail: `${formatInteger(intervalFrames)} internal frames ≈ ${formatDuration(intervalFrames / 60)}`,
      value: policyCadenceSummary(intervalFrames, actionRepeat),
    },
    {
      label: "Outside-bounds handling",
      detail: config.reward.suspend_progress_while_outside_track_bounds
        ? "Bucket payout pauses while outside track bounds."
        : "Outside-bounds movement can still cross and pay buckets.",
      value: config.reward.suspend_progress_while_outside_track_bounds
        ? "Paused outside bounds"
        : "Live outside bounds",
    },
    {
      label: "Re-entry cap",
      detail:
        reentryCap === null
          ? "Deferred off-track recovery distance is uncapped."
          : `${formatNumber(reentryCap)} units recoverable after re-entry`,
      value:
        reentryCap === null
          ? "Unlimited"
          : `${formatNumber(reentryCap / bucketDistance)} buckets ≈ ${formatSignedNumber((reentryCap / bucketDistance) * bucketReward)} reward`,
    },
  ];
}

function policyCadenceSummary(intervalFrames: number, actionRepeat: number) {
  if (intervalFrames <= actionRepeat) {
    return `Immediate within env step at repeat x${actionRepeat}`;
  }
  const policySteps = intervalFrames / actionRepeat;
  return `Batched across about ${formatNumber(policySteps)} policy steps at repeat x${actionRepeat}`;
}

function formatInteger(value: number) {
  return value.toLocaleString();
}

function formatNumber(value: number) {
  if (Number.isInteger(value)) {
    return formatInteger(value);
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatSignedNumber(value: number) {
  const formatted = formatNumber(value);
  return value > 0 ? `+${formatted}` : formatted;
}

function formatDuration(value: number) {
  if (value < 60) {
    return `${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} s`;
  }
  const wholeMinutes = Math.floor(value / 60);
  const remainingSeconds = value - wholeMinutes * 60;
  if (Math.abs(remainingSeconds - Math.round(remainingSeconds)) < 1e-9) {
    return `${wholeMinutes}m ${Math.round(remainingSeconds)}s`;
  }
  return `${wholeMinutes}m ${remainingSeconds.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}s`;
}

function roundProgressValue(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}
