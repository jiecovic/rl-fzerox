// web/run-manager/src/entities/runConfig/ui/sections/vehicle/AdaptiveEngineControls.tsx
import { useMemo } from "react";
import type { ManagedRunConfig } from "@/shared/api/contract";
import {
  bucketSideCountFromRawValues,
  centeredEngineBuckets,
  engineSliderStepPercentLabel,
} from "@/shared/domain/engineBuckets";
import { Button } from "@/shared/ui/Button";
import {
  IntegerField,
  NumberField,
  RangeNumberField,
  SegmentedChoiceStrip,
} from "@/shared/ui/configFields";

interface AdaptiveEngineControlsProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  onChange: (patch: Partial<ManagedRunConfig["vehicle"]>) => void;
}

export function AdaptiveEngineControls({
  config,
  defaultConfig,
  onChange,
}: AdaptiveEngineControlsProps) {
  const vehicle = config.vehicle;
  const defaultVehicle = defaultConfig.vehicle;
  const isBanditBackend = vehicle.adaptive_engine_tuner_backend === "bandit";
  const experimentalBackendLabel = experimentalEngineTunerBackendLabel(
    vehicle.adaptive_engine_tuner_backend,
  );
  const banditBucketSideCount =
    bucketSideCountFromRawValues(vehicle.adaptive_engine_bandit_bucket_raw_values) ||
    bucketSideCountFromRawValues(defaultVehicle.adaptive_engine_bandit_bucket_raw_values);
  const banditBuckets = useMemo(
    () => vehicle.adaptive_engine_bandit_bucket_raw_values,
    [vehicle.adaptive_engine_bandit_bucket_raw_values],
  );

  function setBanditBucketSideCount(sideCount: number) {
    const adaptive_engine_bandit_bucket_raw_values = centeredEngineBuckets({
      sideCount,
      minimum: vehicle.engine_setting_min_raw_value,
      maximum: vehicle.engine_setting_max_raw_value,
    });
    if (adaptive_engine_bandit_bucket_raw_values.length > 0) {
      onChange({ adaptive_engine_bandit_bucket_raw_values });
    }
  }

  return (
    <div className="grid gap-3 border-t border-app-border pt-3">
      <div className="grid gap-1">
        <strong className="text-[13px] text-app-text">Adaptive engine tuner</strong>
        <small className="m-0 text-xs leading-snug text-app-muted">
          Learns reset-time engine settings from the selected score while preserving uniform
          exploration.
        </small>
      </div>
      <div className="grid gap-2">
        <strong className="text-[13px] text-app-text">Tuner backend</strong>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex min-h-9 items-center border border-app-border bg-app-surface-muted px-3 text-sm font-semibold text-app-text">
            Bandit
          </span>
          {experimentalBackendLabel !== null ? (
            <>
              <span className="inline-flex min-h-9 items-center border border-amber-400/45 bg-amber-400/10 px-3 text-sm font-semibold text-amber-200">
                legacy {experimentalBackendLabel}
              </span>
              <Button onClick={() => onChange({ adaptive_engine_tuner_backend: "bandit" })}>
                Use bandit
              </Button>
            </>
          ) : null}
        </div>
        <small className="m-0 text-xs leading-snug text-app-muted">
          Bandit is the maintained tuner path. GP and MLP remain loadable as experimental legacy
          backends for old configs and future work, but they are not exposed for new tuning.
        </small>
      </div>
      {isBanditBackend ? (
        <div className="grid gap-2">
          <strong className="text-[13px] text-app-text">Bandit objective</strong>
          <SegmentedChoiceStrip
            ariaLabel="Bandit engine tuner objective"
            options={[
              {
                active: vehicle.adaptive_engine_tuner_objective === "finish_time",
                key: "finish_time",
                label: "Fastest",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "finish_time" }),
              },
              {
                active: vehicle.adaptive_engine_tuner_objective === "safe_finish_time",
                key: "safe_finish_time",
                label: "Safe fastest",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "safe_finish_time" }),
              },
              {
                active: vehicle.adaptive_engine_tuner_objective === "finish_rate",
                key: "finish_rate",
                label: "Safest",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "finish_rate" }),
              },
            ]}
          />
          <small className="m-0 text-xs leading-snug text-app-muted">
            Fastest ranks successful finish times. Safe fastest requires a minimum finish rate, then
            ranks by finish time. Safest ranks only finish rate. Return and completion stay recorded
            as diagnostics.
          </small>
        </div>
      ) : null}
      <div className="grid gap-3 lg:grid-cols-3">
        {isBanditBackend ? (
          <NumberField
            help="Generates explicit centered engine buckets. 5 means five values below 50%, 50%, and five values above 50%."
            label="Buckets per side"
            resetValue={bucketSideCountFromRawValues(
              defaultVehicle.adaptive_engine_bandit_bucket_raw_values,
            )}
            step="1"
            value={banditBucketSideCount}
            onChange={setBanditBucketSideCount}
          />
        ) : null}
        {isBanditBackend && vehicle.adaptive_engine_tuner_objective === "safe_finish_time" ? (
          <RangeNumberField
            help="Minimum observed finish rate required before a bucket is ranked by finish time."
            label="Safe finish rate"
            max={1}
            min={0}
            numberStep="0.01"
            rangeStep={0.01}
            resetValue={defaultVehicle.adaptive_engine_safe_finish_rate_threshold}
            value={vehicle.adaptive_engine_safe_finish_rate_threshold}
            onChange={(adaptive_engine_safe_finish_rate_threshold) =>
              onChange({ adaptive_engine_safe_finish_rate_threshold })
            }
          />
        ) : null}
        {isBanditBackend &&
        (vehicle.adaptive_engine_tuner_objective === "safe_finish_time" ||
          vehicle.adaptive_engine_tuner_objective === "finish_rate") ? (
          <IntegerField
            help="Episodes per engine bucket before finish-rate sampling uses the observed finish/fail ratio."
            label="Rate warmup"
            max={4096}
            min={0}
            resetValue={defaultVehicle.adaptive_engine_min_finish_rate_observations}
            value={vehicle.adaptive_engine_min_finish_rate_observations}
            onChange={(adaptive_engine_min_finish_rate_observations) =>
              onChange({ adaptive_engine_min_finish_rate_observations })
            }
          />
        ) : null}
        {isBanditBackend ? (
          <RangeNumberField
            help="Probability of taking a uniformly random engine value."
            label="Uniform exploration"
            max={1}
            min={0}
            numberStep="0.01"
            rangeStep={0.01}
            resetValue={defaultVehicle.adaptive_engine_uniform_exploration}
            value={vehicle.adaptive_engine_uniform_exploration}
            onChange={(adaptive_engine_uniform_exploration) =>
              onChange({ adaptive_engine_uniform_exploration })
            }
          />
        ) : null}
      </div>
      {isBanditBackend ? <BanditBucketPreview buckets={banditBuckets} /> : null}
      {!isBanditBackend ? (
        <p className="m-0 border border-amber-400/45 bg-amber-400/10 p-3 text-xs leading-normal text-amber-200">
          This config uses the experimental {experimentalBackendLabel} backend. It remains accepted
          for compatibility, but new run-manager tuning should use Bandit until the model-backed
          tuners are redesigned around the current bandit telemetry.
        </p>
      ) : null}
    </div>
  );
}

function experimentalEngineTunerBackendLabel(
  backend: ManagedRunConfig["vehicle"]["adaptive_engine_tuner_backend"],
) {
  if (backend === "gaussian_process") {
    return "GP";
  }
  if (backend === "mlp_ensemble") {
    return "MLP";
  }
  return null;
}

function BanditBucketPreview({ buckets }: { buckets: readonly number[] }) {
  const hasBuckets = buckets.length > 0;
  return (
    <div className="grid gap-2 border border-app-border bg-app-surface-muted p-3">
      <div className="flex items-center justify-between gap-3">
        <strong className="text-[13px] text-app-text">Bandit buckets</strong>
        <span className="text-xs tabular-nums text-app-muted">{buckets.length} values</span>
      </div>
      {hasBuckets ? (
        <div className="flex flex-wrap gap-1.5">
          {buckets.map((bucket) => (
            <span
              className="min-w-11 border border-app-border bg-app-surface px-2 py-1 text-center text-xs font-semibold tabular-nums text-app-text"
              key={bucket}
            >
              {engineSliderStepPercentLabel(bucket)}
            </span>
          ))}
        </div>
      ) : (
        <span className="text-xs text-app-danger">
          No centered bucket falls inside the selected engine range.
        </span>
      )}
    </div>
  );
}
