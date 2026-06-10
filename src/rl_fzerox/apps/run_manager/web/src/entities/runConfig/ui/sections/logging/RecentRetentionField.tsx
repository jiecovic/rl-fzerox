// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/logging/RecentRetentionField.tsx

import {
  recentRetentionFiniteMax,
  recentRetentionSliderMax,
  recentRetentionSliderTicks,
  recentRetentionSliderValue,
  recentRetentionSummary,
  recentRetentionValueFromSlider,
} from "@/entities/runConfig/ui/sections/logging/derived";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { IntegerTextInput } from "@/shared/ui/configFields";
import { FieldLabel } from "@/shared/ui/configFields/label";
import { resetHandler } from "@/shared/ui/configFields/reset";
import { Slider } from "@/shared/ui/configFields/slider";
import { FieldShell, RangeReadonly, RangeRow } from "@/shared/ui/Field";

interface RecentRetentionFieldProps {
  help: string;
  label: string;
  train: ManagedRunConfig["train"];
  defaultTrain: ManagedRunConfig["train"];
  onChange: (value: number | null) => void;
}

export function RecentRetentionField({
  help,
  label,
  train,
  defaultTrain,
  onChange,
}: RecentRetentionFieldProps) {
  const finiteMax = recentRetentionFiniteMax();
  const sliderMax = recentRetentionSliderMax();
  const sliderValue = recentRetentionSliderValue(train);
  const resetValue = defaultTrain.recent_checkpoint_limit;

  return (
    <FieldShell>
      <FieldLabel
        help={help}
        label={label}
        onReset={resetHandler(train.recent_checkpoint_limit, resetValue, onChange)}
      />
      <RangeRow>
        <Slider
          ariaLabel={`${label} slider`}
          max={sliderMax}
          min={1}
          step={1}
          ticks={recentRetentionSliderTicks()}
          value={sliderValue}
          valueLabel={
            train.recent_checkpoint_limit === null
              ? "unlimited"
              : String(train.recent_checkpoint_limit)
          }
          onChange={(value) => onChange(recentRetentionValueFromSlider(Math.round(value)))}
        />
        {train.recent_checkpoint_limit === null ? (
          <RangeReadonly>{recentRetentionSummary(train)}</RangeReadonly>
        ) : (
          <IntegerTextInput
            aria-label={label}
            className="h-[34px] indent-0 tabular-nums"
            max={finiteMax}
            min={1}
            value={train.recent_checkpoint_limit}
            onChange={onChange}
          />
        )}
      </RangeRow>
    </FieldShell>
  );
}
