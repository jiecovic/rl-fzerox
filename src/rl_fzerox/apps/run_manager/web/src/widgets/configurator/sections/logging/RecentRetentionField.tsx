// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/logging/RecentRetentionField.tsx

import type { ManagedRunConfig } from "@/shared/api/contract";
import { FieldShell, RangeReadonly, RangeRow } from "@/shared/ui/Field";
import { IntegerTextInput } from "@/widgets/configurator/fields";
import { FieldLabel } from "@/widgets/configurator/fields/label";
import { resetHandler } from "@/widgets/configurator/fields/reset";
import { Slider } from "@/widgets/configurator/fields/slider";
import {
  recentRetentionFiniteMax,
  recentRetentionSliderMax,
  recentRetentionSliderTicks,
  recentRetentionSliderValue,
  recentRetentionSummary,
  recentRetentionValueFromSlider,
} from "@/widgets/configurator/sections/logging/derived";

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
