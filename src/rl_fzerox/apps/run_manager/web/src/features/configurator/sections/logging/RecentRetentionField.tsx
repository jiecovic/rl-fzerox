import { useEffect, useState } from "react";

import { formatInteger } from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";
import { Slider } from "@/features/configurator/fields/slider";
import {
  recentRetentionFiniteMax,
  recentRetentionSliderMax,
  recentRetentionSliderTicks,
  recentRetentionSliderValue,
  recentRetentionSummary,
  recentRetentionValueFromSlider,
} from "@/features/configurator/sections/logging/derived";
import type { ManagedRunConfig } from "@/shared/api/contract";

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
  const [rawValue, setRawValue] = useState(
    train.recent_checkpoint_limit === null ? "" : formatInteger(train.recent_checkpoint_limit),
  );

  useEffect(() => {
    setRawValue(
      train.recent_checkpoint_limit === null ? "" : formatInteger(train.recent_checkpoint_limit),
    );
  }, [train.recent_checkpoint_limit]);

  function commitManualValue() {
    if (train.recent_checkpoint_limit === null) {
      setRawValue("");
      return;
    }
    const parsed = Number(rawValue.replace(/[,_\s]/g, ""));
    if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > finiteMax) {
      setRawValue(formatInteger(train.recent_checkpoint_limit));
      return;
    }
    onChange(parsed);
    setRawValue(formatInteger(parsed));
  }

  return (
    <div className="field-shell range-field">
      <FieldLabel
        help={help}
        label={label}
        onReset={resetHandler(train.recent_checkpoint_limit, resetValue, onChange)}
      />
      <div className="range-row">
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
          <div className="range-readonly">{recentRetentionSummary(train)}</div>
        ) : (
          <input
            aria-label={label}
            inputMode="numeric"
            spellCheck={false}
            value={rawValue}
            onBlur={commitManualValue}
            onChange={(event) => setRawValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
        )}
      </div>
    </div>
  );
}
