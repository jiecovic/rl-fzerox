// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/policy/FeatureDimField.tsx
import { useEffect, useState } from "react";
import { SegmentedChoiceStrip } from "@/features/configurator/fields";
import { formatInteger } from "@/features/configurator/fields/format";
import { FieldLabel } from "@/features/configurator/fields/label";
import { resetHandler } from "@/features/configurator/fields/reset";

type FeatureDim = "auto" | number;

interface FeatureDimFieldProps {
  help: string;
  label: string;
  resetValue?: FeatureDim;
  value: FeatureDim;
  onChange: (value: FeatureDim) => void;
}

const DEFAULT_CUSTOM_FEATURE_DIM = 512;

export function FeatureDimField({
  help,
  label,
  resetValue,
  value,
  onChange,
}: FeatureDimFieldProps) {
  const numericValue = typeof value === "number" ? value : defaultCustomValue(resetValue);
  const [rawValue, setRawValue] = useState(formatInteger(numericValue));

  useEffect(() => {
    if (typeof value === "number") {
      setRawValue(formatInteger(value));
    }
  }, [value]);

  function commitCustomValue() {
    const parsed = Number(rawValue.replace(/[,_\s]/g, ""));
    if (!Number.isSafeInteger(parsed) || parsed <= 0) {
      setRawValue(formatInteger(numericValue));
      return;
    }
    onChange(parsed);
    setRawValue(formatInteger(parsed));
  }

  function enableAuto() {
    onChange("auto");
  }

  function enableCustom() {
    if (typeof value === "number") {
      return;
    }
    const nextValue = defaultCustomValue(resetValue);
    onChange(nextValue);
    setRawValue(formatInteger(nextValue));
  }

  return (
    <div className="field-shell">
      <FieldLabel help={help} label={label} onReset={resetHandler(value, resetValue, onChange)} />
      <div className="feature-dim-field">
        <SegmentedChoiceStrip
          ariaLabel={`${label} mode`}
          options={[
            {
              active: value === "auto",
              key: "auto",
              label: "Auto",
              onClick: enableAuto,
            },
            {
              active: typeof value === "number",
              key: "custom",
              label: "Custom",
              onClick: enableCustom,
            },
          ]}
        />
        {typeof value === "number" ? (
          <input
            aria-label={label}
            className="feature-dim-input"
            inputMode="numeric"
            spellCheck={false}
            value={rawValue}
            onBlur={commitCustomValue}
            onChange={(event) => setRawValue(event.target.value)}
          />
        ) : null}
      </div>
    </div>
  );
}

function defaultCustomValue(resetValue: FeatureDim | undefined) {
  return typeof resetValue === "number" ? resetValue : DEFAULT_CUSTOM_FEATURE_DIM;
}
