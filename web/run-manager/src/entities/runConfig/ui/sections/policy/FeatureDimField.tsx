// web/run-manager/src/entities/runConfig/ui/sections/policy/FeatureDimField.tsx
import { IntegerTextInput, SegmentedChoiceStrip } from "@/shared/ui/configFields";
import { FieldLabel } from "@/shared/ui/configFields/label";
import { resetHandler } from "@/shared/ui/configFields/reset";

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

  function enableAuto() {
    onChange("auto");
  }

  function enableCustom() {
    if (typeof value === "number") {
      return;
    }
    const nextValue = defaultCustomValue(resetValue);
    onChange(nextValue);
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
          <IntegerTextInput
            aria-label={label}
            className="feature-dim-input"
            min={1}
            value={numericValue}
            onChange={onChange}
          />
        ) : null}
      </div>
    </div>
  );
}

function defaultCustomValue(resetValue: FeatureDim | undefined) {
  return typeof resetValue === "number" ? resetValue : DEFAULT_CUSTOM_FEATURE_DIM;
}
