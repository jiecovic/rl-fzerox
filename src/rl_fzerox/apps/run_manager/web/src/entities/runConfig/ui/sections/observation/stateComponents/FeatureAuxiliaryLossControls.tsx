// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/observation/stateComponents/FeatureAuxiliaryLossControls.tsx
import type { ReactNode } from "react";
import { BoundedDecimalInput } from "@/entities/runConfig/ui/sections/observation/stateComponents/BoundedDecimalInput";
import { ToggleSwitch } from "@/shared/ui/configFields";

interface FeatureAuxiliaryLossControlsProps {
  auxiliaryEnabled: boolean;
  disabled: boolean;
  extraControls?: ReactNode;
  featureLabel: string;
  groundedOnly: boolean;
  lossEnabled: boolean;
  supportsGroundedOnly: boolean;
  weight: number;
  onGroundedOnlyChange: (checked: boolean) => void;
  onToggle: (checked: boolean) => void;
  onWeightChange: (value: number) => void;
}

export function FeatureAuxiliaryLossControls({
  auxiliaryEnabled,
  disabled,
  extraControls,
  featureLabel,
  groundedOnly,
  lossEnabled,
  supportsGroundedOnly,
  weight,
  onGroundedOnlyChange,
  onToggle,
  onWeightChange,
}: FeatureAuxiliaryLossControlsProps) {
  if (disabled) {
    return <span className="state-feature-aux-unsupported">not available</span>;
  }

  const weightEnabled = auxiliaryEnabled && lossEnabled;
  return (
    <div className="state-feature-aux-controls">
      <ToggleSwitch
        checked={lossEnabled}
        disabled={false}
        hideLabel
        label={`${featureLabel} auxiliary loss enabled`}
        tooltip={lossEnabled ? "Disable aux loss" : "Enable aux loss"}
        onChange={onToggle}
      />
      <div className="state-feature-aux-weight">
        <BoundedDecimalInput
          className="state-feature-aux-weight-input"
          disabled={!weightEnabled}
          label={`${featureLabel} auxiliary loss weight`}
          min={0.01}
          step="0.01"
          value={weight}
          onChange={onWeightChange}
        />
      </div>
      {supportsGroundedOnly ? (
        <label className="state-feature-aux-grounded">
          <input
            checked={groundedOnly}
            disabled={!weightEnabled}
            type="checkbox"
            onChange={(event) => onGroundedOnlyChange(event.target.checked)}
          />
          <span>grounded only</span>
        </label>
      ) : null}
      {extraControls}
    </div>
  );
}
