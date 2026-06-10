// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/observation/stateComponents/StateComponentPanel.tsx

import {
  componentSummary,
  isRowZeroed,
  type StateFeatureRow,
  stateComponentInfoForConfig,
  stateFeatureRows,
} from "@/entities/runConfig/ui/sections/observation/featureRows";
import { ComponentSettings } from "@/entities/runConfig/ui/sections/observation/stateComponents/ComponentSettings";
import {
  type AuxiliaryStateTargetName,
  isRowIncluded,
  type StateComponentInfo,
} from "@/entities/runConfig/ui/sections/observation/stateComponents/model";
import { StateFeatureTable } from "@/entities/runConfig/ui/sections/observation/stateComponents/StateFeatureTable";
import type { ManagedRunConfig, StateComponentConfig } from "@/shared/api/contract";
import { ToggleSwitch } from "@/shared/ui/configFields";

interface StateComponentPanelProps {
  checkpointLocked: boolean;
  componentInfo: StateComponentInfo;
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  open: boolean;
  updateComponent: (name: string, patch: Partial<StateComponentConfig>) => void;
  onAuxiliaryGroundedOnlyChange: (
    targetName: AuxiliaryStateTargetName,
    groundedOnly: boolean,
  ) => void;
  onAuxiliaryLossToggle: (targetName: AuxiliaryStateTargetName, enabled: boolean) => void;
  onAuxiliaryLossWeightChange: (targetName: AuxiliaryStateTargetName, weight: number) => void;
  onComponentEnabledChange: (componentName: StateComponentConfig["name"], enabled: boolean) => void;
  onFeatureDropoutChange: (featureNames: readonly string[], dropoutProb: number) => void;
  onFeatureIncludedChange: (
    componentInfo: StateComponentInfo,
    componentName: StateComponentConfig["name"],
    featureNames: readonly string[],
    included: boolean,
  ) => void;
  onOpenChange: (name: string, open: boolean) => void;
  onRowsIncludedChange: (
    componentInfo: StateComponentInfo,
    componentName: StateComponentConfig["name"],
    rows: readonly StateFeatureRow[],
    included: boolean,
  ) => void;
}

export function StateComponentPanel({
  checkpointLocked,
  componentInfo,
  config,
  defaultConfig,
  open,
  updateComponent,
  onAuxiliaryGroundedOnlyChange,
  onAuxiliaryLossToggle,
  onAuxiliaryLossWeightChange,
  onComponentEnabledChange,
  onFeatureDropoutChange,
  onFeatureIncludedChange,
  onOpenChange,
  onRowsIncludedChange,
}: StateComponentPanelProps) {
  const component = config.observation.state_components.find(
    (item) => item.name === componentInfo.name,
  );
  const defaultComponent = defaultConfig.observation.state_components.find(
    (item) => item.name === componentInfo.name,
  );
  const displayComponent = component ?? defaultComponent;
  if (displayComponent === undefined) {
    return null;
  }

  const enabled = component !== undefined;
  const effectiveComponentInfo = stateComponentInfoForConfig(componentInfo, config);
  const rows = stateFeatureRows(displayComponent.name, effectiveComponentInfo.features);
  const notIncludedCount = enabled
    ? rows.filter((row) => !isRowIncluded(effectiveComponentInfo, displayComponent, row)).length
    : rows.length;
  const zeroedCount = rows.filter(
    (row) =>
      isRowIncluded(effectiveComponentInfo, displayComponent, row) && isRowZeroed(config, row),
  ).length;
  const allRowsIncluded = enabled && notIncludedCount === 0;

  return (
    <details
      className="config-disclosure state-component-panel"
      open={open}
      onToggle={(event) => onOpenChange(displayComponent.name, event.currentTarget.open)}
    >
      <summary className="config-disclosure-summary text-sm font-bold text-app-text">
        <span className="config-disclosure-title">
          <span className="config-disclosure-copy">
            <strong>{componentInfo.label}</strong>
            <small>
              {componentSummary(
                displayComponent.name,
                rows.length,
                rows.reduce((count, row) => count + row.featureNames.length, 0),
                zeroedCount,
                notIncludedCount,
                enabled,
              )}
            </small>
          </span>
        </span>
        <ToggleSwitch
          checked={enabled}
          disabled={checkpointLocked}
          hideLabel
          label="category enabled"
          tooltip={
            checkpointLocked
              ? "Forked checkpoints keep the original state-vector shape."
              : enabled
                ? "Disable category"
                : "Enable category"
          }
          onChange={(checked) => onComponentEnabledChange(displayComponent.name, checked)}
        />
      </summary>

      <div className="config-disclosure-body gap-3">
        <ComponentSettings
          checkpointLocked={checkpointLocked}
          component={displayComponent}
          disabled={!enabled}
          updateComponent={updateComponent}
        />
        <StateFeatureTable
          allRowsIncluded={allRowsIncluded}
          auxiliaryEnabled={config.policy.auxiliary_state_enabled}
          checkpointLocked={checkpointLocked}
          component={displayComponent}
          componentInfo={effectiveComponentInfo}
          config={config}
          enabled={enabled}
          rows={rows}
          updateComponent={updateComponent}
          onAuxiliaryGroundedOnlyChange={onAuxiliaryGroundedOnlyChange}
          onAuxiliaryLossToggle={onAuxiliaryLossToggle}
          onAuxiliaryLossWeightChange={onAuxiliaryLossWeightChange}
          onFeatureDropoutChange={onFeatureDropoutChange}
          onFeatureIncludedChange={(featureNames, included) =>
            onFeatureIncludedChange(
              effectiveComponentInfo,
              displayComponent.name,
              featureNames,
              included,
            )
          }
          onRowsIncludedChange={(included) =>
            onRowsIncludedChange(effectiveComponentInfo, displayComponent.name, rows, included)
          }
        />
      </div>
    </details>
  );
}
