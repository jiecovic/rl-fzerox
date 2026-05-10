// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/stateComponents/StateFeatureTable.tsx
import { ToggleSwitch } from "@/features/configurator/fields";
import type { ManagedRunConfig, StateComponentConfig } from "@/shared/api/contract";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";
import {
  rowDropoutProb,
  type StateFeatureRow,
  TRACK_POSITION_PROGRESS_ROW_ID,
} from "../featureRows";
import { ProgressSourceToggle } from "../ProgressSourceToggle";
import { FeatureAuxiliaryLossControls } from "./FeatureAuxiliaryLossControls";
import { FeatureDropoutInput } from "./FeatureDropoutInput";
import {
  type AuxiliaryStateTargetName,
  findAuxiliaryLoss,
  isRowIncluded,
  type StateComponentInfo,
} from "./model";

interface StateFeatureTableProps {
  allRowsIncluded: boolean;
  auxiliaryEnabled: boolean;
  checkpointLocked: boolean;
  component: StateComponentConfig;
  componentInfo: StateComponentInfo;
  config: ManagedRunConfig;
  enabled: boolean;
  rows: readonly StateFeatureRow[];
  updateComponent: (name: string, patch: Partial<StateComponentConfig>) => void;
  onAuxiliaryGroundedOnlyChange: (
    targetName: AuxiliaryStateTargetName,
    groundedOnly: boolean,
  ) => void;
  onAuxiliaryLossToggle: (targetName: AuxiliaryStateTargetName, enabled: boolean) => void;
  onAuxiliaryLossWeightChange: (targetName: AuxiliaryStateTargetName, weight: number) => void;
  onFeatureDropoutChange: (featureNames: readonly string[], dropoutProb: number) => void;
  onFeatureIncludedChange: (featureNames: readonly string[], included: boolean) => void;
  onRowsIncludedChange: (included: boolean) => void;
}

export function StateFeatureTable({
  allRowsIncluded,
  auxiliaryEnabled,
  checkpointLocked,
  component,
  componentInfo,
  config,
  enabled,
  rows,
  updateComponent,
  onAuxiliaryGroundedOnlyChange,
  onAuxiliaryLossToggle,
  onAuxiliaryLossWeightChange,
  onFeatureDropoutChange,
  onFeatureIncludedChange,
  onRowsIncludedChange,
}: StateFeatureTableProps) {
  return (
    <table className="state-feature-table">
      <thead>
        <tr>
          <th scope="col">Entry</th>
          <th scope="col">Type</th>
          <th scope="col">Range / size</th>
          <th scope="col">
            <span className="state-feature-column-label">
              <span>Episode dropout</span>
              <HelpTooltipButton
                label="Episode dropout"
                text="Probability that this feature is replaced with 0 for the whole episode. 0 means never. 1 means always."
              />
            </span>
          </th>
          <th scope="col">
            <span className="state-feature-column-label">
              <span>Use value</span>
              <HelpTooltipButton
                label="Use value"
                text="On: the policy receives this feature value. Off: the input slot remains, but the policy receives 0."
              />
            </span>
          </th>
          <th scope="col">
            <span className="state-feature-enabled-header">
              <span>Policy input</span>
              <ToggleSwitch
                checked={allRowsIncluded}
                disabled={!enabled || checkpointLocked}
                hideLabel
                label="all entries used as policy input"
                tooltip={
                  checkpointLocked
                    ? "Forked checkpoints keep the original state-vector shape."
                    : allRowsIncluded
                      ? "Remove every row in this category from the observation state"
                      : "Include every row in this category in the observation state"
                }
                onChange={onRowsIncludedChange}
              />
            </span>
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <StateFeatureTableRow
            auxiliaryEnabled={auxiliaryEnabled}
            checkpointLocked={checkpointLocked}
            component={component}
            componentInfo={componentInfo}
            config={config}
            enabled={enabled}
            key={row.id}
            row={row}
            updateComponent={updateComponent}
            onAuxiliaryGroundedOnlyChange={onAuxiliaryGroundedOnlyChange}
            onAuxiliaryLossToggle={onAuxiliaryLossToggle}
            onAuxiliaryLossWeightChange={onAuxiliaryLossWeightChange}
            onFeatureDropoutChange={onFeatureDropoutChange}
            onFeatureIncludedChange={onFeatureIncludedChange}
          />
        ))}
      </tbody>
    </table>
  );
}

interface StateFeatureTableRowProps {
  auxiliaryEnabled: boolean;
  checkpointLocked: boolean;
  component: StateComponentConfig;
  componentInfo: StateComponentInfo;
  config: ManagedRunConfig;
  enabled: boolean;
  row: StateFeatureRow;
  updateComponent: (name: string, patch: Partial<StateComponentConfig>) => void;
  onAuxiliaryGroundedOnlyChange: (
    targetName: AuxiliaryStateTargetName,
    groundedOnly: boolean,
  ) => void;
  onAuxiliaryLossToggle: (targetName: AuxiliaryStateTargetName, enabled: boolean) => void;
  onAuxiliaryLossWeightChange: (targetName: AuxiliaryStateTargetName, weight: number) => void;
  onFeatureDropoutChange: (featureNames: readonly string[], dropoutProb: number) => void;
  onFeatureIncludedChange: (featureNames: readonly string[], included: boolean) => void;
}

function StateFeatureTableRow({
  auxiliaryEnabled,
  checkpointLocked,
  component,
  componentInfo,
  config,
  enabled,
  row,
  updateComponent,
  onAuxiliaryGroundedOnlyChange,
  onAuxiliaryLossToggle,
  onAuxiliaryLossWeightChange,
  onFeatureDropoutChange,
  onFeatureIncludedChange,
}: StateFeatureTableRowProps) {
  const rowIncluded = enabled && isRowIncluded(componentInfo, component, row);
  const effectiveDropoutProb = rowIncluded ? rowDropoutProb(config, row) : 0;
  const auxiliaryTargetName = row.auxiliaryTargetName;
  const auxiliaryLoss = findAuxiliaryLoss(config.policy, auxiliaryTargetName);
  const progressSourceControl =
    row.id === TRACK_POSITION_PROGRESS_ROW_ID ? (
      <ProgressSourceToggle
        disabled={!enabled}
        value={component.progress_source ?? "segment_progress"}
        onChange={(value) =>
          updateComponent(component.name, {
            progress_source: value,
          })
        }
      />
    ) : null;

  return (
    <tr>
      <td>
        <div className="state-feature-entry">
          <div className="state-feature-name">
            <span>{row.label}</span>
            <HelpTooltipButton label={row.label} text={row.help} />
          </div>
          {auxiliaryTargetName !== undefined ? (
            <FeatureAuxiliaryLossControls
              auxiliaryEnabled={auxiliaryEnabled}
              disabled={false}
              extraControls={progressSourceControl}
              featureLabel={row.label}
              groundedOnly={auxiliaryLoss?.grounded_only ?? false}
              lossEnabled={auxiliaryLoss !== null}
              supportsGroundedOnly={row.auxiliarySupportsGroundedOnly}
              weight={auxiliaryLoss?.weight ?? 1.0}
              onGroundedOnlyChange={(checked) =>
                onAuxiliaryGroundedOnlyChange(auxiliaryTargetName, checked)
              }
              onToggle={(checked) => onAuxiliaryLossToggle(auxiliaryTargetName, checked)}
              onWeightChange={(value) => onAuxiliaryLossWeightChange(auxiliaryTargetName, value)}
            />
          ) : (
            progressSourceControl
          )}
        </div>
      </td>
      <td className="state-feature-kind">{row.kind}</td>
      <td className="state-feature-range">{row.range}</td>
      <td className="state-feature-dropout-cell">
        <FeatureDropoutInput
          disabled={!rowIncluded}
          label={`${row.label} episode dropout`}
          value={effectiveDropoutProb}
          onChange={(value) => onFeatureDropoutChange(row.featureNames, value)}
        />
      </td>
      <td className="state-feature-toggle-cell">
        <ToggleSwitch
          checked={effectiveDropoutProb < 1.0}
          disabled={!rowIncluded}
          hideLabel
          label={`${row.label} uses real value`}
          tooltip={
            effectiveDropoutProb < 1.0
              ? "Turn off to send 0 for this input."
              : "Turn on to use this feature value."
          }
          onChange={(checked) => onFeatureDropoutChange(row.featureNames, checked ? 0.0 : 1.0)}
        />
      </td>
      <td>
        <ToggleSwitch
          checked={rowIncluded}
          disabled={!enabled || checkpointLocked}
          hideLabel
          label="use entry as policy input"
          tooltip={
            checkpointLocked
              ? "Forked checkpoints keep the original state-vector shape."
              : rowIncluded
                ? "Remove this entry from the observation state"
                : "Include this entry in the observation state"
          }
          onChange={(checked) => onFeatureIncludedChange(row.featureNames, checked)}
        />
      </td>
    </tr>
  );
}
