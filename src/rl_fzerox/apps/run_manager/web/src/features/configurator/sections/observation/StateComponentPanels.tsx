// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/StateComponentPanels.tsx
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import type { ConfigMetadata, ManagedRunConfig, StateComponentConfig } from "@/shared/api/contract";

import { allStateComponentsOpen, type StateFeatureRow } from "./featureRows";
import {
  type AuxiliaryStateTargetName,
  type StateComponentInfo,
  setAuxiliaryGroundedOnlyPatch,
  setAuxiliaryLossEnabledPatch,
  setAuxiliaryLossWeightPatch,
  setAuxiliaryStateEnabledPatch,
  setComponentEnabledPatch,
  setFeatureDropoutPatch,
  setFeatureIncludedPatch,
  setRowsIncludedPatch,
  updateComponentPatch,
} from "./stateComponents/model";
import { StateAuxiliaryToolbar } from "./stateComponents/StateAuxiliaryToolbar";
import { StateComponentPanel } from "./stateComponents/StateComponentPanel";

interface StateComponentPanelsProps {
  checkpointLocked?: boolean;
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateObservation: (patch: Partial<ManagedRunConfig["observation"]>) => void;
  updatePolicy: (patch: Partial<ManagedRunConfig["policy"]>) => void;
}

export function StateComponentPanels({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  updateObservation,
  updatePolicy,
}: StateComponentPanelsProps) {
  const [openSections, setOpenSections] = usePersistentDisclosureMap(
    "run-manager:observation:state-components",
    allStateComponentsOpen(metadata, false),
  );

  const setSectionOpen = (name: string, open: boolean) => {
    setOpenSections((current) => ({ ...current, [name]: open }));
  };

  function updateComponent(name: string, patch: Partial<StateComponentConfig>) {
    updateObservation(updateComponentPatch(config, name, patch));
  }

  function setComponentEnabled(componentName: StateComponentConfig["name"], enabled: boolean) {
    updateObservation(
      setComponentEnabledPatch({
        componentName,
        config,
        defaultConfig,
        enabled,
        metadata,
      }),
    );
  }

  function setFeatureIncluded(
    componentInfo: StateComponentInfo,
    componentName: StateComponentConfig["name"],
    featureNames: readonly string[],
    included: boolean,
  ) {
    updateObservation(
      setFeatureIncludedPatch({
        componentInfo,
        componentName,
        config,
        featureNames,
        included,
      }),
    );
  }

  function setRowsIncluded(
    componentInfo: StateComponentInfo,
    componentName: StateComponentConfig["name"],
    rows: readonly StateFeatureRow[],
    included: boolean,
  ) {
    updateObservation(
      setRowsIncludedPatch({
        active: included,
        componentInfo,
        componentName,
        config,
        rows,
      }),
    );
  }

  function setFeatureDropoutProb(featureNames: readonly string[], dropoutProb: number) {
    updateObservation(setFeatureDropoutPatch(config, featureNames, dropoutProb));
  }

  function setAuxiliaryStateEnabled(enabled: boolean) {
    updatePolicy(setAuxiliaryStateEnabledPatch(config.policy, enabled));
  }

  function setAuxiliaryLossEnabled(targetName: AuxiliaryStateTargetName, enabled: boolean) {
    updatePolicy(setAuxiliaryLossEnabledPatch(config.policy, targetName, enabled));
  }

  function setAuxiliaryLossWeight(targetName: AuxiliaryStateTargetName, weight: number) {
    updatePolicy(setAuxiliaryLossWeightPatch(config.policy, targetName, weight));
  }

  function setAuxiliaryGroundedOnly(targetName: AuxiliaryStateTargetName, groundedOnly: boolean) {
    updatePolicy(setAuxiliaryGroundedOnlyPatch(config.policy, targetName, groundedOnly));
  }

  return (
    <div className="state-vector-editor">
      <StateAuxiliaryToolbar
        enabled={config.policy.auxiliary_state_enabled}
        onChange={setAuxiliaryStateEnabled}
      />
      <DisclosureToolbar
        collapseLabel="Collapse all state-vector categories"
        expandLabel="Expand all state-vector categories"
        onCollapseAll={() => setOpenSections(allStateComponentsOpen(metadata, false))}
        onExpandAll={() => setOpenSections(allStateComponentsOpen(metadata, true))}
      />
      {metadata.state_components.map((componentInfo) => (
        <StateComponentPanel
          checkpointLocked={checkpointLocked}
          componentInfo={componentInfo}
          config={config}
          defaultConfig={defaultConfig}
          key={componentInfo.name}
          open={openSections[componentInfo.name] ?? false}
          updateComponent={updateComponent}
          onAuxiliaryGroundedOnlyChange={setAuxiliaryGroundedOnly}
          onAuxiliaryLossToggle={setAuxiliaryLossEnabled}
          onAuxiliaryLossWeightChange={setAuxiliaryLossWeight}
          onComponentEnabledChange={setComponentEnabled}
          onFeatureDropoutChange={setFeatureDropoutProb}
          onFeatureIncludedChange={setFeatureIncluded}
          onOpenChange={setSectionOpen}
          onRowsIncludedChange={setRowsIncluded}
        />
      ))}
    </div>
  );
}
