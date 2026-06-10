// web/run-manager/src/entities/runConfig/ui/sections/observation/StateComponentPanels.tsx

import type { ConfigSectionPatch } from "@/entities/runConfig/model/state";
import {
  allStateComponentsOpen,
  type StateFeatureRow,
} from "@/entities/runConfig/ui/sections/observation/featureRows";
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
} from "@/entities/runConfig/ui/sections/observation/stateComponents/model";
import { StateAuxiliaryToolbar } from "@/entities/runConfig/ui/sections/observation/stateComponents/StateAuxiliaryToolbar";
import { StateComponentPanel } from "@/entities/runConfig/ui/sections/observation/stateComponents/StateComponentPanel";
import type { ConfigMetadata, ManagedRunConfig, StateComponentConfig } from "@/shared/api/contract";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/shared/ui/config/disclosureState";

interface StateComponentPanelsProps {
  checkpointLocked?: boolean;
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateObservation: (patch: ConfigSectionPatch<"observation">) => void;
  updatePolicy: (patch: ConfigSectionPatch<"policy">) => void;
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
    updateObservation((currentConfig) => updateComponentPatch(currentConfig, name, patch));
  }

  function setComponentEnabled(componentName: StateComponentConfig["name"], enabled: boolean) {
    updateObservation((currentConfig) =>
      setComponentEnabledPatch({
        componentName,
        config: currentConfig,
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
    updateObservation((currentConfig) =>
      setFeatureIncludedPatch({
        componentInfo,
        componentName,
        config: currentConfig,
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
    updateObservation((currentConfig) =>
      setRowsIncludedPatch({
        active: included,
        componentInfo,
        componentName,
        config: currentConfig,
        rows,
      }),
    );
  }

  function setFeatureDropoutProb(featureNames: readonly string[], dropoutProb: number) {
    updateObservation((currentConfig) =>
      setFeatureDropoutPatch(currentConfig, featureNames, dropoutProb),
    );
  }

  function setAuxiliaryStateEnabled(enabled: boolean) {
    updatePolicy((currentConfig) => setAuxiliaryStateEnabledPatch(currentConfig.policy, enabled));
  }

  function setAuxiliaryLossEnabled(targetName: AuxiliaryStateTargetName, enabled: boolean) {
    updatePolicy((currentConfig) =>
      setAuxiliaryLossEnabledPatch(currentConfig.policy, targetName, enabled),
    );
  }

  function setAuxiliaryLossWeight(targetName: AuxiliaryStateTargetName, weight: number) {
    updatePolicy((currentConfig) =>
      setAuxiliaryLossWeightPatch(currentConfig.policy, targetName, weight),
    );
  }

  function setAuxiliaryGroundedOnly(targetName: AuxiliaryStateTargetName, groundedOnly: boolean) {
    updatePolicy((currentConfig) =>
      setAuxiliaryGroundedOnlyPatch(currentConfig.policy, targetName, groundedOnly),
    );
  }

  return (
    <div className="grid gap-2.5">
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
