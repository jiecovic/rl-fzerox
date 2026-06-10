// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/observation/stateComponents/model.ts

import type {
  ConfigMetadata,
  ManagedRunConfig,
  StateComponentConfig,
  StateFeatureDropoutConfig,
} from "@/shared/api/contract";
import {
  type StateFeatureRow,
  stateComponentInfoForConfig,
} from "@/widgets/configurator/sections/observation/featureRows";

export type StateComponentInfo = ConfigMetadata["state_components"][number];
type ObservationPatch = Partial<ManagedRunConfig["observation"]>;
type PolicyPatch = Partial<ManagedRunConfig["policy"]>;
type AuxiliaryStateLoss = ManagedRunConfig["policy"]["auxiliary_state_losses"][number];
export type AuxiliaryStateTargetName = AuxiliaryStateLoss["name"];

export function updateComponentPatch(
  config: ManagedRunConfig,
  name: string,
  patch: Partial<StateComponentConfig>,
): ObservationPatch {
  return {
    state_components: config.observation.state_components.map((component) =>
      component.name === name ? { ...component, ...patch } : component,
    ),
  };
}

export function setComponentEnabledPatch({
  componentName,
  config,
  defaultConfig,
  enabled,
  metadata,
}: {
  componentName: StateComponentConfig["name"];
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  enabled: boolean;
  metadata: ConfigMetadata;
}): ObservationPatch {
  const currentComponents = new Map(
    config.observation.state_components.map((component) => [component.name, component]),
  );
  const nextComponentNames = new Set<StateComponentConfig["name"]>(currentComponents.keys());
  if (enabled) {
    nextComponentNames.add(componentName);
  } else {
    nextComponentNames.delete(componentName);
  }

  const componentInfo = metadata.state_components.find((item) => item.name === componentName);
  const componentFeatureNames = new Set(
    componentInfo === undefined
      ? []
      : stateComponentInfoForConfig(componentInfo, config).features.map((feature) => feature.name),
  );

  return {
    state_components: defaultConfig.observation.state_components
      .filter((component) => nextComponentNames.has(component.name))
      .map((component) => currentComponents.get(component.name) ?? component),
    state_feature_dropouts: config.observation.state_feature_dropouts.filter(
      (feature) => enabled || !componentFeatureNames.has(feature.name),
    ),
  };
}

export function setFeatureIncludedPatch({
  componentInfo,
  componentName,
  config,
  featureNames,
  included,
}: {
  componentInfo: StateComponentInfo;
  componentName: StateComponentConfig["name"];
  config: ManagedRunConfig;
  featureNames: readonly string[];
  included: boolean;
}): ObservationPatch {
  const component = config.observation.state_components.find((item) => item.name === componentName);
  if (component === undefined) {
    return {};
  }

  const nextIncludedFeatures = new Set(includedFeatureNames(componentInfo, component));
  if (included) {
    for (const featureName of featureNames) {
      nextIncludedFeatures.add(featureName);
    }
  } else {
    for (const featureName of featureNames) {
      nextIncludedFeatures.delete(featureName);
    }
  }

  const removedFeatureNames = new Set(included ? [] : featureNames);
  return {
    state_components: config.observation.state_components.map((item) =>
      item.name === componentName
        ? {
            ...item,
            included_features: normalizedIncludedFeatures(componentInfo, nextIncludedFeatures),
          }
        : item,
    ),
    state_feature_dropouts: included
      ? config.observation.state_feature_dropouts
      : config.observation.state_feature_dropouts.filter(
          (feature) => !removedFeatureNames.has(feature.name),
        ),
  };
}

export function setRowsIncludedPatch({
  active,
  componentInfo,
  componentName,
  config,
  rows,
}: {
  active: boolean;
  componentInfo: StateComponentInfo;
  componentName: StateComponentConfig["name"];
  config: ManagedRunConfig;
  rows: readonly StateFeatureRow[];
}): ObservationPatch {
  const component = config.observation.state_components.find((item) => item.name === componentName);
  if (component === undefined) {
    return {};
  }

  const featureNames = rows.flatMap((row) => row.featureNames);
  const selectedFeatures = active ? new Set(featureNames) : new Set<string>();
  const featureNameSet = new Set(featureNames);

  return {
    state_components: config.observation.state_components.map((item) =>
      item.name === componentName
        ? {
            ...item,
            included_features: normalizedIncludedFeatures(componentInfo, selectedFeatures),
          }
        : item,
    ),
    state_feature_dropouts: active
      ? config.observation.state_feature_dropouts
      : config.observation.state_feature_dropouts.filter(
          (feature) => !featureNameSet.has(feature.name),
        ),
  };
}

export function setFeatureDropoutPatch(
  config: ManagedRunConfig,
  featureNames: readonly string[],
  dropoutProb: number,
): ObservationPatch {
  const featureNameSet = new Set(featureNames);
  const retainedDropouts = config.observation.state_feature_dropouts.filter(
    (feature) => !featureNameSet.has(feature.name),
  );
  const clampedDropoutProb = Math.max(0, Math.min(1, dropoutProb));
  const nextEntries = featureNames
    .map((featureName) => {
      const nextEntry: StateFeatureDropoutConfig = {
        name: featureName,
        dropout_prob: clampedDropoutProb,
      };
      return nextEntry.dropout_prob <= 0.0 ? null : nextEntry;
    })
    .filter((entry): entry is StateFeatureDropoutConfig => entry !== null);

  return {
    state_feature_dropouts:
      nextEntries.length === 0
        ? retainedDropouts
        : [...retainedDropouts, ...nextEntries].sort((left, right) =>
            left.name.localeCompare(right.name),
          ),
  };
}

function includedFeatureNames(
  componentInfo: StateComponentInfo,
  component: StateComponentConfig,
): readonly string[] {
  return component.included_features ?? defaultIncludedFeatureNames(componentInfo);
}

export function isRowIncluded(
  componentInfo: StateComponentInfo,
  component: StateComponentConfig,
  row: StateFeatureRow,
) {
  const includedFeatures = new Set(includedFeatureNames(componentInfo, component));
  return row.featureNames.every((featureName) => includedFeatures.has(featureName));
}

export function findAuxiliaryLoss(
  policy: ManagedRunConfig["policy"],
  targetName: AuxiliaryStateTargetName | undefined,
) {
  if (targetName === undefined) {
    return null;
  }
  return policy.auxiliary_state_losses.find((loss) => loss.name === targetName) ?? null;
}

export function setAuxiliaryStateEnabledPatch(
  policy: ManagedRunConfig["policy"],
  enabled: boolean,
): PolicyPatch {
  return {
    auxiliary_state_enabled: enabled,
    auxiliary_state_losses: enabled ? policy.auxiliary_state_losses : [],
  };
}

export function setAuxiliaryLossEnabledPatch(
  policy: ManagedRunConfig["policy"],
  targetName: AuxiliaryStateTargetName,
  enabled: boolean,
): PolicyPatch {
  const retainedLosses = policy.auxiliary_state_losses.filter((loss) => loss.name !== targetName);
  const nextLosses = enabled
    ? [
        ...retainedLosses,
        {
          name: targetName,
          weight: 1.0,
          grounded_only: false,
        },
      ].sort((left, right) => left.name.localeCompare(right.name))
    : retainedLosses;

  return {
    auxiliary_state_enabled: enabled ? true : policy.auxiliary_state_enabled,
    auxiliary_state_losses: nextLosses,
  };
}

export function setAuxiliaryLossWeightPatch(
  policy: ManagedRunConfig["policy"],
  targetName: AuxiliaryStateTargetName,
  weight: number,
): PolicyPatch {
  return {
    auxiliary_state_enabled: true,
    auxiliary_state_losses: policy.auxiliary_state_losses.map((loss) =>
      loss.name === targetName ? { ...loss, weight } : loss,
    ),
  };
}

export function setAuxiliaryGroundedOnlyPatch(
  policy: ManagedRunConfig["policy"],
  targetName: AuxiliaryStateTargetName,
  groundedOnly: boolean,
): PolicyPatch {
  return {
    auxiliary_state_enabled: true,
    auxiliary_state_losses: policy.auxiliary_state_losses.map((loss) =>
      loss.name === targetName ? { ...loss, grounded_only: groundedOnly } : loss,
    ),
  };
}

function normalizedIncludedFeatures(
  componentInfo: StateComponentInfo,
  selectedFeatures: ReadonlySet<string>,
) {
  const orderedIncludedFeatures = orderedFeatureNames(componentInfo, selectedFeatures);
  const defaultIncludedFeatures = defaultIncludedFeatureNames(componentInfo);
  return arraysEqual(orderedIncludedFeatures, defaultIncludedFeatures)
    ? null
    : orderedIncludedFeatures;
}

function defaultIncludedFeatureNames(componentInfo: StateComponentInfo): string[] {
  return componentInfo.features
    .filter((feature) => feature.default_enabled)
    .map((feature) => feature.name);
}

function orderedFeatureNames(
  componentInfo: StateComponentInfo,
  selectedFeatures: ReadonlySet<string>,
): string[] {
  return componentInfo.features
    .map((feature) => feature.name)
    .filter((featureName) => selectedFeatures.has(featureName));
}

function arraysEqual(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
