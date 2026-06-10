// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/action/types.ts

import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import type { ConfigSectionPatch, ConfigSetter } from "@/widgets/configurator/configurator/state";

export interface ActionSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  checkpointLocked?: boolean;
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
}

export interface ActionUpdateContext {
  config: ManagedRunConfig;
  checkpointLocked?: boolean;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateAction: (patch: ConfigSectionPatch<"action">) => void;
  updatePolicy: (patch: ConfigSectionPatch<"policy">) => void;
  updateTrain: (patch: ConfigSectionPatch<"train">) => void;
  setConfig: ConfigSetter;
}
