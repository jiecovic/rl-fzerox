// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/action/types.ts

import type { ConfigSectionPatch, ConfigSetter } from "@/entities/runConfig/model/state";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

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
