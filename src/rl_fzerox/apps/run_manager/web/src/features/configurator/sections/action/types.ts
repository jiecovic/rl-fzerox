import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

export interface ActionSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  checkpointLocked?: boolean;
  metadata: ConfigMetadata;
  setConfig: (config: ManagedRunConfig) => void;
}

export interface ActionUpdateContext {
  config: ManagedRunConfig;
  checkpointLocked?: boolean;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateAction: (patch: Partial<ManagedRunConfig["action"]>) => void;
  updatePolicy: (patch: Partial<ManagedRunConfig["policy"]>) => void;
  setConfig: (config: ManagedRunConfig) => void;
}
