// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/tracks/types.ts

import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import type { ConfigSetter } from "@/widgets/configurator/configurator/state";

export interface TracksSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
}

export type BuiltInCourse = ConfigMetadata["built_in_courses"][number];
export type GpDifficulty = ManagedRunConfig["tracks"]["gp_difficulties"][number];
export type TracksConfig = ManagedRunConfig["tracks"];
export type TrackUpdate = (patch: Partial<TracksConfig>) => void;

export type TrackCupView = ConfigMetadata["track_cups"][number] & {
  courses: (BuiltInCourse & { cup_order_index: number })[];
};
