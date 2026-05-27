// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/types.ts
import type { ConfigSetter } from "@/features/configurator/configurator/state";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

export interface TracksSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
}

export type BuiltInCourse = ConfigMetadata["built_in_courses"][number];
export type GpDifficulty = NonNullable<ManagedRunConfig["tracks"]["gp_difficulty"]>;
export type TracksConfig = ManagedRunConfig["tracks"];
export type TrackUpdate = (patch: Partial<TracksConfig>) => void;

export type TrackCupView = ConfigMetadata["track_cups"][number] & {
  courses: (BuiltInCourse & { cup_order_index: number })[];
};
