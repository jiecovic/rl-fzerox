// web/run-manager/src/entities/runConfig/ui/sections/tracks/types.ts

import type { ConfigSetter } from "@/entities/runConfig/model/state";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

export interface TracksSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  gpDifficultySelection?: "multi" | "single";
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
  showSampling?: boolean;
}

export type BuiltInCourse = ConfigMetadata["built_in_courses"][number];
export type GpDifficulty = ManagedRunConfig["tracks"]["gp_difficulties"][number];
export type TracksConfig = ManagedRunConfig["tracks"];
export type TrackUpdate = (patch: Partial<TracksConfig>) => void;

export type TrackCupView = ConfigMetadata["track_cups"][number] & {
  courses: (BuiltInCourse & { cup_order_index: number })[];
};
