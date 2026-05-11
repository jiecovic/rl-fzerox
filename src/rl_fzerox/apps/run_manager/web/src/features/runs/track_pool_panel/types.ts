// src/rl_fzerox/apps/run_manager/web/src/features/runs/track_pool_panel/types.ts
import type {
  ConfigMetadata,
  ManagedRunDetail,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";

export interface RunTrackPoolPanelProps {
  canReset: boolean;
  isResetting: boolean;
  metadata: ConfigMetadata;
  onReset: () => void;
  run: ManagedRunDetail;
  state: TrackSamplingRuntimeState | null;
}

export type TrackPoolCourseView = {
  completedEnvSteps: number | null;
  currentProbability: number | null;
  episodeCount: number | null;
  episodeShare: number | null;
  finishedEpisodeCount: number | null;
  id: string;
  label: string;
  stepShare: number | null;
  successRate: number | null;
  successSampleCount: number | null;
};

export type TrackPoolCupView = {
  completedEnvSteps: number;
  currentProbability: number;
  entries: TrackPoolCourseView[];
  episodeCount: number;
  episodeShare: number;
  finishedEpisodeCount: number;
  id: string;
  label: string;
  stepShare: number;
  successRate: number | null;
  successSampleCount: number;
};

export type TrackPoolView = {
  cups: TrackPoolCupView[];
  totalCourses: number;
  totalEnvSteps: number;
  totalEpisodes: number;
};
