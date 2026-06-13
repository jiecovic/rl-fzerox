// web/run-manager/src/entities/trackPool/model/types.ts
import type {
  ConfigMetadata,
  ManagedRunDetail,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";

export interface RunTrackPoolPanelProps {
  canReset: boolean;
  isClearingAltBaselines: boolean;
  isResetting: boolean;
  metadata: ConfigMetadata;
  onClearAltBaselines: () => void;
  onReset: () => void;
  run: ManagedRunDetail;
  state: TrackSamplingRuntimeState | null;
}

export type TrackPoolCourseView = {
  completedEnvSteps: number | null;
  currentProbability: number | null;
  emaCompletionFraction: number | null;
  episodeCount: number | null;
  episodeShare: number | null;
  finishedEpisodeCount: number | null;
  generationEmaCompletionFraction: number | null;
  generationEpisodeCount: number | null;
  generationFinishedEpisodeCount: number | null;
  generationSuccessRate: number | null;
  generationSuccessSampleCount: number | null;
  generatedCourseGeneration: number | null;
  generatedCourseSlot: number | null;
  id: string;
  label: string;
  stepShare: number | null;
  targetStepShare: number | null;
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
  stepMetricLabel: string;
  showStepTarget: boolean;
  totalCourses: number;
  totalEnvSteps: number;
  totalEpisodes: number;
};
