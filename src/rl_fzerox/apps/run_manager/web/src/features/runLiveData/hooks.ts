// src/rl_fzerox/apps/run_manager/web/src/features/runLiveData/hooks.ts
import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchPolicyPreview,
  fetchRunTrackSamplingState,
  subscribeRunTrackSamplingUpdates,
} from "@/shared/api/client";
import type {
  ManagedRun,
  ManagedRunConfig,
  PolicyArchitecturePreview,
  TrackSamplingRuntimeEntry,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { useDocumentVisible } from "@/shared/browser/useDocumentVisible";

const TRACK_SAMPLING_LIVE = {
  fallbackPollMs: 5_000,
} as const;

export function useRunClock(status: ManagedRun["status"]): number {
  const [nowMs, setNowMs] = useState(() => Date.now());
  const documentVisible = useDocumentVisible();

  useEffect(() => {
    if (status !== "running" || !documentVisible) {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1_000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [documentVisible, status]);

  return nowMs;
}

export function useRunPolicyPreview(
  config: ManagedRunConfig,
  enabled: boolean,
): {
  policyPreview: PolicyArchitecturePreview | null;
  previewError: string | null;
} {
  const [policyPreview, setPolicyPreview] = useState<PolicyArchitecturePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setPreviewError(null);
      return undefined;
    }

    let ignore = false;
    const controller = new AbortController();
    setPreviewError(null);
    void fetchPolicyPreview(config, { signal: controller.signal })
      .then((preview) => {
        if (!ignore) {
          setPolicyPreview(preview);
        }
      })
      .catch((caught) => {
        if (!ignore) {
          setPolicyPreview(null);
          setPreviewError(
            caught instanceof Error ? caught.message : "failed to compute policy preview",
          );
        }
      });
    return () => {
      ignore = true;
      controller.abort();
    };
  }, [config, enabled]);

  return { policyPreview, previewError };
}

export function useRunTrackSamplingState(
  runId: string,
  status: ManagedRun["status"],
): {
  setTrackSamplingState: (state: TrackSamplingRuntimeState | null) => void;
  trackSamplingError: string | null;
  trackSamplingState: TrackSamplingRuntimeState | null;
} {
  const [trackSamplingState, setTrackSamplingState] = useState<TrackSamplingRuntimeState | null>(
    null,
  );
  const [trackSamplingError, setTrackSamplingError] = useState<string | null>(null);
  const trackSamplingStateKeyRef = useRef<string | null>(null);
  const documentVisible = useDocumentVisible();
  const commitTrackSamplingState = useCallback(
    (state: TrackSamplingRuntimeState | null) => {
      const key = trackSamplingStateKey(runId, state);
      trackSamplingStateKeyRef.current = key;
      setTrackSamplingState(state);
    },
    [runId],
  );

  useEffect(() => {
    commitTrackSamplingState(null);
    setTrackSamplingError(null);
  }, [commitTrackSamplingState]);

  useEffect(() => {
    if (!documentVisible) {
      return undefined;
    }

    let unsubscribe: (() => void) | null = null;
    let fallbackPollInterval: number | null = null;
    let ignore = false;
    let inFlight = false;
    let liveConnected = false;
    let activeController: AbortController | null = null;

    async function loadTrackSamplingState() {
      if (inFlight || liveConnected || document.visibilityState === "hidden") {
        return;
      }
      inFlight = true;
      const controller = new AbortController();
      activeController = controller;
      try {
        const state = await fetchRunTrackSamplingState(runId, { signal: controller.signal });
        if (!ignore) {
          const key = trackSamplingStateKey(runId, state);
          if (trackSamplingStateKeyRef.current !== key) {
            trackSamplingStateKeyRef.current = key;
            setTrackSamplingState(state);
          }
          setTrackSamplingError(null);
        }
      } catch (caught) {
        if (!ignore) {
          setTrackSamplingError(
            caught instanceof Error ? caught.message : "failed to load track-pool stats",
          );
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        inFlight = false;
      }
    }

    function startFallbackPolling() {
      if (fallbackPollInterval !== null) {
        return;
      }
      void loadTrackSamplingState();
      if (status !== "running") {
        return;
      }
      fallbackPollInterval = window.setInterval(() => {
        void loadTrackSamplingState();
      }, TRACK_SAMPLING_LIVE.fallbackPollMs);
    }

    function stopFallbackPolling() {
      if (fallbackPollInterval === null) {
        return;
      }
      window.clearInterval(fallbackPollInterval);
      fallbackPollInterval = null;
    }

    if (status !== "running") {
      void loadTrackSamplingState();
      return () => {
        ignore = true;
        activeController?.abort();
      };
    }

    unsubscribe = subscribeRunTrackSamplingUpdates(runId, {
      onConnectionChange: (connected) => {
        liveConnected = connected;
        if (connected) {
          stopFallbackPolling();
          setTrackSamplingError(null);
          return;
        }
        startFallbackPolling();
      },
      onError: (caught) => {
        setTrackSamplingError(caught.message);
      },
      onState: (state) => {
        commitTrackSamplingState(state);
        setTrackSamplingError(null);
      },
    });

    return () => {
      ignore = true;
      activeController?.abort();
      unsubscribe?.();
      stopFallbackPolling();
    };
  }, [commitTrackSamplingState, documentVisible, runId, status]);

  return {
    setTrackSamplingState: commitTrackSamplingState,
    trackSamplingError,
    trackSamplingState,
  };
}

function trackSamplingStateKey(runId: string, state: TrackSamplingRuntimeState | null) {
  if (state === null) {
    return `${runId}:none`;
  }
  return [
    runId,
    state.sampling_mode,
    state.action_repeat,
    state.update_episodes,
    state.ema_alpha,
    state.max_weight_scale,
    state.adaptive_completion_weight,
    state.adaptive_target_completion,
    state.adaptive_min_confidence_episodes,
    state.adaptive_confidence_scale,
    state.update_count,
    state.episodes_since_update,
    ...state.entries.map(trackSamplingEntryKey),
  ].join("\0");
}

function trackSamplingEntryKey(entry: TrackSamplingRuntimeEntry) {
  return [
    entry.track_id,
    entry.course_key,
    entry.label,
    entry.current_weight,
    entry.current_probability,
    entry.episode_count,
    entry.finished_episode_count,
    entry.success_sample_count,
    nullableNumberKey(entry.success_rate),
    entry.generation_episode_count,
    entry.generation_finished_episode_count,
    entry.generation_success_sample_count,
    nullableNumberKey(entry.generation_success_rate),
    nullableNumberKey(entry.generation_ema_completion_fraction),
    entry.target_step_share,
    entry.completed_frames,
    entry.completed_env_steps,
    entry.step_share,
    nullableNumberKey(entry.ema_episode_frames),
    nullableNumberKey(entry.ema_completion_fraction),
    nullableNumberKey(entry.generated_course_slot),
    nullableNumberKey(entry.generated_course_generation),
  ].join("\u0001");
}

function nullableNumberKey(value: number | null) {
  return value === null ? "" : String(value);
}
