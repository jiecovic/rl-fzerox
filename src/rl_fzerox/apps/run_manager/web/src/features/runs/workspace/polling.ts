// src/rl_fzerox/apps/run_manager/web/src/features/runs/workspace/polling.ts
import { useEffect, useState } from "react";

import { fetchPolicyPreview, fetchRunTrackSamplingState } from "@/shared/api/client";
import type {
  ManagedRun,
  ManagedRunConfig,
  PolicyArchitecturePreview,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";

export function useRunClock(status: ManagedRun["status"]): number {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (status !== "running") {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1_000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [status]);

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
    setPreviewError(null);
    void fetchPolicyPreview(config)
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
    };
  }, [config, enabled]);

  return { policyPreview, previewError };
}

export function useRunTrackSamplingState(
  runId: string,
  status: ManagedRun["status"],
): {
  setTrackSamplingState: (state: TrackSamplingRuntimeState | null) => void;
  trackSamplingState: TrackSamplingRuntimeState | null;
} {
  const [trackSamplingState, setTrackSamplingState] = useState<TrackSamplingRuntimeState | null>(
    null,
  );

  useEffect(() => {
    let ignore = false;
    let inFlight = false;
    let activeController: AbortController | null = null;

    async function loadTrackSamplingState() {
      if (inFlight) {
        return;
      }
      inFlight = true;
      const controller = new AbortController();
      activeController = controller;
      try {
        const state = await fetchRunTrackSamplingState(runId, { signal: controller.signal });
        if (!ignore) {
          setTrackSamplingState(state);
        }
      } catch {
        if (!ignore) {
          setTrackSamplingState(null);
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        inFlight = false;
      }
    }

    void loadTrackSamplingState();
    if (status !== "running") {
      return () => {
        ignore = true;
        activeController?.abort();
      };
    }
    const intervalId = window.setInterval(() => {
      void loadTrackSamplingState();
    }, 2_000);
    return () => {
      ignore = true;
      activeController?.abort();
      window.clearInterval(intervalId);
    };
  }, [runId, status]);

  return { setTrackSamplingState, trackSamplingState };
}
