// src/rl_fzerox/apps/run_manager/web/src/shared/api/client/live.ts

import { parseApiPayload } from "@/shared/api/client/http";
import {
  type ManagedRun,
  type RunsLiveUpdate,
  type RunTrackSamplingLiveUpdate,
  runsLiveUpdateSchema,
  runTrackSamplingLiveUpdateSchema,
  type TrackSamplingRuntimeState,
} from "@/shared/api/contract";

const RUN_LIVE_RECONNECT_DELAY_MS = 1_500;

interface LiveSocketSubscriptionOptions<TUpdate> {
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
  onUpdate: (update: TUpdate) => void;
  parseUpdate: (payload: unknown) => TUpdate;
  path: string;
}

export interface RunLiveSubscriptionOptions {
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
  onRuns: (runs: ManagedRun[]) => void;
}

export interface RunTrackSamplingLiveSubscriptionOptions {
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
  onState: (state: TrackSamplingRuntimeState | null) => void;
}

export function subscribeRunLiveUpdates({
  onConnectionChange,
  onError,
  onRuns,
}: RunLiveSubscriptionOptions): () => void {
  return subscribeLiveSocket<RunsLiveUpdate>({
    onConnectionChange,
    onError,
    onUpdate: (parsed) => {
      if (parsed.type === "runs_error") {
        onError?.(new Error(`live run refresh failed: ${parsed.message}`));
        return;
      }
      onRuns(parsed.runs);
    },
    parseUpdate: (payload) => parseApiPayload(runsLiveUpdateSchema, payload),
    path: "/api/runs/live",
  });
}

export function subscribeRunTrackSamplingUpdates(
  runId: string,
  { onConnectionChange, onError, onState }: RunTrackSamplingLiveSubscriptionOptions,
): () => void {
  return subscribeLiveSocket<RunTrackSamplingLiveUpdate>({
    onConnectionChange,
    onError,
    onUpdate: (parsed) => {
      if (parsed.type === "track_sampling_error") {
        onError?.(new Error(`live track-pool refresh failed: ${parsed.message}`));
        return;
      }
      onState(parsed.state);
    },
    parseUpdate: (payload) => parseApiPayload(runTrackSamplingLiveUpdateSchema, payload),
    path: `/api/runs/${encodeURIComponent(runId)}/track-sampling/live`,
  });
}

function subscribeLiveSocket<TUpdate>({
  onConnectionChange,
  onError,
  onUpdate,
  parseUpdate,
  path,
}: LiveSocketSubscriptionOptions<TUpdate>): () => void {
  let closed = false;
  let reconnectTimer: number | null = null;
  let socket: WebSocket | null = null;

  function connect() {
    if (closed) {
      return;
    }
    socket = new WebSocket(apiWebSocketUrl(path));
    socket.addEventListener("open", () => {
      onConnectionChange?.(true);
    });
    socket.addEventListener("message", (event) => {
      try {
        onUpdate(parseUpdate(JSON.parse(String(event.data))));
      } catch (caught) {
        onError?.(caught instanceof Error ? caught : new Error("invalid live update payload"));
      }
    });
    socket.addEventListener("close", () => {
      socket = null;
      if (closed) {
        return;
      }
      onConnectionChange?.(false);
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, RUN_LIVE_RECONNECT_DELAY_MS);
    });
  }

  connect();

  return () => {
    closed = true;
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
    }
    socket?.close();
  };
}

function apiWebSocketUrl(path: string) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}
