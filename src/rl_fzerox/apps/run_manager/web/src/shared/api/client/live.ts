// src/rl_fzerox/apps/run_manager/web/src/shared/api/client/live.ts

import { parseApiPayload } from "@/shared/api/client/http";
import { type ManagedRun, runsLiveMessageSchema } from "@/shared/api/contract";

const RUN_LIVE_RECONNECT_DELAY_MS = 1_500;

export interface RunLiveSubscriptionOptions {
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
  onRuns: (runs: ManagedRun[]) => void;
}

export function subscribeRunLiveUpdates({
  onConnectionChange,
  onError,
  onRuns,
}: RunLiveSubscriptionOptions): () => void {
  let closed = false;
  let reconnectTimer: number | null = null;
  let socket: WebSocket | null = null;

  function connect() {
    if (closed) {
      return;
    }
    socket = new WebSocket(apiWebSocketUrl("/api/runs/live"));
    socket.addEventListener("open", () => {
      onConnectionChange?.(true);
    });
    socket.addEventListener("message", (event) => {
      try {
        const parsed = parseApiPayload(runsLiveMessageSchema, JSON.parse(String(event.data)));
        onRuns(parsed.runs);
      } catch (caught) {
        onError?.(caught instanceof Error ? caught : new Error("invalid run update payload"));
      }
    });
    socket.addEventListener("close", () => {
      socket = null;
      if (closed) {
        return;
      }
      onConnectionChange?.(false);
      reconnectTimer = window.setTimeout(connect, RUN_LIVE_RECONNECT_DELAY_MS);
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
