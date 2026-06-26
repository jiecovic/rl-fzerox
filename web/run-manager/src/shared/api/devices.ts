// web/run-manager/src/shared/api/devices.ts
import type { ConfigMetadata, WatchDevice } from "@/shared/api/contract";

export function runtimeDeviceOptions(metadata: ConfigMetadata | null): readonly WatchDevice[] {
  return metadata?.runtime.cuda_available === true ? ["cuda", "cpu"] : ["cpu"];
}

export function normalizeRuntimeDevice(
  device: WatchDevice,
  metadata: ConfigMetadata | null,
): WatchDevice {
  return device === "cuda" && metadata?.runtime.cuda_available !== true ? "cpu" : device;
}
