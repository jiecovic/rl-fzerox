// web/run-manager/src/shared/api/renderers.ts
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

export type RendererName = ManagedRunConfig["environment"]["renderer"];

export function rendererNames(
  metadata: ConfigMetadata,
  selectedRenderer: RendererName,
): RendererName[] {
  const renderers = metadata.observation_source_geometries.map((geometry) => geometry.renderer);
  if (!renderers.includes(selectedRenderer)) {
    renderers.unshift(selectedRenderer);
  }
  return Array.from(new Set(renderers));
}
