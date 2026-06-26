// web/run-manager/src/app/runtimeAssets.ts
import type { ConfigMetadata } from "@/shared/api/contract";

export interface RuntimeAssetNotice {
  key: string;
  message: string;
}

export function runtimeAssetNotice(metadata: ConfigMetadata | null): RuntimeAssetNotice | null {
  const missingAssets = metadata?.runtime_assets.filter((asset) => !asset.exists) ?? [];
  if (missingAssets.length === 0) {
    return null;
  }
  const assetList = missingAssets.map((asset) => `${asset.label} (${asset.path})`).join(", ");
  const prefix = missingAssets.length === 1 ? "Missing runtime asset" : "Missing runtime assets";
  return {
    key: missingAssets.map((asset) => `${asset.id}:${asset.path}`).join("|"),
    message:
      `${prefix}: ${assetList}. ` +
      "Add them under local/, then refresh this page before launching training, watch, or evaluation.",
  };
}
