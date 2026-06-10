// web/run-manager/src/widgets/configurator/draftName.ts
import type { ManagedDraft, ManagedRunConfig } from "@/shared/api/contract";

export function configuratorDraftName(
  baseConfig: ManagedRunConfig,
  initialDraftName: string | undefined,
  loadedDraft: ManagedDraft | null,
) {
  return loadedDraft?.name ?? initialDraftName ?? defaultDraftName(baseConfig);
}

function defaultDraftName(_: ManagedRunConfig) {
  return "ppo_allcups_recurrent";
}
