import { fetchConfigMetadata, fetchDrafts, fetchRuns, fetchTemplates } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedTemplate,
} from "@/shared/api/contract";

export interface ManagerData {
  drafts: ManagedDraft[];
  metadata: ConfigMetadata;
  runs: ManagedRun[];
  templates: ManagedTemplate[];
}

export async function loadManagerData(): Promise<ManagerData> {
  const [templates, drafts, runs, metadata] = await Promise.all([
    fetchTemplates(),
    fetchDrafts(),
    fetchRuns(),
    fetchConfigMetadata(),
  ]);
  return { drafts, metadata, runs, templates };
}
