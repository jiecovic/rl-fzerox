import { fetchDrafts, fetchRuns, fetchTemplates } from "@/shared/api/client";
import type { ManagedDraft, ManagedRun, ManagedTemplate } from "@/shared/api/contract";

export interface ManagerData {
  drafts: ManagedDraft[];
  runs: ManagedRun[];
  templates: ManagedTemplate[];
}

export async function loadManagerData(): Promise<ManagerData> {
  const [templates, drafts, runs] = await Promise.all([
    fetchTemplates(),
    fetchDrafts(),
    fetchRuns(),
  ]);
  return { drafts, runs, templates };
}
