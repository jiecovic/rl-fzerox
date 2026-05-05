import type { DraftEditorSession, ForkSource } from "@/app/workspace/types";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";

export function allKnownWorkspaceNames({
  drafts,
  draftEditors,
  runs,
}: {
  drafts: readonly ManagedDraft[];
  draftEditors: readonly DraftEditorSession[];
  runs: readonly ManagedRun[];
}) {
  const names = new Set<string>();
  for (const draft of drafts) {
    names.add(draft.name);
  }
  for (const run of runs) {
    names.add(run.name);
  }
  for (const session of draftEditors) {
    names.add(session.title);
  }
  return names;
}

export function reservedWorkspaceNamesForSession({
  drafts,
  draftEditors,
  sessionId,
}: {
  drafts: readonly ManagedDraft[];
  draftEditors: readonly DraftEditorSession[];
  sessionId: DraftEditorSession["sessionId"];
}) {
  const names = new Set<string>();
  for (const draft of drafts) {
    names.add(draft.name);
  }
  for (const session of draftEditors) {
    if (session.sessionId !== sessionId) {
      names.add(session.title);
    }
  }
  return [...names];
}

export function resolveForkSourceRunLabel(source: ForkSource | null, runs: readonly ManagedRun[]) {
  if (source === null) {
    return null;
  }
  const run = runs.find((candidate) => candidate.id === source.runId) ?? null;
  return run?.name ?? source.runId;
}
