import type {
  DraftEditorSession,
  ForkSource,
  RunSession,
  WorkspaceTab,
} from "@/app/workspace/types";
import type { ManagedDraft, ManagedRun } from "@/shared/api/contract";

export function editorSessionId(seed: string): `editor:${string}` {
  return `editor:${seed}`;
}

export function runSessionId(seed: string): `run:${string}` {
  return `run:${seed}`;
}

export function normalizeDraftTabTitle(title: string) {
  const trimmed = title.trim();
  return trimmed.length > 0 ? trimmed : "New draft";
}

export function defaultDraftName() {
  return "ppo_allcups_recurrent";
}

export function nextAvailableDraftName(baseName: string, takenNames: Iterable<string>) {
  const normalizedTaken = new Set(
    [...takenNames].map((name) => name.trim().toLowerCase()).filter((name) => name.length > 0),
  );
  if (!normalizedTaken.has(baseName.toLowerCase())) {
    return baseName;
  }
  let suffix = 2;
  while (normalizedTaken.has(`${baseName} ${suffix}`.toLowerCase())) {
    suffix += 1;
  }
  return `${baseName} ${suffix}`;
}

export function upsertDraft(current: ManagedDraft[], nextDraft: ManagedDraft) {
  const withoutPrevious = current.filter((draft) => draft.id !== nextDraft.id);
  return [nextDraft, ...withoutPrevious].sort(compareDrafts);
}

export function compareDrafts(left: ManagedDraft, right: ManagedDraft) {
  if (left.updated_at !== right.updated_at) {
    return right.updated_at.localeCompare(left.updated_at);
  }
  return right.id.localeCompare(left.id);
}

export function upsertRun(current: ManagedRun[], nextRun: ManagedRun) {
  const withoutPrevious = current.filter((run) => run.id !== nextRun.id);
  return [nextRun, ...withoutPrevious].sort(compareRuns);
}

export function compareRuns(left: ManagedRun, right: ManagedRun) {
  if (left.created_at !== right.created_at) {
    return right.created_at.localeCompare(left.created_at);
  }
  return right.id.localeCompare(left.id);
}

export function draftForkSource(draft: ManagedDraft): ForkSource | null {
  if (draft.source_run_id === null || draft.source_artifact === null) {
    return null;
  }
  return {
    runId: draft.source_run_id,
    artifact: draft.source_artifact,
  };
}

export function buildWorkspaceTabs(
  draftEditors: readonly DraftEditorSession[],
  runTabs: readonly RunSession[],
  runs: readonly ManagedRun[],
): WorkspaceTab[] {
  return [
    { id: "drafts", label: "Drafts" },
    { id: "runs", label: "Runs" },
    { id: "charts", label: "Charts" },
    ...runTabs.map((session) => ({
      id: session.sessionId,
      label: `Run · ${runs.find((run) => run.id === session.runId)?.name ?? session.title}`,
      closable: true,
      tone: "run" as const,
    })),
    ...draftEditors.map((session) => ({
      id: session.sessionId,
      label: `${session.forkSource === null ? "Draft" : "Fork draft"} · ${session.title}`,
      closable: true,
      tone: "draft" as const,
    })),
  ];
}
