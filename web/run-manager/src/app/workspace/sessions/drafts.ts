// web/run-manager/src/app/workspace/sessions/drafts.ts
import { draftForkSource, normalizeDraftTabTitle } from "@/app/workspace/model";
import type { DraftEditorSession } from "@/app/workspace/types";
import type { ManagedDraft } from "@/shared/api/contract";

export function openDraftSession(current: DraftEditorSession[], draft: ManagedDraft) {
  const sessionId = `editor:${draft.id}` as const;
  if (current.some((session) => session.sessionId === sessionId)) {
    return current;
  }
  return [
    ...current,
    {
      currentConfig: draft.config,
      currentDraftName: draft.name,
      draftId: draft.id,
      forkSource: draftForkSource(draft),
      initialDraftName: draft.name,
      initialConfig: null,
      loadedDraft: draft,
      sessionId,
      title: draft.name,
    },
  ];
}

export function createDraftSession(
  current: DraftEditorSession[],
  {
    forkSource = null,
    initialConfig = null,
    initialDraftName,
    sessionId,
  }: {
    forkSource?: DraftEditorSession["forkSource"];
    initialConfig?: DraftEditorSession["initialConfig"];
    initialDraftName: string;
    sessionId: DraftEditorSession["sessionId"];
  },
) {
  return [
    ...current,
    {
      currentConfig: initialConfig,
      currentDraftName: initialDraftName,
      draftId: null,
      forkSource,
      initialDraftName,
      initialConfig,
      loadedDraft: null,
      sessionId,
      title: initialDraftName,
    },
  ];
}

export function patchDraftSessions(
  current: DraftEditorSession[],
  sessionId: DraftEditorSession["sessionId"],
  patch: Partial<Omit<DraftEditorSession, "sessionId">>,
) {
  let changed = false;
  const next = current.map((session) => {
    if (session.sessionId !== sessionId) {
      return session;
    }
    const updated = { ...session, ...patch };
    changed =
      updated.currentDraftName !== session.currentDraftName ||
      updated.currentConfig !== session.currentConfig ||
      updated.title !== session.title ||
      updated.initialDraftName !== session.initialDraftName ||
      updated.initialConfig !== session.initialConfig ||
      updated.forkSource !== session.forkSource ||
      updated.draftId !== session.draftId ||
      updated.loadedDraft !== session.loadedDraft;
    return changed ? updated : session;
  });
  return changed ? next : current;
}

export function normalizeDraftSessionTitle(title: string) {
  return normalizeDraftTabTitle(title);
}

export function closeDraftSession(
  current: DraftEditorSession[],
  sessionId: DraftEditorSession["sessionId"],
) {
  const closingIndex = current.findIndex((session) => session.sessionId === sessionId);
  if (closingIndex === -1) {
    return null;
  }
  const remaining = current.filter((session) => session.sessionId !== sessionId);
  const fallbackSession =
    remaining[closingIndex - 1] ?? remaining[closingIndex] ?? remaining.at(-1) ?? null;
  return {
    fallbackTabId: fallbackSession?.sessionId ?? "drafts",
    remaining,
  } as const;
}

export function closeDraftSessionsForDraft(current: DraftEditorSession[], draftId: string) {
  const removedSessionIds = new Set(
    current.filter((session) => session.draftId === draftId).map((session) => session.sessionId),
  );
  if (removedSessionIds.size === 0) {
    return null;
  }
  return {
    remaining: current.filter((session) => !removedSessionIds.has(session.sessionId)),
    removedSessionIds,
  } as const;
}

export function closeDraftSessionsForSourceRuns(
  current: DraftEditorSession[],
  runIds: readonly string[],
) {
  if (runIds.length === 0) {
    return null;
  }
  const sourceRunIds = new Set(runIds);
  const removedSessionIds = new Set(
    current
      .filter((session) => {
        const sourceRunId = session.loadedDraft?.source_run_id ?? session.forkSource?.runId ?? null;
        return sourceRunId !== null && sourceRunIds.has(sourceRunId);
      })
      .map((session) => session.sessionId),
  );
  if (removedSessionIds.size === 0) {
    return null;
  }
  return {
    remaining: current.filter((session) => !removedSessionIds.has(session.sessionId)),
    removedSessionIds,
  } as const;
}
