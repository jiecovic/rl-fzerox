import type { RunSession } from "@/app/workspace/types";
import type { ManagedRun } from "@/shared/api/contract";

export function openRunSession(current: RunSession[], run: ManagedRun) {
  const sessionId = `run:${run.id}` as const;
  if (current.some((session) => session.sessionId === sessionId)) {
    return current;
  }
  return [...current, { runId: run.id, sessionId, title: run.name }];
}

export function closeRunSession(current: RunSession[], sessionId: RunSession["sessionId"]) {
  const closingIndex = current.findIndex((session) => session.sessionId === sessionId);
  if (closingIndex === -1) {
    return null;
  }
  const remaining = current.filter((session) => session.sessionId !== sessionId);
  const fallbackSession =
    remaining[closingIndex - 1] ?? remaining[closingIndex] ?? remaining.at(-1) ?? null;
  return {
    fallbackTabId: fallbackSession?.sessionId ?? "runs",
    remaining,
  } as const;
}

export function closeRunSessionsForRuns(current: RunSession[], runIds: readonly string[]) {
  if (runIds.length === 0) {
    return null;
  }
  const runIdSet = new Set(runIds);
  const removedSessionIds = new Set(
    current.filter((session) => runIdSet.has(session.runId)).map((session) => session.sessionId),
  );
  if (removedSessionIds.size === 0) {
    return null;
  }
  return {
    remaining: current.filter((session) => !removedSessionIds.has(session.sessionId)),
    removedSessionIds,
  } as const;
}
