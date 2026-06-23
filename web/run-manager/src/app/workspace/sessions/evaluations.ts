// web/run-manager/src/app/workspace/sessions/evaluations.ts
import { evaluationSessionId } from "@/app/workspace/model";
import type { EvaluationSession } from "@/app/workspace/types";
import type { ManagedEvaluation } from "@/shared/api/contract";

interface CloseEvaluationSessionResult {
  fallbackTabId: "evaluations";
  remaining: EvaluationSession[];
}

interface CloseEvaluationSessionsForEvaluationResult {
  remaining: EvaluationSession[];
  removedSessionIds: ReadonlySet<EvaluationSession["sessionId"]>;
}

export function openEvaluationSession(
  current: readonly EvaluationSession[],
  evaluation: ManagedEvaluation,
): EvaluationSession[] {
  const session: EvaluationSession = {
    evaluationId: evaluation.id,
    sessionId: evaluationSessionId(evaluation.id),
    title: evaluation.name,
  };
  if (current.some((candidate) => candidate.sessionId === session.sessionId)) {
    return current.map((candidate) =>
      candidate.sessionId === session.sessionId ? session : candidate,
    );
  }
  return [...current, session];
}

export function closeEvaluationSession(
  current: readonly EvaluationSession[],
  sessionId: EvaluationSession["sessionId"],
): CloseEvaluationSessionResult | null {
  const remaining = current.filter((session) => session.sessionId !== sessionId);
  if (remaining.length === current.length) {
    return null;
  }
  return { fallbackTabId: "evaluations", remaining };
}

export function closeEvaluationSessionsForEvaluation(
  current: readonly EvaluationSession[],
  evaluationId: string,
): CloseEvaluationSessionsForEvaluationResult | null {
  const removedSessionIds = new Set<EvaluationSession["sessionId"]>();
  const remaining = current.filter((session) => {
    if (session.evaluationId !== evaluationId) {
      return true;
    }
    removedSessionIds.add(session.sessionId);
    return false;
  });
  if (removedSessionIds.size === 0) {
    return null;
  }
  return { remaining, removedSessionIds };
}

export function renameEvaluationSession(
  current: readonly EvaluationSession[],
  evaluationId: string,
  title: string,
): EvaluationSession[] {
  return current.map((session) =>
    session.evaluationId === evaluationId ? { ...session, title } : session,
  );
}
