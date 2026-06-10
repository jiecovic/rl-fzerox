// web/run-manager/src/entities/saveGame/model.ts
import type {
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
  SaveUnlockTargetStatus,
} from "@/shared/api/contract";

export interface UnlockTargetSummary {
  failed: number;
  locked: number;
  pending: number;
  skipped: number;
  succeeded: number;
  total: number;
}

export function summarizeSaveGameTargets(saveGame: ManagedSaveGame): UnlockTargetSummary {
  return summarizeTargets(saveGame.unlock_progress?.targets ?? []);
}

export function unlockCompletionFraction(summary: UnlockTargetSummary): number {
  if (summary.total === 0) {
    return 0;
  }
  return summary.succeeded / summary.total;
}

export function countSaveGameAttempts(saveGame: ManagedSaveGame): number {
  return saveGame.attempts.length;
}

export function countRunningAttempts(saveGame: ManagedSaveGame): number {
  return saveGame.attempts.filter((attempt) => attempt.status === "running").length;
}

export function careerSaveHasStarted(saveGame: ManagedSaveGame): boolean {
  const targetSummary = summarizeSaveGameTargets(saveGame);
  return targetSummary.succeeded > 0 || targetSummary.failed > 0 || targetSummary.skipped > 0;
}

export function nextUnlockTarget(saveGame: ManagedSaveGame): ManagedSaveUnlockTarget | null {
  return saveGame.unlock_progress?.next_target ?? null;
}

export function formatUnlockTarget(target: ManagedSaveUnlockTarget): string {
  const parts = [target.difficulty, target.cup_id, target.course_id]
    .map(formatTargetPart)
    .filter((value): value is string => value !== null);
  return parts.length === 0 ? target.label : `${target.label}: ${parts.join(" / ")}`;
}

export function formatUnlockTargetStatus(status: SaveUnlockTargetStatus): string {
  return titleizeIdentifier(status);
}

export function unlockTargetStatusClass(status: SaveUnlockTargetStatus): string {
  switch (status) {
    case "succeeded":
      return "border-emerald-300/40 bg-emerald-400/10 text-emerald-200";
    case "failed":
      return "border-rose-300/40 bg-rose-400/10 text-rose-200";
    case "skipped":
      return "border-amber-300/40 bg-amber-400/10 text-amber-200";
    case "locked":
      return "border-app-border bg-app-surface text-app-muted opacity-60";
    case "pending":
      return "border-app-border bg-app-surface-muted text-app-muted";
  }
}

export function titleizeIdentifier(value: string): string {
  return value
    .split("_")
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

function summarizeTargets(targets: readonly ManagedSaveUnlockTarget[]): UnlockTargetSummary {
  const summary: UnlockTargetSummary = {
    failed: 0,
    locked: 0,
    pending: 0,
    skipped: 0,
    succeeded: 0,
    total: targets.length,
  };
  for (const target of targets) {
    summary[target.status] += 1;
  }
  return summary;
}

function formatTargetPart(value: string | null): string | null {
  return value === null ? null : titleizeIdentifier(value);
}
