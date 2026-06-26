// web/run-manager/src/app/workspace/model.ts
import type {
  DraftEditorSession,
  EvaluationSession,
  ForkSource,
  PrimaryWorkspaceTabId,
  RunSession,
  SaveGameSession,
  WorkspaceTab,
  WorkspaceTabId,
} from "@/app/workspace/types";
import { isPinnedRun } from "@/entities/run/model/runtime";
import type {
  ManagedDraft,
  ManagedEvaluation,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
  ManagedSaveGame,
  ManagedSaveGameStatus,
} from "@/shared/api/contract";

export function editorSessionId(seed: string): `editor:${string}` {
  return `editor:${seed}`;
}

export function runSessionId(seed: string): `run:${string}` {
  return `run:${seed}`;
}

export function evaluationSessionId(seed: string): `evaluation:${string}` {
  return `evaluation:${seed}`;
}

export function saveGameSessionId(seed: string): `save-game:${string}` {
  return `save-game:${seed}`;
}

export const primaryWorkspaceTabs = [
  { id: "drafts", icon: "draft", label: "Drafts" },
  { id: "runs", icon: "run", label: "Runs" },
  { id: "checkpoints", icon: "checkpoint", label: "Checkpoints", shortLabel: "Ckpt" },
  { id: "charts", icon: "charts", label: "Charts" },
  { id: "evaluations", icon: "evaluation", label: "Evaluations", shortLabel: "Eval" },
  { id: "save-games", icon: "career", label: "Career Mode", shortLabel: "Career" },
] satisfies readonly (WorkspaceTab & { id: PrimaryWorkspaceTabId })[];

export function activePrimaryWorkspaceTabId(id: WorkspaceTabId): PrimaryWorkspaceTabId {
  switch (id) {
    case "drafts":
    case "runs":
    case "checkpoints":
    case "charts":
    case "evaluations":
    case "save-games":
      return id;
    default:
      if (id.startsWith("editor:")) {
        return "drafts";
      }
      if (id.startsWith("run:")) {
        return "runs";
      }
      if (id.startsWith("evaluation:")) {
        return "evaluations";
      }
      return "save-games";
  }
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

export function nextAvailableSaveGameName(takenNames: Iterable<string>) {
  return nextCounterName(defaultCareerSaveName(new Date()), takenNames);
}

function defaultCareerSaveName(createdAt: Date) {
  const iso = createdAt.toISOString();
  const date = iso.slice(0, 10).replaceAll("-", "");
  const time = iso.slice(11, 16).replace(":", "");
  return `career ${date}-${time}`;
}

export function nextForkDraftName(
  sourceRun: ManagedRun,
  runs: readonly ManagedRun[],
  takenNames: Iterable<string>,
) {
  return nextCounterName(forkDraftBaseName(sourceRun, runs), takenNames);
}

function forkDraftBaseName(sourceRun: ManagedRun, runs: readonly ManagedRun[]) {
  const lineageRuns = runs.filter((run) => run.lineage_id === sourceRun.lineage_id);
  const baseRun = [...lineageRuns].sort(compareLineageBaseRuns).at(0) ?? sourceRun;
  return stripForkSuffixes(baseRun.name);
}

function compareLineageBaseRuns(left: ManagedRun, right: ManagedRun) {
  const leftRootRank = lineageBaseRank(left);
  const rightRootRank = lineageBaseRank(right);
  if (leftRootRank !== rightRootRank) {
    return leftRootRank - rightRootRank;
  }
  if (left.lineage_step_offset !== right.lineage_step_offset) {
    return left.lineage_step_offset - right.lineage_step_offset;
  }
  if (left.created_at !== right.created_at) {
    return left.created_at.localeCompare(right.created_at);
  }
  return left.id.localeCompare(right.id);
}

function lineageBaseRank(run: ManagedRun) {
  return run.parent_run_id === null && run.source_run_id === null ? 0 : 1;
}

function stripForkSuffixes(name: string) {
  let current = normalizeDraftTabTitle(name);
  while (true) {
    const withoutCounter = current.replace(/\s+\d+$/, "").trim();
    const withoutFork = withoutCounter.replace(/\s+(?:best\s+)?fork$/i, "").trim();
    const next = withoutFork.length > 0 ? withoutFork : current;
    if (next === current) {
      return current;
    }
    current = next;
  }
}

function nextCounterName(baseName: string, takenNames: Iterable<string>) {
  const trimmedBaseName = normalizeDraftTabTitle(baseName);
  const normalizedBaseName = trimmedBaseName.toLowerCase();
  let highestCounter = 0;

  for (const name of takenNames) {
    const normalizedName = name.trim().toLowerCase();
    if (normalizedName === normalizedBaseName) {
      highestCounter = Math.max(highestCounter, 1);
      continue;
    }
    const prefix = `${normalizedBaseName} `;
    if (!normalizedName.startsWith(prefix)) {
      continue;
    }
    const suffix = normalizedName.slice(prefix.length);
    if (/^\d+$/.test(suffix)) {
      highestCounter = Math.max(highestCounter, Number(suffix));
    }
  }

  return highestCounter === 0 ? trimmedBaseName : `${trimmedBaseName} ${highestCounter + 1}`;
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

export function upsertSaveGame(current: ManagedSaveGame[], nextSaveGame: ManagedSaveGame) {
  const withoutPrevious = current.filter((saveGame) => saveGame.id !== nextSaveGame.id);
  return [nextSaveGame, ...withoutPrevious].sort(compareSaveGames);
}

export function upsertEvaluation(current: ManagedEvaluation[], nextEvaluation: ManagedEvaluation) {
  const withoutPrevious = current.filter((evaluation) => evaluation.id !== nextEvaluation.id);
  return [nextEvaluation, ...withoutPrevious].sort(compareEvaluations);
}

export function compareEvaluations(left: ManagedEvaluation, right: ManagedEvaluation) {
  if (left.created_at !== right.created_at) {
    return right.created_at.localeCompare(left.created_at);
  }
  return right.id.localeCompare(left.id);
}

export function upsertSaveGameStatus(current: ManagedSaveGame[], status: ManagedSaveGameStatus) {
  const existing = current.find((saveGame) => saveGame.id === status.id);
  if (existing === undefined) {
    return current;
  }
  if (sameSaveGameStatus(existing, status)) {
    return current;
  }
  return upsertSaveGame(current, {
    ...existing,
    ...status,
    attempts: existing.attempts,
    course_setups: existing.course_setups,
    cup_setups: existing.cup_setups,
  });
}

function sameSaveGameStatus(saveGame: ManagedSaveGame, status: ManagedSaveGameStatus): boolean {
  return (
    saveGame.name === status.name &&
    saveGame.status === status.status &&
    saveGame.runner_active === status.runner_active &&
    saveGame.save_path === status.save_path &&
    saveGame.created_at === status.created_at &&
    saveGame.last_finished_at === status.last_finished_at &&
    JSON.stringify(saveGame.runner_settings) === JSON.stringify(status.runner_settings) &&
    JSON.stringify(saveGame.unlock_progress) === JSON.stringify(status.unlock_progress)
  );
}

export function runSummaryFromDetail(run: ManagedRunDetail): ManagedRun {
  const { config: _config, ...summary } = run;
  void _config;
  return summary;
}

export function compareRuns(left: ManagedRun, right: ManagedRun) {
  const leftPinned = isPinnedRun(left);
  const rightPinned = isPinnedRun(right);
  if (leftPinned !== rightPinned) {
    return leftPinned ? -1 : 1;
  }
  if (left.created_at !== right.created_at) {
    return right.created_at.localeCompare(left.created_at);
  }
  return right.id.localeCompare(left.id);
}

export function compareSaveGames(left: ManagedSaveGame, right: ManagedSaveGame) {
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
    copyAltBaselines: true,
    sourceEngineTunerBackend: null,
    sourceEngineTuning: null,
    sourceEngineTuningKnown: false,
  };
}

export function forkInitialConfig(config: ManagedRunConfig): ManagedRunConfig {
  return {
    ...config,
    policy: {
      ...config.policy,
      gas_on_logit: 0,
      air_brake_on_logit: 0,
      spin_idle_logit: 0,
    },
  };
}

export function configuratorBaseConfigForDraftEditor(
  defaultConfig: ManagedRunConfig,
  session: DraftEditorSession,
): ManagedRunConfig {
  if (session.forkSource !== null && session.initialConfig !== null) {
    return session.initialConfig;
  }
  return defaultConfig;
}

export function buildWorkspaceSessionTabs(
  draftEditors: readonly DraftEditorSession[],
  evaluationSessions: readonly EvaluationSession[],
  evaluations: readonly ManagedEvaluation[],
  runTabs: readonly RunSession[],
  runs: readonly ManagedRun[],
  saveGameSessions: readonly SaveGameSession[],
): WorkspaceTab[] {
  return [
    ...runTabs.map((session) => {
      const run = runs.find((candidate) => candidate.id === session.runId);
      return {
        id: session.sessionId,
        icon: "run" as const,
        label: `Run · ${run?.name ?? session.title}`,
        activity: run?.status === "running" ? ("running" as const) : undefined,
        closable: true,
        tone: "run" as const,
      };
    }),
    ...evaluationSessions.map((session) => {
      const evaluation = evaluations.find((candidate) => candidate.id === session.evaluationId);
      return {
        id: session.sessionId,
        icon: "evaluation" as const,
        label: `Eval · ${evaluation?.name ?? session.title}`,
        activity: evaluation?.status === "running" ? ("running" as const) : undefined,
        closable: true,
      };
    }),
    ...draftEditors.map((session) => ({
      id: session.sessionId,
      icon: "draft" as const,
      label: `${session.forkSource === null ? "Draft" : "Fork draft"} · ${session.title}`,
      closable: true,
      tone: "draft" as const,
    })),
    ...saveGameSessions.map((session) => ({
      id: session.sessionId,
      icon: "career" as const,
      label: `Career · ${session.title}`,
      closable: true,
    })),
  ];
}
