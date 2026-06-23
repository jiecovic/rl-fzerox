// web/run-manager/src/widgets/saveGameWorkspace/model.ts
import type { SaveGameSession } from "@/app/workspace/types";
import { type UnlockTargetSummary, unlockTargetKey } from "@/entities/saveGame/model";
import { parseAttemptSeed } from "@/features/careerRunner/model/runnerSeed";
import {
  resolveSavedCourseSetup,
  resolveSavedCupSetup,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import type {
  ConfigMetadata,
  ManagedSaveGame,
  ManagedSaveUnlockTarget,
} from "@/shared/api/contract";

const EMPTY_STRING_SET: ReadonlySet<string> = new Set<string>();

export function launchableTargetStatus(target: ManagedSaveUnlockTarget): boolean {
  return target.status === "pending" || target.status === "succeeded";
}

export function parseTargetClearGoal(text: string): number {
  const parsed = Number.parseInt(text.trim(), 10);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, parsed);
}

export function resolveLaunchCupVehicleId(
  cupSetups: ManagedSaveGame["cup_setups"],
  unlockedVehicleIds: readonly string[],
  target: ManagedSaveUnlockTarget,
): string | null {
  return resolveSavedCupSetup(cupSetups, target)?.vehicle_id ?? unlockedVehicleIds[0] ?? null;
}

export function startableUnlockTargetKeys({
  builtInCourses,
  courseSetups,
  cupSetups,
  disabled,
  targets,
  unlockedVehicleIds,
}: {
  builtInCourses: ConfigMetadata["built_in_courses"];
  courseSetups: ManagedSaveGame["course_setups"];
  cupSetups: ManagedSaveGame["cup_setups"];
  disabled: boolean;
  targets: readonly ManagedSaveUnlockTarget[];
  unlockedVehicleIds: readonly string[];
}): ReadonlySet<string> {
  if (disabled) {
    return EMPTY_STRING_SET;
  }
  const keys = new Set<string>();
  for (const target of targets) {
    if (
      launchableTargetStatus(target) &&
      resolveSavedCourseSetup(courseSetups, target, builtInCourses) !== null &&
      resolveLaunchCupVehicleId(cupSetups, unlockedVehicleIds, target) !== null
    ) {
      keys.add(unlockTargetKey(target));
    }
  }
  return keys;
}

export function targetSummaryHasStarted(summary: UnlockTargetSummary): boolean {
  return summary.succeeded > 0 || summary.failed > 0 || summary.skipped > 0;
}

export function runnerSettingsDirty(saveGame: ManagedSaveGame, session: SaveGameSession): boolean {
  const settings = saveGame.runner_settings;
  const attemptSeed = parseAttemptSeed(session.attemptSeedText);
  const persistedAttemptSeed =
    settings.attempt_seed === null ? null : String(settings.attempt_seed);
  return (
    attemptSeed === "invalid" ||
    persistedAttemptSeed !== attemptSeed ||
    settings.device !== session.runnerDevice ||
    settings.renderer !== session.runnerRenderer ||
    settings.policy_mode !== session.policyMode ||
    settings.recording_enabled !== session.recordingEnabled ||
    settings.recording_input_hud_enabled !== session.recordingInputHudEnabled ||
    settings.recording_upscale_factor !== session.recordingUpscaleFactor ||
    settings.recording_path !== null ||
    settings.target_restart_on_retire !== session.perfectRun ||
    settings.target_clear_goal !== parseTargetClearGoal(session.targetClearGoalText) ||
    settings.keep_failed_recordings !== session.keepFailedPerfectRunVideos ||
    settings.reload_policy_between_attempts !== session.reloadPolicyBetweenAttempts
  );
}
