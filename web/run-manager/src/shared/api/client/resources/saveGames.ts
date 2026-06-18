// web/run-manager/src/shared/api/client/resources/saveGames.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  type CareerModeRunnerLaunchRequest,
  createSaveGameResponseSchema,
  deleteSaveGameResponseSchema,
  importSaveEngineTuningResponseSchema,
  type ManagedSaveGame,
  type ManagedSaveGameStatus,
  openSaveGameDirectoryResponseSchema,
  type SaveEngineTuningCourseSetupRecommendation,
  type SaveGameRunnerSettingsUpdateRequest,
  type SavePolicyArtifact,
  saveGameStatusResponseSchema,
  saveGamesResponseSchema,
  updateSaveGameRunnerSettingsResponseSchema,
  upsertSaveCourseSetupResponseSchema,
  watchRunResponseSchema,
} from "@/shared/api/contract";

export async function fetchSaveGames(): Promise<ManagedSaveGame[]> {
  const payload = parseApiPayload(saveGamesResponseSchema, await getJson("/api/save-games"));
  return payload.save_games;
}

export async function fetchSaveGameStatus(saveGameId: string): Promise<ManagedSaveGameStatus> {
  const payload = parseApiPayload(
    saveGameStatusResponseSchema,
    await getJson(`/api/save-games/${encodeURIComponent(saveGameId)}/status`),
  );
  return payload.save_game;
}

export async function createSaveGame(name: string): Promise<ManagedSaveGame> {
  const response = await fetch("/api/save-games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const payload = parseApiPayload(createSaveGameResponseSchema, await parseJson(response));
  return payload.save_game;
}

export async function renameSaveGame(saveGameId: string, name: string): Promise<ManagedSaveGame> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const payload = parseApiPayload(createSaveGameResponseSchema, await parseJson(response));
  return payload.save_game;
}

export async function updateSaveGameRunnerSettings({
  attemptSeed,
  device,
  policyMode,
  recordingEnabled,
  recordingInputHudEnabled,
  recordingUpscaleFactor,
  recordingPath,
  renderer,
  saveGameId,
  targetRestartOnRetire,
  targetClearGoal,
  keepFailedRecordings,
  reloadPolicyBetweenAttempts,
}: SaveGameRunnerSettingsUpdateRequest): Promise<ManagedSaveGame> {
  const response = await fetch(
    `/api/save-games/${encodeURIComponent(saveGameId)}/runner-settings`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device,
        renderer,
        attempt_seed: attemptSeed === null ? null : Number(attemptSeed),
        policy_mode: policyMode,
        recording_enabled: recordingEnabled,
        recording_input_hud_enabled: recordingInputHudEnabled,
        recording_upscale_factor: recordingUpscaleFactor,
        recording_path: recordingPath,
        target_restart_on_retire: targetRestartOnRetire,
        target_clear_goal: targetClearGoal,
        keep_failed_recordings: keepFailedRecordings,
        reload_policy_between_attempts: reloadPolicyBetweenAttempts,
      }),
    },
  );
  const payload = parseApiPayload(
    updateSaveGameRunnerSettingsResponseSchema,
    await parseJson(response),
  );
  return payload.save_game;
}

export async function deleteSaveGame(saveGameId: string): Promise<void> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}`, {
    method: "DELETE",
  });
  parseApiPayload(deleteSaveGameResponseSchema, await parseJson(response));
}

export async function openSaveGameDirectory(saveGameId: string): Promise<void> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/open-dir`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  parseApiPayload(openSaveGameDirectoryResponseSchema, await parseJson(response));
}

export async function startCareerModeRunner({
  attemptSeed,
  device,
  policyMode,
  recordingEnabled,
  recordingInputHudEnabled,
  recordingUpscaleFactor,
  recordingPath,
  renderer,
  saveGameId,
  singleTarget,
  perfectRun,
  keepFailedRecordings,
  reloadPolicyBetweenAttempts,
  targetClearGoal,
  target,
}: CareerModeRunnerLaunchRequest): Promise<"started" | "already_running"> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/runner`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device,
      renderer,
      attempt_seed: attemptSeed === null ? null : Number(attemptSeed),
      policy_mode: policyMode,
      recording_enabled: recordingEnabled,
      recording_input_hud_enabled: recordingInputHudEnabled,
      recording_upscale_factor: recordingUpscaleFactor,
      recording_path: recordingEnabled ? recordingPath : null,
      single_target: singleTarget,
      perfect_run: perfectRun,
      keep_failed_recordings: keepFailedRecordings,
      reload_policy_between_attempts: reloadPolicyBetweenAttempts,
      target_clear_goal: targetClearGoal,
      ...(target === null
        ? {}
        : {
            target_kind: target.kind,
            difficulty: target.difficulty,
            cup_id: target.cup_id,
            course_id: target.course_id,
          }),
    }),
  });
  const payload = parseApiPayload(watchRunResponseSchema, await parseJson(response));
  return payload.status;
}

export async function upsertSaveCourseSetup({
  courseId,
  cupId,
  difficulty,
  engineSettingRawValue,
  policyArtifact,
  policyRunId,
  saveGameId,
}: {
  courseId?: string | null;
  cupId?: string | null;
  difficulty?: string | null;
  engineSettingRawValue: number;
  policyArtifact: SavePolicyArtifact;
  policyRunId: string;
  saveGameId: string;
}): Promise<ManagedSaveGame> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/course-setups`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      difficulty: difficulty ?? null,
      cup_id: cupId ?? null,
      course_id: courseId ?? null,
      policy_run_id: policyRunId,
      policy_artifact: policyArtifact,
      engine_setting_raw_value: engineSettingRawValue,
    }),
  });
  const payload = parseApiPayload(upsertSaveCourseSetupResponseSchema, await parseJson(response));
  return payload.save_game;
}

export async function importSaveEngineTuning({
  courseSetups,
  policyArtifact,
  policyRunId,
  saveGameId,
}: {
  courseSetups: readonly {
    courseId: string;
    cupId: string;
    difficulty?: string | null;
    vehicleId: string;
  }[];
  policyArtifact: SavePolicyArtifact;
  policyRunId: string;
  saveGameId: string;
}): Promise<readonly SaveEngineTuningCourseSetupRecommendation[]> {
  const response = await fetch(
    `/api/save-games/${encodeURIComponent(saveGameId)}/course-setups/import-engine-tuning`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        course_setups: courseSetups.map((setup) => ({
          difficulty: setup.difficulty ?? null,
          cup_id: setup.cupId,
          course_id: setup.courseId,
          vehicle_id: setup.vehicleId,
        })),
        policy_artifact: policyArtifact,
        policy_run_id: policyRunId,
      }),
    },
  );
  const payload = parseApiPayload(importSaveEngineTuningResponseSchema, await parseJson(response));
  return payload.recommendations;
}

export async function upsertSaveCupSetup({
  cupId,
  difficulty,
  saveGameId,
  vehicleId,
}: {
  cupId: string;
  difficulty?: string | null;
  saveGameId: string;
  vehicleId: string;
}): Promise<ManagedSaveGame> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/cup-setups`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      cup_id: cupId,
      difficulty: difficulty ?? null,
      vehicle_id: vehicleId,
    }),
  });
  const payload = parseApiPayload(upsertSaveCourseSetupResponseSchema, await parseJson(response));
  return payload.save_game;
}
