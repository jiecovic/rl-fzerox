// web/run-manager/src/shared/api/client/resources/saveGames.ts
import { getJson, parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  type CourseSetupScope,
  createSaveGameResponseSchema,
  type ManagedSaveGame,
  type ManagedSaveUnlockTarget,
  openSaveGameDirectoryResponseSchema,
  type PolicyPlaybackMode,
  type SavePolicyArtifact,
  saveGamesResponseSchema,
  upsertSaveCourseSetupResponseSchema,
  type WatchDevice,
  type WatchRenderer,
  watchRunResponseSchema,
} from "@/shared/api/contract";

export async function fetchSaveGames(): Promise<ManagedSaveGame[]> {
  const payload = parseApiPayload(saveGamesResponseSchema, await getJson("/api/save-games"));
  return payload.save_games;
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

export async function openSaveGameDirectory(saveGameId: string): Promise<void> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/open-dir`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  parseApiPayload(openSaveGameDirectoryResponseSchema, await parseJson(response));
}

export async function startCareerModeRunner(
  saveGameId: string,
  device: WatchDevice,
  renderer: WatchRenderer | null,
  attemptSeed: string | null,
  policyMode: PolicyPlaybackMode,
  target: ManagedSaveUnlockTarget | null = null,
): Promise<"started" | "already_running"> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/runner`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device,
      renderer,
      attempt_seed: attemptSeed === null ? null : Number(attemptSeed),
      policy_mode: policyMode,
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
  scope,
  vehicleId,
}: {
  courseId?: string | null;
  cupId?: string | null;
  difficulty?: string | null;
  engineSettingRawValue: number;
  policyArtifact: SavePolicyArtifact;
  policyRunId: string;
  saveGameId: string;
  scope: CourseSetupScope;
  vehicleId: string;
}): Promise<ManagedSaveGame> {
  const response = await fetch(`/api/save-games/${encodeURIComponent(saveGameId)}/course-setups`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scope,
      difficulty: difficulty ?? null,
      cup_id: cupId ?? null,
      course_id: courseId ?? null,
      policy_run_id: policyRunId,
      policy_artifact: policyArtifact,
      vehicle_id: vehicleId,
      engine_setting_raw_value: engineSettingRawValue,
    }),
  });
  const payload = parseApiPayload(upsertSaveCourseSetupResponseSchema, await parseJson(response));
  return payload.save_game;
}
