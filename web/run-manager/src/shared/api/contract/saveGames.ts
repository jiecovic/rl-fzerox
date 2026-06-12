// web/run-manager/src/shared/api/contract/saveGames.ts
import { z } from "zod";

import type { WatchDevice, WatchRenderer } from "@/shared/api/contract/enums";

export const saveGameStatusSchema = z.enum(["created", "running", "paused", "finished", "failed"]);

export const saveUnlockInspectionStatusSchema = z.enum(["not_inspected", "inspected"]);

export const saveUnlockTargetStatusSchema = z.enum([
  "pending",
  "locked",
  "succeeded",
  "failed",
  "skipped",
]);

export const saveAttemptStatusSchema = z.enum(["running", "succeeded", "failed"]);

export const savePolicyArtifactSchema = z.enum(["latest", "best"]);

export const policyPlaybackModeSchema = z.enum(["deterministic", "stochastic"]);

export const managedSaveCourseSetupSchema = z.object({
  id: z.string(),
  save_game_id: z.string(),
  difficulty: z.string().nullable(),
  cup_id: z.string().nullable(),
  course_id: z.string().nullable(),
  policy_run_id: z.string(),
  policy_artifact: savePolicyArtifactSchema,
  engine_setting_raw_value: z.number().int().min(0).max(100),
  created_at: z.string(),
  updated_at: z.string(),
});

export const managedSaveCupSetupSchema = z.object({
  id: z.string(),
  save_game_id: z.string(),
  difficulty: z.string().nullable(),
  cup_id: z.string(),
  vehicle_id: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const managedSaveUnlockTargetSchema = z.object({
  sequence_index: z.number(),
  kind: z.string(),
  status: saveUnlockTargetStatusSchema,
  label: z.string(),
  difficulty: z.string().nullable(),
  cup_id: z.string().nullable(),
  course_id: z.string().nullable(),
});

export const managedSaveUnlockProgressSchema = z.object({
  inspection_status: saveUnlockInspectionStatusSchema,
  completed_count: z.number(),
  total_count: z.number(),
  unlocked_vehicle_count: z.number(),
  unlocked_vehicle_ids: z.array(z.string()),
  next_target: managedSaveUnlockTargetSchema.nullable(),
  targets: z.array(managedSaveUnlockTargetSchema),
});

export const managedSaveAttemptSchema = z.object({
  id: z.string(),
  save_game_id: z.string(),
  target_kind: z.string().nullable(),
  status: saveAttemptStatusSchema,
  difficulty: z.string().nullable(),
  cup_id: z.string().nullable(),
  course_id: z.string().nullable(),
  started_at: z.string(),
  finished_at: z.string().nullable(),
  finish_position: z.number().nullable(),
  finish_time_s: z.number().nullable(),
  failure_reason: z.string().nullable(),
});

export const managedSaveGameSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: saveGameStatusSchema,
  runner_active: z.boolean(),
  save_path: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  last_finished_at: z.string().nullable(),
  unlock_progress: managedSaveUnlockProgressSchema.nullable(),
  attempts: z.array(managedSaveAttemptSchema),
  course_setups: z.array(managedSaveCourseSetupSchema),
  cup_setups: z.array(managedSaveCupSetupSchema),
});

export const saveGamesResponseSchema = z.object({
  save_games: z.array(managedSaveGameSchema),
});

export const createSaveGameResponseSchema = z.object({
  save_game: managedSaveGameSchema,
});

export const upsertSaveCourseSetupResponseSchema = z.object({
  save_game: managedSaveGameSchema,
});

export const importSaveEngineTuningResponseSchema = z.object({
  applied: z.array(
    z.object({
      setup_id: z.string(),
      difficulty: z.string().nullable(),
      cup_id: z.string().nullable(),
      course_id: z.string().nullable(),
      vehicle_id: z.string(),
      engine_setting_raw_value: z.number().int().min(0).max(100),
      mean_score: z.number().nullable(),
      attempts: z.number().int().nonnegative(),
    }),
  ),
  save_game: managedSaveGameSchema,
});

export const openSaveGameDirectoryResponseSchema = z.object({
  opened: z.boolean(),
});

export type ManagedSaveGame = z.infer<typeof managedSaveGameSchema>;
export type ManagedSaveAttempt = z.infer<typeof managedSaveAttemptSchema>;
export type ManagedSaveCourseSetup = z.infer<typeof managedSaveCourseSetupSchema>;
export type ManagedSaveCupSetup = z.infer<typeof managedSaveCupSetupSchema>;
export type ManagedSaveUnlockProgress = z.infer<typeof managedSaveUnlockProgressSchema>;
export type ManagedSaveUnlockTarget = z.infer<typeof managedSaveUnlockTargetSchema>;
export type PolicyPlaybackMode = z.infer<typeof policyPlaybackModeSchema>;
export type SaveAttemptStatus = z.infer<typeof saveAttemptStatusSchema>;
export type SavePolicyArtifact = z.infer<typeof savePolicyArtifactSchema>;
export type SaveUnlockTargetStatus = z.infer<typeof saveUnlockTargetStatusSchema>;

export interface CareerModeRunnerLaunchRequest {
  attemptSeed: string | null;
  device: WatchDevice;
  policyMode: PolicyPlaybackMode;
  recordingEnabled: boolean;
  recordingPath: string | null;
  renderer: WatchRenderer | null;
  saveGameId: string;
  target: ManagedSaveUnlockTarget | null;
}
