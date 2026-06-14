// web/run-manager/src/test/app/workspaceModel.test.ts
import { describe, expect, it } from "vitest";

import { compareRuns, nextForkDraftName, upsertSaveGameStatus } from "@/app/workspace/model";
import type { ManagedSaveGameStatus } from "@/shared/api/contract";
import { runFixture, saveGameFixture } from "@/test/fixtures";

describe("workspace naming", () => {
  it("increments fork names from the lineage root name", () => {
    const rootRun = runFixture({
      created_at: "2026-05-03T18:52:02+00:00",
      id: "run-root",
      lineage_id: "lineage-001",
      name: "72 x 96 IMPALA",
    });
    const forkRun = runFixture({
      created_at: "2026-05-03T19:52:02+00:00",
      id: "run-fork",
      lineage_id: rootRun.lineage_id,
      lineage_step_offset: 1_000_000,
      name: "72 x 96 IMPALA 2",
      parent_run_id: rootRun.id,
      source_run_id: rootRun.id,
    });

    expect(nextForkDraftName(forkRun, [forkRun, rootRun], [rootRun.name, forkRun.name])).toBe(
      "72 x 96 IMPALA 3",
    );
  });

  it("normalizes legacy fork suffixes before incrementing", () => {
    const rootRun = runFixture({
      id: "run-root",
      lineage_id: "lineage-001",
      name: "ppo_test_1",
    });
    const forkRun = runFixture({
      id: "run-fork",
      lineage_id: rootRun.lineage_id,
      lineage_step_offset: 1_000_000,
      name: "ppo_test_1 best fork",
      parent_run_id: rootRun.id,
      source_run_id: rootRun.id,
    });

    expect(nextForkDraftName(forkRun, [forkRun, rootRun], [rootRun.name, forkRun.name])).toBe(
      "ppo_test_1 2",
    );
  });
});

describe("workspace run ordering", () => {
  it("pins running runs before newer inactive runs", () => {
    const stoppedRun = runFixture({
      created_at: "2026-05-04T08:00:00+00:00",
      id: "stopped-run",
      status: "stopped",
    });
    const failedRun = runFixture({
      created_at: "2026-05-05T08:00:00+00:00",
      id: "failed-run",
      status: "failed",
    });
    const runningRun = runFixture({
      created_at: "2026-05-03T08:00:00+00:00",
      id: "running-run",
      status: "running",
    });

    expect([stoppedRun, failedRun, runningRun].sort(compareRuns).map((run) => run.id)).toEqual([
      "running-run",
      "failed-run",
      "stopped-run",
    ]);
  });
});

describe("workspace save-game status merge", () => {
  it("preserves full editor data when merging slim status updates", () => {
    const saveGame = saveGameFixture({
      attempts: [
        {
          id: "attempt-001",
          course_id: null,
          cup_id: "jack",
          difficulty: "novice",
          failure_reason: null,
          finish_position: null,
          finish_time_s: null,
          finished_at: null,
          save_game_id: "save-001",
          started_at: "2026-06-02T10:30:00+00:00",
          status: "running",
          target_kind: "clear_gp_cup",
        },
      ],
      course_setups: [
        {
          id: "setup-001",
          course_id: "mute_city",
          cup_id: "jack",
          difficulty: "novice",
          engine_setting_raw_value: 50,
          policy_artifact: "best",
          policy_run_id: "run-policy",
          save_game_id: "save-001",
          created_at: "2026-06-02T10:30:00+00:00",
          updated_at: "2026-06-02T10:30:00+00:00",
        },
      ],
    });
    const status: ManagedSaveGameStatus = {
      id: saveGame.id,
      name: saveGame.name,
      status: "running",
      runner_active: true,
      save_path: saveGame.save_path,
      created_at: saveGame.created_at,
      updated_at: "2026-06-02T10:31:00+00:00",
      last_finished_at: saveGame.last_finished_at,
      runner_settings: saveGame.runner_settings,
      unlock_progress: saveGame.unlock_progress,
    };

    const [updated] = upsertSaveGameStatus([saveGame], status);

    expect(updated.status).toBe("running");
    expect(updated.runner_active).toBe(true);
    expect(updated.attempts).toBe(saveGame.attempts);
    expect(updated.course_setups).toBe(saveGame.course_setups);
    expect(updated.cup_setups).toBe(saveGame.cup_setups);
  });

  it("returns the same array when a slim status update has no changes", () => {
    const saveGame = saveGameFixture();
    const current = [saveGame];
    const status: ManagedSaveGameStatus = {
      id: saveGame.id,
      name: saveGame.name,
      status: saveGame.status,
      runner_active: saveGame.runner_active,
      save_path: saveGame.save_path,
      created_at: saveGame.created_at,
      updated_at: saveGame.updated_at,
      last_finished_at: saveGame.last_finished_at,
      runner_settings: saveGame.runner_settings,
      unlock_progress: saveGame.unlock_progress,
    };

    expect(upsertSaveGameStatus(current, status)).toBe(current);
  });
});
