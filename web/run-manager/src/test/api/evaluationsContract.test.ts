// web/run-manager/src/test/api/evaluationsContract.test.ts

import { describe, expect, it } from "vitest";
import { evaluationsResponseSchema } from "@/shared/api/contract";
import { managedRunConfigFixture } from "@/test/fixtures";

describe("evaluations API contract", () => {
  it("accepts checkpoint mtimes larger than JavaScript safe integers as strings", () => {
    const baselineSuite = {
      id: "time_attack_all_courses-v1",
      preset_id: "time_attack_all_courses",
      preset_version: 1,
      status: "not_created",
      suite_dir: "local/evaluations/_baseline_suites/time_attack_all_courses-v1",
      manifest_path: null,
      error_message: null,
      created_at: "2026-06-22T10:00:00+00:00",
      updated_at: "2026-06-22T10:00:00+00:00",
      materialized_at: null,
    };
    const target = {
      mode: "time_attack_course",
      course_ids: [],
      cup_ids: [],
      difficulties: [],
      vehicle_ids: [],
      repeats_per_target: 1,
    };
    const config = managedRunConfigFixture;
    const payload = evaluationsResponseSchema.parse({
      evaluations: [
        {
          id: "eval-001",
          name: "Eval 1",
          status: "created",
          evaluation_dir: "local/evaluations/eval-001",
          source_run_id: "run-001",
          source_artifact: "latest",
          preset_id: "time_attack_all_courses",
          preset_version: 1,
          policy_mode: "deterministic",
          seed: 123,
          target,
          config,
          checkpoint: {
            source_run_id: "run-001",
            source_run_name: "Run 1",
            artifact: "latest",
            source_policy_path: "local/runs/run-001/checkpoints/latest/policy.zip",
            copied_policy_path: "local/evaluations/eval-001/checkpoint_snapshot/policy.zip",
            source_model_path: null,
            copied_model_path: null,
            local_num_timesteps: 123,
            lineage_num_timesteps: 456,
            source_mtime_ns: "1765000000000000123",
          },
          created_at: "2026-06-22T10:00:00+00:00",
          updated_at: "2026-06-22T10:00:00+00:00",
          started_at: null,
          finished_at: null,
          result_json_path: null,
          error_message: null,
          progress: {
            completed_attempts: 0,
            total_attempts: null,
            result_status: null,
          },
          result_summary: null,
          baseline_suite: baselineSuite,
        },
      ],
      presets: [
        {
          id: "time_attack_all_courses",
          name: "Time Attack course · all courses",
          version: 1,
          seed: 123,
          renderer: "gliden64",
          target,
          builtin: true,
          created_at: "2026-06-22T10:00:00+00:00",
          updated_at: "2026-06-22T10:00:00+00:00",
        },
      ],
      baseline_suites: [baselineSuite],
    });

    expect(payload.evaluations[0]?.checkpoint.source_mtime_ns).toBe("1765000000000000123");
  });
});
