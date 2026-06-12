// web/run-manager/src/test/app/workspaceModel.test.ts
import { describe, expect, it } from "vitest";

import { compareRuns, nextForkDraftName } from "@/app/workspace/model";
import { runFixture } from "@/test/fixtures";

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
