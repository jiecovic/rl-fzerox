// web/run-manager/src/test/entities/runLineage/lineages.test.ts
import { describe, expect, it } from "vitest";
import {
  buildLineageBuckets,
  buildLineageGroups,
  runtimeFpsLabel,
} from "@/entities/runLineage/model/lineages";
import { runFixture } from "@/test/fixtures";

describe("run lineage ordering", () => {
  it("pins lineages and buckets with running runs before newer inactive work", () => {
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("expected run fixture to include runtime");
    }
    const runningRun = runFixture({
      created_at: "2026-05-01T08:00:00+00:00",
      id: "running-run",
      lineage_groups: ["Long jobs"],
      lineage_id: "running-lineage",
      name: "active older run",
      runtime: {
        ...runtime,
        updated_at: "2026-05-01T08:30:00+00:00",
      },
      status: "running",
    });
    const stoppedRun = runFixture({
      created_at: "2026-05-06T08:00:00+00:00",
      id: "stopped-run",
      lineage_groups: ["Recent jobs"],
      lineage_id: "stopped-lineage",
      name: "newer stopped run",
      runtime: null,
      status: "stopped",
    });

    const lineages = buildLineageGroups([stoppedRun, runningRun], []);
    const buckets = buildLineageBuckets(lineages);

    expect(lineages.map((lineage) => lineage.id)).toEqual(["running-lineage", "stopped-lineage"]);
    expect(buckets.map((bucket) => bucket.id)).toEqual(["long-jobs", "recent-jobs"]);
  });

  it("omits missing runtime fps instead of showing placeholders", () => {
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("expected run fixture to include runtime");
    }
    const fpsRun = runFixture({
      runtime: {
        ...runtime,
        episode_reward_mean: null,
        fps: 875,
      },
    });
    const emptyRuntimeRun = runFixture({
      runtime: {
        ...runtime,
        episode_reward_mean: null,
        fps: null,
      },
    });

    expect(runtimeFpsLabel(fpsRun)).toBe("875 fps");
    expect(runtimeFpsLabel(emptyRuntimeRun)).toBeNull();
  });
});
