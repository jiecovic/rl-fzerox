// web/run-manager/src/test/entities/run/runtime.test.ts
import { describe, expect, it } from "vitest";
import {
  latestActiveStartupMessage,
  lineageSimGameTimeLabel,
  progressNote,
} from "@/entities/run/model/runtime";
import { runFixture } from "@/test/fixtures";

describe("run runtime labels", () => {
  it("keeps setup activity separate from pre-runtime progress text", () => {
    const run = runFixture({
      runtime: null,
      recent_events: [
        {
          created_at: "2026-05-03T18:55:20+00:00",
          kind: "startup_materialize",
          message:
            "Resolving track sampling baselines: 13/384 complete " +
            "(existing 0, cache 0, generated 13); next Mute City",
        },
      ],
    });

    expect(progressNote(run)).toBe(
      "Target 50,000,000 steps. Runtime metrics appear after the first callback flush.",
    );
    expect(latestActiveStartupMessage(run)).toMatch(/Resolving track sampling baselines/);
  });

  it("keeps startup activity only while it is newer than runtime metrics", () => {
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const run = runFixture({
      runtime: {
        ...runtime,
        updated_at: "2026-05-03T18:55:00+00:00",
      },
      recent_events: [
        {
          created_at: "2026-05-03T18:55:20+00:00",
          kind: "startup_materialize",
          message: "Building training environments",
        },
      ],
    });

    expect(latestActiveStartupMessage(run)).toBe("Building training environments");
    expect(
      latestActiveStartupMessage(
        runFixture({
          runtime: {
            ...runtime,
            updated_at: "2026-05-03T18:55:25+00:00",
          },
          recent_events: run.recent_events,
        }),
      ),
    ).toBeNull();
  });

  it("hides startup activity after a newer failure event", () => {
    const run = runFixture({
      status: "failed",
      runtime: null,
      recent_events: [
        {
          created_at: "2026-05-03T18:56:00+00:00",
          kind: "failed",
          message: "training failed: missing checkpoint",
        },
        {
          created_at: "2026-05-03T18:55:20+00:00",
          kind: "startup_resume",
          message: "Loading latest checkpoint",
        },
      ],
    });

    expect(progressNote(run)).toBe("training failed: missing checkpoint");
    expect(latestActiveStartupMessage(run)).toBeNull();
  });

  it("derives lineage sim time from source checkpoint frames", () => {
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const parentRun = runFixture({
      id: "parent",
      action_repeat: 4,
      runtime: {
        ...runtime,
        num_timesteps: 2_000,
      },
    });
    const childRun = runFixture({
      id: "child",
      parent_run_id: "parent",
      source_num_timesteps: 1_000,
      lineage_step_offset: 1_000,
      action_repeat: 1,
      runtime: {
        ...runtime,
        num_timesteps: 500,
      },
    });

    expect(lineageSimGameTimeLabel(childRun, [parentRun, childRun])).toBe("1m 15s");
  });

  it("falls back to lineage step offset when source parents are unavailable", () => {
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const forkRun = runFixture({
      parent_run_id: "missing-source-run",
      lineage_step_offset: 30_000,
      action_repeat: 2,
      runtime: {
        ...runtime,
        num_timesteps: 10_000,
      },
    });

    expect(lineageSimGameTimeLabel(forkRun, [forkRun])).toBe("22m 13s");
  });
});
