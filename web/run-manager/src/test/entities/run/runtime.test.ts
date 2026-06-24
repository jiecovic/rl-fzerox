// web/run-manager/src/test/entities/run/runtime.test.ts
import { describe, expect, it } from "vitest";
import { latestActiveStartupMessage } from "@/entities/run/model/runtime";
import { runFixture } from "@/test/fixtures";

describe("run runtime labels", () => {
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
});
