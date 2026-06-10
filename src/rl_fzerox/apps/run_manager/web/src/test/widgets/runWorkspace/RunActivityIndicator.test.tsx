// src/rl_fzerox/apps/run_manager/web/src/test/widgets/runWorkspace/RunActivityIndicator.test.tsx
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RunActivityIndicator } from "@/entities/run/ui/RunActivityIndicator";
import { runFixture } from "@/test/fixtures";
import { cleanup, render, screen } from "@/test/render";

describe("RunActivityIndicator", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-03T18:55:30+00:00"));
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows live status from worker heartbeat and metrics age separately", () => {
    const baseRuntime = runFixture().runtime;
    if (baseRuntime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const run = runFixture({
      worker_heartbeat_at: "2026-05-03T18:55:27+00:00",
      runtime: {
        ...baseRuntime,
        updated_at: "2026-05-03T18:55:00+00:00",
      },
    });

    render(<RunActivityIndicator run={run} />);

    expect(screen.getByText(/live/)).toBeInTheDocument();
    expect(screen.getByText(/last metrics 30s ago/)).toBeInTheDocument();
  });

  it("falls back to stale heartbeat and metrics labels when the worker is not live", () => {
    const baseRuntime = runFixture().runtime;
    if (baseRuntime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const run = runFixture({
      worker_heartbeat_at: "2026-05-03T18:55:10+00:00",
      runtime: {
        ...baseRuntime,
        updated_at: "2026-05-03T18:55:00+00:00",
      },
    });

    render(<RunActivityIndicator run={run} />);

    expect(screen.queryByText(/^live$/)).not.toBeInTheDocument();
    expect(screen.getByText(/last heartbeat 20s ago/)).toBeInTheDocument();
    expect(screen.getByText(/last metrics 30s ago/)).toBeInTheDocument();
  });
});
