// src/rl_fzerox/apps/run_manager/web/src/test/features/runs/RunChartPoints.test.ts
import { describe, expect, it } from "vitest";

import { plotSeriesKey } from "@/features/runs/charts/run_plot_card/model";
import { buildEnvStepRatePoints, metricPoints } from "@/features/runs/chartsPanel/model/points";
import { runFixture, runMetricSampleFixture } from "@/test/fixtures";

describe("run chart points", () => {
  it("bridges a forked run back to its lineage offset", () => {
    const samples = [
      runMetricSampleFixture({
        lineage_num_timesteps: 1_010_240,
        num_timesteps: 10_240,
        metrics: {
          ...runMetricSampleFixture().metrics,
          "rollout/ep_rew_mean": 4.2,
        },
      }),
    ];

    expect(metricPoints(samples, "rollout/ep_rew_mean")).toEqual([
      { step: 1_000_000, value: 4.2 },
      { step: 1_010_240, value: 4.2 },
    ]);
  });

  it("uses the same fork bridge for derived env step rate points", () => {
    const run = runFixture({
      id: "run-fork",
      lineage_step_offset: 1_000_000,
      source_num_timesteps: 1_000_000,
      started_at: "2026-05-03T18:52:10+00:00",
      status: "stopped",
    });
    const samples = [
      runMetricSampleFixture({
        created_at: "2026-05-03T18:55:00+00:00",
        lineage_num_timesteps: 1_010_240,
        num_timesteps: 10_240,
        metrics: {
          ...runMetricSampleFixture().metrics,
          "time/fps": 912,
        },
      }),
    ];

    expect(buildEnvStepRatePoints(run, samples)).toEqual([
      { step: 1_000_000, value: 912 },
      { step: 1_010_240, value: 912 },
    ]);
  });

  it("changes plot series keys when display styling changes", () => {
    const baseSeries = [
      {
        color: "var(--accent)",
        latest: 1,
        name: "run a",
        points: [{ step: 1, value: 1 }],
        runId: "run-a",
      },
    ];

    expect(plotSeriesKey(baseSeries)).not.toBe(
      plotSeriesKey([{ ...baseSeries[0], color: "var(--run-accent)" }]),
    );
  });
});
