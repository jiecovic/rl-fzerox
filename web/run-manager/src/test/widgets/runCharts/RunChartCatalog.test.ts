// web/run-manager/src/test/widgets/runCharts/RunChartCatalog.test.ts

import { describe, expect, it } from "vitest";
import { buildChartGroups } from "@/entities/runChart/model";
import { runFixture, runMetricSampleFixture } from "@/test/fixtures";

describe("run chart catalog", () => {
  it("hides legacy global std and groups action uncertainty metrics under PPO", () => {
    const run = runFixture();
    const groups = buildChartGroups([run], {
      [run.id]: [
        runMetricSampleFixture({
          metrics: {
            "time/total_timesteps": 1_000,
            "train/std": 0.42,
            "train_entropy/air_brake": 0.31,
            "train_entropy_weight/air_brake": 3.2,
            "train_std/pitch": 0.08,
          },
        }),
      ],
    });

    const chartTitles = groups.flatMap((group) => group.charts.map((chart) => chart.title));
    expect(chartTitles).not.toContain("Policy std");

    const ppoGroup = groups.find((group) => group.id === "optimization");
    expect(ppoGroup?.charts.map((chart) => chart.title)).toEqual([
      "Entropy: Air Brake",
      "Entropy weight: Air Brake",
      "Policy std: Pitch",
    ]);
  });
});
