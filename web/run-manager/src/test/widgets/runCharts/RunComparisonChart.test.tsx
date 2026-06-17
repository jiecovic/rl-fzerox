// web/run-manager/src/test/widgets/runCharts/RunComparisonChart.test.tsx

import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunPlotCardProps } from "@/entities/runChart/ui/runPlotCard/types";
import { runFixture, runMetricSampleFixture } from "@/test/fixtures";
import { cleanup, render } from "@/test/render";
import {
  RunComparisonChart,
  type RunComparisonSeriesGroup,
} from "@/widgets/runCharts/chartsPanel/RunComparisonChart";

const { runPlotCardMock } = vi.hoisted(() => ({
  runPlotCardMock: vi.fn<(props: RunPlotCardProps) => null>(),
}));

vi.mock("@/entities/runChart/ui/RunPlotCard", () => ({
  RunPlotCard: (props: RunPlotCardProps) => {
    runPlotCardMock(props);
    return null;
  },
}));

describe("RunComparisonChart", () => {
  afterEach(() => {
    cleanup();
    runPlotCardMock.mockClear();
  });

  it("keeps branch-grouped runs as separate plotted segments", () => {
    const rootRun = runFixture({ id: "run-root", name: "root" });
    const forkRun = runFixture({ id: "run-fork", name: "fork" });
    const branchGroup: RunComparisonSeriesGroup = {
      color: "#86efac",
      id: "branch-root",
      label: "root · 2 runs",
      runIds: [rootRun.id, forkRun.id],
    };

    render(
      <RunComparisonChart
        buildPoints={(_run, samples) =>
          samples.map((sample) => ({
            step: sample.lineage_num_timesteps,
            value: sample.episode_reward_mean ?? 0,
          }))
        }
        emptyText="No data"
        metricsByRun={{
          [forkRun.id]: [
            runMetricSampleFixture({
              episode_reward_mean: 3,
              lineage_num_timesteps: 30,
              run_id: forkRun.id,
            }),
          ],
          [rootRun.id]: [
            runMetricSampleFixture({
              episode_reward_mean: 1,
              lineage_num_timesteps: 10,
              run_id: rootRun.id,
            }),
          ],
        }}
        runs={[rootRun, forkRun]}
        seriesGroups={[branchGroup]}
        seriesUnit="branches"
        title="Episode reward"
      />,
    );

    const props = runPlotCardMock.mock.calls[0]?.[0];
    expect(props?.series).toEqual([
      {
        color: branchGroup.color,
        groupId: branchGroup.id,
        latest: 1,
        name: branchGroup.label,
        points: [{ step: 10, value: 1 }],
        runId: rootRun.id,
      },
      {
        color: branchGroup.color,
        groupId: branchGroup.id,
        latest: 3,
        name: branchGroup.label,
        points: [{ step: 30, value: 3 }],
        runId: forkRun.id,
      },
    ]);
  });
});
