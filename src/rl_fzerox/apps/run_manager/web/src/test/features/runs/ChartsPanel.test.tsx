import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ChartsPanel } from "@/features/runs/ChartsPanel";
import { runFixture, runMetricSampleFixture } from "@/test/fixtures";

const fetchFreshRunMetricsMock = vi.fn();
const getCachedRunMetricsMock = vi.fn();

vi.mock("@/shared/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/client")>("@/shared/api/client");
  return {
    ...actual,
    fetchFreshRunMetrics: (runId: string, rangeMode: "recent" | "full") =>
      fetchFreshRunMetricsMock(runId, rangeMode),
    getCachedRunMetrics: (runId: string, rangeMode: "recent" | "full") =>
      getCachedRunMetricsMock(runId, rangeMode),
  };
});

describe("ChartsPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("does not re-inject an old focused run when run polling refreshes", async () => {
    window.localStorage.clear();
    const user = userEvent.setup();
    fetchFreshRunMetricsMock.mockResolvedValue([]);
    getCachedRunMetricsMock.mockReturnValue(null);

    const focusedRun = runFixture({
      id: "run-focused",
      name: "ppo_test_1 fork fork",
      created_at: "2026-05-04T08:58:00+00:00",
      status: "stopped",
    });
    const selectedRun = runFixture({
      id: "run-selected",
      name: "ppo_test_1 fork",
      created_at: "2026-05-04T08:39:23+00:00",
      status: "stopped",
    });

    const { rerender } = render(
      <ChartsPanel focusedRunId={focusedRun.id} runs={[focusedRun, selectedRun]} />,
    );

    const focusedRow = rowForRun(focusedRun.name);
    const selectedRow = rowForRun(selectedRun.name);

    await user.click(within(selectedRow).getByRole("checkbox"));
    await user.click(within(focusedRow).getByRole("checkbox"));

    expect(selectedLegendNames()).toEqual([selectedRun.name]);

    rerender(
      <ChartsPanel focusedRunId={focusedRun.id} runs={[{ ...focusedRun }, { ...selectedRun }]} />,
    );

    expect(selectedLegendNames()).toEqual([selectedRun.name]);
    expect(selectionSwatchColor(selectedRun.name)).toBe(legendSwatchColor(selectedRun.name));
  });

  it("keeps selected runs across tab remounts", async () => {
    window.localStorage.clear();
    fetchFreshRunMetricsMock.mockResolvedValue([]);
    getCachedRunMetricsMock.mockReturnValue(null);

    const rootRun = runFixture({
      id: "run-root",
      name: "ppo_test_1",
      created_at: "2026-05-03T18:52:02+00:00",
      status: "stopped",
    });
    const forkRun = runFixture({
      id: "run-fork",
      name: "ppo_test_1 fork",
      created_at: "2026-05-04T08:39:23+00:00",
      status: "stopped",
    });
    const secondForkRun = runFixture({
      id: "run-fork-2",
      name: "ppo_test_1 fork fork",
      created_at: "2026-05-04T08:58:00+00:00",
      status: "stopped",
    });
    window.localStorage.setItem("run-chart-selected-runs", JSON.stringify([forkRun.id]));

    const firstRender = render(
      <ChartsPanel focusedRunId={null} runs={[secondForkRun, forkRun, rootRun]} />,
    );

    firstRender.unmount();
    cleanup();

    render(<ChartsPanel focusedRunId={null} runs={[secondForkRun, forkRun, rootRun]} />);

    await waitFor(() => {
      expect(selectedLegendNames()).toEqual([forkRun.name]);
    });
  });

  it("groups selected runs by lineage in the selection panel and legend", async () => {
    window.localStorage.clear();
    fetchFreshRunMetricsMock.mockResolvedValue([]);
    getCachedRunMetricsMock.mockReturnValue(null);

    const rootRun = runFixture({
      id: "run-root",
      name: "ppo_test_1",
      lineage_id: "lineage-a",
      created_at: "2026-05-03T18:52:02+00:00",
      status: "stopped",
    });
    const forkRun = runFixture({
      id: "run-fork",
      name: "ppo_test_1 fork",
      lineage_id: "lineage-a",
      parent_run_id: "run-root",
      source_run_id: "run-root",
      created_at: "2026-05-04T08:39:23+00:00",
      status: "stopped",
    });
    const secondRootRun = runFixture({
      id: "run-other-root",
      name: "ppo_masked_lidar",
      lineage_id: "lineage-b",
      created_at: "2026-05-04T09:30:00+00:00",
      status: "stopped",
    });

    render(<ChartsPanel focusedRunId={null} runs={[forkRun, secondRootRun, rootRun]} />);

    expect(screen.getByLabelText("ppo_test_1 lineage runs")).toBeInTheDocument();
    expect(screen.getByLabelText("ppo_masked_lidar lineage runs")).toBeInTheDocument();
    expect(screen.getByText("2 runs")).toBeInTheDocument();
    expect(screen.getByText("1 runs")).toBeInTheDocument();
  });

  it("selects and clears a whole lineage from its header checkbox", async () => {
    window.localStorage.clear();
    const user = userEvent.setup();
    fetchFreshRunMetricsMock.mockResolvedValue([]);
    getCachedRunMetricsMock.mockReturnValue(null);

    const rootRun = runFixture({
      id: "run-root",
      name: "ppo_test_1",
      lineage_id: "lineage-a",
      created_at: "2026-05-03T18:52:02+00:00",
      status: "stopped",
    });
    const forkRun = runFixture({
      id: "run-fork",
      name: "ppo_test_1 fork",
      lineage_id: "lineage-a",
      parent_run_id: "run-root",
      source_run_id: "run-root",
      created_at: "2026-05-04T08:39:23+00:00",
      status: "stopped",
    });
    const secondRootRun = runFixture({
      id: "run-other-root",
      name: "ppo_masked_lidar",
      lineage_id: "lineage-b",
      created_at: "2026-05-04T09:30:00+00:00",
      status: "stopped",
    });

    render(<ChartsPanel focusedRunId={null} runs={[forkRun, secondRootRun, rootRun]} />);

    const lineageSection = screen.getByLabelText("ppo_test_1 lineage runs");
    if (!(lineageSection instanceof HTMLElement)) {
      throw new Error("lineage section not found");
    }
    const lineageToggle = lineageSection.querySelector(
      'input[aria-label="Select lineage ppo_test_1"]',
    );
    if (!(lineageToggle instanceof HTMLInputElement)) {
      throw new Error("lineage toggle not found");
    }

    await user.click(screen.getByRole("button", { name: "Clear" }));
    await user.click(lineageToggle);
    expect(selectedLegendNames()).toEqual(["ppo_test_1 fork", "ppo_test_1"]);

    await user.click(lineageToggle);
    expect(
      screen.getByText("Select at least one run to render comparison plots."),
    ).toBeInTheDocument();
  });

  it("collapses and expands a lineage group in the selection panel", async () => {
    window.localStorage.clear();
    const user = userEvent.setup();
    fetchFreshRunMetricsMock.mockResolvedValue([]);
    getCachedRunMetricsMock.mockReturnValue(null);

    const rootRun = runFixture({
      id: "run-root",
      name: "ppo_test_1",
      lineage_id: "lineage-a",
      created_at: "2026-05-03T18:52:02+00:00",
      status: "stopped",
    });
    const forkRun = runFixture({
      id: "run-fork",
      name: "ppo_test_1 fork",
      lineage_id: "lineage-a",
      parent_run_id: "run-root",
      source_run_id: "run-root",
      created_at: "2026-05-04T08:39:23+00:00",
      status: "stopped",
    });

    render(<ChartsPanel focusedRunId={null} runs={[forkRun, rootRun]} />);

    const lineageSection = screen.getByLabelText("ppo_test_1 lineage runs");
    expect(within(lineageSection).getByText("ppo_test_1 fork")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Collapse lineage ppo_test_1" }));
    expect(within(lineageSection).queryByText("ppo_test_1 fork")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Expand lineage ppo_test_1" }));
    expect(within(lineageSection).getByText("ppo_test_1 fork")).toBeInTheDocument();
  });

  it("unmounts chart cards for collapsed groups", async () => {
    window.localStorage.clear();
    const user = userEvent.setup();
    fetchFreshRunMetricsMock.mockResolvedValue([
      runMetricSampleFixture(),
      runMetricSampleFixture({
        created_at: "2026-05-03T18:56:00+00:00",
        lineage_num_timesteps: 1_300_000,
        num_timesteps: 1_300_000,
        metrics: {
          ...runMetricSampleFixture().metrics,
          "rollout/ep_rew_mean": 4.8,
        },
      }),
    ]);
    getCachedRunMetricsMock.mockReturnValue(null);

    render(<ChartsPanel focusedRunId={null} runs={[runFixture({ status: "stopped" })]} />);

    await waitFor(() => {
      expect(screen.getByText("Episode reward")).toBeInTheDocument();
    });

    const rolloutSummary = screen.getByText("Rollout").closest("summary");
    if (!(rolloutSummary instanceof HTMLElement)) {
      throw new Error("rollout disclosure summary not found");
    }
    await user.click(rolloutSummary);

    expect(screen.queryByText("Episode reward")).not.toBeInTheDocument();
  });
});

function rowForRun(runName: string) {
  const panelLabel = screen
    .getAllByText("Selected runs")
    .find((element) => element.closest(".run-chart-selection-panel") !== null);
  const panel = panelLabel?.closest(".run-chart-selection-panel");
  if (!(panel instanceof HTMLElement)) {
    throw new Error("selection panel not found");
  }
  const row = [...panel.querySelectorAll(".run-chart-selection-row")].find((element) => {
    const label = element.querySelector(".run-chart-selection-copy strong");
    return label?.textContent?.trim() === runName;
  });
  if (!(row instanceof HTMLElement)) {
    throw new Error(`selection row not found for ${runName}`);
  }
  return row;
}

function selectedLegendNames() {
  return within(latestLegend())
    .getAllByRole("button")
    .map((button) => button.textContent?.trim() ?? "");
}

function selectionSwatchColor(runName: string) {
  const row = rowForRun(runName);
  const swatch = row.querySelector(".run-chart-selection-swatch");
  if (!(swatch instanceof HTMLElement)) {
    throw new Error(`selection swatch not found for ${runName}`);
  }
  return swatch.style.background;
}

function legendSwatchColor(runName: string) {
  const legend = latestLegend();
  const button = within(legend)
    .getAllByRole("button")
    .find((item) => item.textContent?.trim() === runName);
  if (!(button instanceof HTMLElement)) {
    throw new Error(`legend button not found for ${runName}`);
  }
  const swatch = button.querySelector(".run-chart-legend-swatch");
  if (!(swatch instanceof HTMLElement)) {
    throw new Error(`legend swatch not found for ${runName}`);
  }
  return swatch.style.background;
}

function latestLegend() {
  const legend = screen.getAllByLabelText("Selected run colors").at(-1);
  if (!(legend instanceof HTMLElement)) {
    throw new Error("selected run legend not found");
  }
  return legend;
}
