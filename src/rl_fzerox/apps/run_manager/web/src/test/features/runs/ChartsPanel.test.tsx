import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChartsPanel } from "@/features/runs/ChartsPanel";
import { runFixture } from "@/test/fixtures";

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
});

function rowForRun(runName: string) {
  const panelLabel = screen
    .getAllByText("Selected runs")
    .find((element) => element.closest(".run-chart-selection-panel") !== null);
  const panel = panelLabel?.closest(".run-chart-selection-panel");
  if (!(panel instanceof HTMLElement)) {
    throw new Error("selection panel not found");
  }
  const label = within(panel).getByText(runName);
  const row = label.closest("label");
  if (row === null) {
    throw new Error(`selection row not found for ${runName}`);
  }
  return row;
}

function selectedLegendNames() {
  return within(latestLegend())
    .getAllByRole("listitem")
    .map((item) => item.textContent?.trim() ?? "");
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
  const row = within(legend)
    .getAllByRole("listitem")
    .find((item) => item.textContent?.trim() === runName);
  if (!(row instanceof HTMLElement)) {
    throw new Error(`legend row not found for ${runName}`);
  }
  const swatch = row.querySelector(".run-chart-legend-swatch");
  if (!(swatch instanceof HTMLElement)) {
    throw new Error(`legend swatch not found for ${runName}`);
  }
  return swatch.style.background;
}

function latestLegend() {
  const legend = screen.getAllByRole("list", { name: "Selected run colors" }).at(-1);
  if (!(legend instanceof HTMLElement)) {
    throw new Error("selected run legend not found");
  }
  return legend;
}
