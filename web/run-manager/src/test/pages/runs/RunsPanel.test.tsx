// web/run-manager/src/test/pages/runs/RunsPanel.test.tsx

import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunsPanel } from "@/pages/runs/RunsPanel";
import { draftFixture, runFixture } from "@/test/fixtures";
import { cleanup, render, screen, within } from "@/test/render";

describe("RunsPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("opens runs and exposes row actions with delete confirmation", async () => {
    const user = userEvent.setup();
    const run = runFixture();
    const failedRun = runFixture({
      id: "run-failed",
      lineage_id: run.lineage_id,
      name: "ppo_test_2",
      parent_run_id: run.id,
      pending_command: null,
      runtime: null,
      status: "failed",
    });
    const drafts = [draftFixture()];
    const onDeleteLineage = vi.fn().mockResolvedValue(undefined);
    const onDeleteRun = vi.fn().mockResolvedValue(undefined);
    const onExportRun = vi.fn().mockResolvedValue(undefined);
    const onOpenRun = vi.fn();
    const onResumeRun = vi.fn().mockResolvedValue(undefined);
    const onStopRun = vi.fn().mockResolvedValue(undefined);

    render(
      <RunsPanel
        drafts={drafts}
        runs={[run, failedRun]}
        onDeleteLineage={onDeleteLineage}
        onDeleteRun={onDeleteRun}
        onExportRun={onExportRun}
        onGlobalError={vi.fn()}
        onImportRunBundle={vi.fn().mockResolvedValue(undefined)}
        onOpenRun={onOpenRun}
        onResumeRun={onResumeRun}
        onStopRun={onStopRun}
        onUpdateLineageGroups={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    const runningRow = screen
      .getAllByRole("button", { name: `Open run ${run.name}` })[0]
      .closest("div");
    const failedRow = screen
      .getByRole("button", { name: `Open run ${failedRun.name}` })
      .closest("div");
    expect(runningRow).not.toBeNull();
    expect(failedRow).not.toBeNull();

    await user.click(screen.getAllByRole("button", { name: `Open run ${run.name}` })[0]);
    expect(onOpenRun).toHaveBeenCalledWith(run);

    await user.click(
      within(runningRow as HTMLElement).getByRole("button", {
        name: `Stop run ${run.name}`,
      }),
    );
    expect(onStopRun).toHaveBeenCalledWith(run);

    await user.click(
      within(failedRow as HTMLElement).getByRole("button", {
        name: `Export run ${failedRun.name}`,
      }),
    );
    expect(onExportRun).toHaveBeenCalledWith(failedRun);

    await user.click(
      within(failedRow as HTMLElement).getByRole("button", {
        name: `Delete run ${failedRun.name}`,
      }),
    );
    expect(screen.getByRole("dialog", { name: "Delete run" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete run" }));
    expect(onDeleteRun).toHaveBeenCalledWith(failedRun);

    expect(screen.getByRole("button", { name: `Delete lineage ${run.name}` })).toBeDisabled();
  });

  it("deletes a stopped lineage", async () => {
    const user = userEvent.setup();
    const run = runFixture({
      pending_command: null,
      status: "stopped",
    });
    const leafRun = runFixture({
      id: "run-leaf",
      lineage_id: run.lineage_id,
      name: "ppo_test_2",
      parent_run_id: run.id,
      pending_command: null,
      runtime: null,
      status: "failed",
    });
    const onDeleteLineage = vi.fn().mockResolvedValue(undefined);

    render(
      <RunsPanel
        drafts={[]}
        runs={[run, leafRun]}
        onDeleteLineage={onDeleteLineage}
        onDeleteRun={vi.fn().mockResolvedValue(undefined)}
        onExportRun={vi.fn().mockResolvedValue(undefined)}
        onGlobalError={vi.fn()}
        onImportRunBundle={vi.fn().mockResolvedValue(undefined)}
        onOpenRun={vi.fn()}
        onResumeRun={vi.fn().mockResolvedValue(undefined)}
        onStopRun={vi.fn().mockResolvedValue(undefined)}
        onUpdateLineageGroups={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    const deleteLineageButton = screen
      .getAllByRole("button", { name: `Delete lineage ${run.name}` })
      .find((button) => !button.hasAttribute("disabled"));
    if (deleteLineageButton === undefined) {
      throw new Error("expected one enabled lineage delete button");
    }
    await user.click(deleteLineageButton);
    expect(screen.getByRole("dialog", { name: "Delete lineage" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete lineage" }));
    expect(onDeleteLineage).toHaveBeenCalledWith(run.lineage_id);
  });

  it("groups lineages and updates a lineage group", async () => {
    const user = userEvent.setup();
    const oldRun = runFixture({
      id: "old-run",
      lineage_groups: ["Old test runs"],
      lineage_id: "old-lineage",
      name: "old experiment",
      status: "stopped",
    });
    const newRun = runFixture({
      id: "new-run",
      lineage_groups: ["CNN sweep", "Current ablations"],
      lineage_id: "new-lineage",
      name: "medium cnn",
      status: "stopped",
    });
    const onUpdateLineageGroups = vi.fn().mockResolvedValue(undefined);

    const { rerender } = render(
      <RunsPanel
        drafts={[]}
        runs={[oldRun, newRun]}
        onDeleteLineage={vi.fn().mockResolvedValue(undefined)}
        onDeleteRun={vi.fn().mockResolvedValue(undefined)}
        onExportRun={vi.fn().mockResolvedValue(undefined)}
        onGlobalError={vi.fn()}
        onImportRunBundle={vi.fn().mockResolvedValue(undefined)}
        onOpenRun={vi.fn()}
        onResumeRun={vi.fn().mockResolvedValue(undefined)}
        onStopRun={vi.fn().mockResolvedValue(undefined)}
        onUpdateLineageGroups={onUpdateLineageGroups}
      />,
    );

    expect(screen.getByText("Old test runs")).toBeInTheDocument();
    expect(screen.getByText("CNN sweep")).toBeInTheDocument();
    expect(screen.getByText("Current ablations")).toBeInTheDocument();
    expect(screen.getByText(/local\/tensorboard_views\/old-test-runs/)).toBeInTheDocument();

    const oldGroupSection = screen.getByLabelText("Old test runs lineage group");
    expect(within(oldGroupSection).getAllByText("old experiment")).not.toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Collapse group Old test runs" }));
    expect(within(oldGroupSection).queryByText("old experiment")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Expand group Old test runs" }));
    expect(within(oldGroupSection).getAllByText("old experiment")).not.toHaveLength(0);

    const groupInput = screen.getAllByRole("textbox", {
      name: "Groups for lineage medium cnn",
    })[0];
    await user.clear(groupInput);
    await user.type(groupInput, "Recurrent sweep, Current ablations");
    rerender(
      <RunsPanel
        drafts={[]}
        runs={[{ ...oldRun }, { ...newRun, lineage_groups: [...newRun.lineage_groups] }]}
        onDeleteLineage={vi.fn().mockResolvedValue(undefined)}
        onDeleteRun={vi.fn().mockResolvedValue(undefined)}
        onExportRun={vi.fn().mockResolvedValue(undefined)}
        onGlobalError={vi.fn()}
        onImportRunBundle={vi.fn().mockResolvedValue(undefined)}
        onOpenRun={vi.fn()}
        onResumeRun={vi.fn().mockResolvedValue(undefined)}
        onStopRun={vi.fn().mockResolvedValue(undefined)}
        onUpdateLineageGroups={onUpdateLineageGroups}
      />,
    );
    const savedGroupInput = screen.getAllByRole("textbox", {
      name: "Groups for lineage medium cnn",
    })[0];
    expect(savedGroupInput).toHaveValue("Recurrent sweep, Current ablations");
    const groupForm = savedGroupInput.closest("form");
    if (!(groupForm instanceof HTMLElement)) {
      throw new Error("lineage group form not found");
    }
    await user.click(within(groupForm).getByRole("button", { name: "Save" }));

    expect(onUpdateLineageGroups).toHaveBeenCalledWith("new-lineage", [
      "Current ablations",
      "Recurrent sweep",
    ]);
  });

  it("pins running runs in a visible active section", () => {
    const runningRun = runFixture({
      created_at: "2026-05-01T08:00:00+00:00",
      id: "running-run",
      lineage_id: "running-lineage",
      name: "active older run",
      status: "running",
    });
    const stoppedRun = runFixture({
      created_at: "2026-05-06T08:00:00+00:00",
      id: "stopped-run",
      lineage_id: "stopped-lineage",
      name: "newer stopped run",
      runtime: null,
      status: "stopped",
    });

    render(
      <RunsPanel
        drafts={[]}
        runs={[stoppedRun, runningRun]}
        onDeleteLineage={vi.fn().mockResolvedValue(undefined)}
        onDeleteRun={vi.fn().mockResolvedValue(undefined)}
        onExportRun={vi.fn().mockResolvedValue(undefined)}
        onGlobalError={vi.fn()}
        onImportRunBundle={vi.fn().mockResolvedValue(undefined)}
        onOpenRun={vi.fn()}
        onResumeRun={vi.fn().mockResolvedValue(undefined)}
        onStopRun={vi.fn().mockResolvedValue(undefined)}
        onUpdateLineageGroups={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    const activeRuns = screen.getAllByRole("region", { name: "Active runs" })[0];

    expect(within(activeRuns).getByText("1 running")).toBeInTheDocument();
    expect(
      within(activeRuns).getByRole("button", { name: "Open run active older run" }),
    ).toBeInTheDocument();
    expect(
      within(activeRuns).queryByRole("button", { name: "Open run newer stopped run" }),
    ).not.toBeInTheDocument();
  });

  it("renders long lineages as root plus recent runs until expanded", async () => {
    const user = userEvent.setup();
    const lineageId = "large-lineage";
    const runs = Array.from({ length: 20 }, (_, index) =>
      runFixture({
        id: `large-run-${index}`,
        lineage_id: lineageId,
        name: index === 0 ? "lineage root" : `fork ${index}`,
        parent_run_id: index === 0 ? null : `large-run-${index - 1}`,
        status: "stopped",
        created_at: `2026-05-${String(index + 1).padStart(2, "0")}T00:00:00+00:00`,
      }),
    );

    render(
      <RunsPanel
        drafts={[]}
        runs={runs}
        onDeleteLineage={vi.fn().mockResolvedValue(undefined)}
        onDeleteRun={vi.fn().mockResolvedValue(undefined)}
        onExportRun={vi.fn().mockResolvedValue(undefined)}
        onGlobalError={vi.fn()}
        onImportRunBundle={vi.fn().mockResolvedValue(undefined)}
        onOpenRun={vi.fn()}
        onResumeRun={vi.fn().mockResolvedValue(undefined)}
        onStopRun={vi.fn().mockResolvedValue(undefined)}
        onUpdateLineageGroups={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    expect(screen.getByRole("button", { name: "Open run lineage root" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Open run fork 1" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open run fork 19" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show all" }));

    expect(screen.getByRole("button", { name: "Open run fork 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show recent" })).toBeInTheDocument();
  });
});
