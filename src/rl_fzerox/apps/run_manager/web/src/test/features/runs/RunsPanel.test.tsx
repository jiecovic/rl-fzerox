// src/rl_fzerox/apps/run_manager/web/src/test/features/runs/RunsPanel.test.tsx
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { RunsPanel } from "@/features/runs/RunsPanel";
import { draftFixture, runFixture } from "@/test/fixtures";

describe("RunsPanel", () => {
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
    const onOpenRun = vi.fn();
    const onResumeRun = vi.fn().mockResolvedValue(undefined);
    const onStopRun = vi.fn().mockResolvedValue(undefined);

    render(
      <RunsPanel
        drafts={drafts}
        runs={[run, failedRun]}
        onDeleteLineage={onDeleteLineage}
        onDeleteRun={onDeleteRun}
        onOpenRun={onOpenRun}
        onResumeRun={onResumeRun}
        onStopRun={onStopRun}
      />,
    );

    const runningRow = screen.getByRole("button", { name: `Open run ${run.name}` }).closest("div");
    const failedRow = screen
      .getByRole("button", { name: `Open run ${failedRun.name}` })
      .closest("div");
    expect(runningRow).not.toBeNull();
    expect(failedRow).not.toBeNull();

    await user.click(screen.getByRole("button", { name: `Open run ${run.name}` }));
    expect(onOpenRun).toHaveBeenCalledWith(run);

    await user.click(
      within(runningRow as HTMLElement).getByRole("button", {
        name: `Stop run ${run.name}`,
      }),
    );
    expect(onStopRun).toHaveBeenCalledWith(run);

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
        onOpenRun={vi.fn()}
        onResumeRun={vi.fn().mockResolvedValue(undefined)}
        onStopRun={vi.fn().mockResolvedValue(undefined)}
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
});
