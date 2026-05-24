// src/rl_fzerox/apps/run_manager/web/src/test/features/drafts/DraftsPanel.test.tsx
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { draftFixture } from "@/test/fixtures";

describe("DraftsPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("opens drafts from the row and deletes them through confirmation", async () => {
    const user = userEvent.setup();
    const draft = draftFixture();
    const onCreateDraft = vi.fn();
    const onOpenDraft = vi.fn();
    const onDeleteDraft = vi.fn().mockResolvedValue(undefined);

    render(
      <DraftsPanel
        drafts={[draft]}
        onCreateDraft={onCreateDraft}
        onDeleteDraft={onDeleteDraft}
        onOpenDraft={onOpenDraft}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Create draft" }));
    expect(onCreateDraft).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /50,000,000 steps/i }));
    expect(onOpenDraft).toHaveBeenCalledWith(draft);

    await user.click(screen.getByRole("button", { name: `Delete draft ${draft.name}` }));
    expect(screen.getByRole("dialog", { name: "Delete draft" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete draft" }));
    expect(onDeleteDraft).toHaveBeenCalledWith(draft);
  });

  it("selects and deletes multiple drafts through one confirmation", async () => {
    const user = userEvent.setup();
    const firstDraft = draftFixture({ id: "draft-001", name: "first draft" });
    const secondDraft = draftFixture({ id: "draft-002", name: "second draft" });
    const onCreateDraft = vi.fn();
    const onOpenDraft = vi.fn();
    const onDeleteDraft = vi.fn().mockResolvedValue(undefined);

    render(
      <DraftsPanel
        drafts={[firstDraft, secondDraft]}
        onCreateDraft={onCreateDraft}
        onDeleteDraft={onDeleteDraft}
        onOpenDraft={onOpenDraft}
      />,
    );

    expect(screen.getByRole("button", { name: "Delete selected" })).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: "Select draft first draft" }));
    await user.click(screen.getByRole("checkbox", { name: "Select draft second draft" }));
    await user.click(screen.getByRole("button", { name: "Delete selected (2)" }));

    expect(screen.getByRole("dialog", { name: "Delete drafts" })).toBeInTheDocument();
    expect(
      screen.getByText("Delete 2 selected drafts? This cannot be undone."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete drafts" }));

    expect(onDeleteDraft).toHaveBeenCalledTimes(2);
    expect(onDeleteDraft).toHaveBeenNthCalledWith(1, firstDraft);
    expect(onDeleteDraft).toHaveBeenNthCalledWith(2, secondDraft);
  });
});
