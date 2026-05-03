import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { draftFixture } from "@/test/fixtures";

describe("DraftsPanel", () => {
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
});
