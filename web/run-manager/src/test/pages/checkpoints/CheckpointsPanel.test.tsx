// web/run-manager/src/test/pages/checkpoints/CheckpointsPanel.test.tsx
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CheckpointsPanel } from "@/pages/checkpoints/CheckpointsPanel";
import { checkpointCatalogFixture } from "@/test/fixtures";
import { render, screen } from "@/test/render";

describe("CheckpointsPanel", () => {
  it("installs an available checkpoint", async () => {
    const user = userEvent.setup();
    const onInstallCheckpoint = vi.fn().mockResolvedValue(undefined);

    render(
      <CheckpointsPanel
        catalog={checkpointCatalogFixture()}
        error={null}
        onGlobalError={vi.fn()}
        onInstallCheckpoint={onInstallCheckpoint}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Install" }));

    expect(onInstallCheckpoint).toHaveBeenCalledWith("blue-falcon-fine-tuned", "v1");
  });

  it("disables install for already installed checkpoints", () => {
    const catalog = checkpointCatalogFixture();
    const installedCatalog = {
      ...catalog,
      entries: catalog.entries.map((entry) => ({
        ...entry,
        installed_checkpoint_id: "blue-falcon-fine-tuned-v1",
      })),
    };

    render(
      <CheckpointsPanel
        catalog={installedCatalog}
        error={null}
        onGlobalError={vi.fn()}
        onInstallCheckpoint={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Installed" })).toBeDisabled();
  });
});
