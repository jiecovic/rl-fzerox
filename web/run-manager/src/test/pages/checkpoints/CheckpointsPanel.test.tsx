// web/run-manager/src/test/pages/checkpoints/CheckpointsPanel.test.tsx
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CheckpointsPanel } from "@/pages/checkpoints/CheckpointsPanel";
import { checkpointCatalogFixture, installedCheckpointFixture } from "@/test/fixtures";
import { cleanup, render, screen } from "@/test/render";

describe("CheckpointsPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("installs an available checkpoint", async () => {
    const user = userEvent.setup();
    const onInstallCheckpoint = vi.fn().mockResolvedValue(undefined);
    const catalog = checkpointCatalogFixture();

    render(
      <CheckpointsPanel
        catalog={catalog}
        error={null}
        onDeleteCheckpoint={vi.fn()}
        onGlobalError={vi.fn()}
        onInstallCheckpoint={onInstallCheckpoint}
        onOpenCheckpoint={vi.fn()}
      />,
    );

    await user.click(screen.getByText(catalog.entries[0].name));

    expect(screen.getByRole("dialog", { name: "Download checkpoint" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Download" }));

    expect(onInstallCheckpoint).toHaveBeenCalledWith("blue-falcon-all-cups", "v1");
  });

  it("opens the download confirmation from the row action", async () => {
    const user = userEvent.setup();

    render(
      <CheckpointsPanel
        catalog={checkpointCatalogFixture()}
        error={null}
        onDeleteCheckpoint={vi.fn()}
        onGlobalError={vi.fn()}
        onInstallCheckpoint={vi.fn()}
        onOpenCheckpoint={vi.fn()}
      />,
    );

    await user.click(
      screen.getByRole("button", {
        name: "Download checkpoint: 72 x 96 IMPALA - like: Blue Falcon All Cups V1",
      }),
    );

    expect(screen.getByRole("dialog", { name: "Download checkpoint" })).toBeInTheDocument();
  });

  it("disables install for already installed checkpoints", () => {
    const catalog = checkpointCatalogFixture();
    const installedCatalog = {
      ...catalog,
      entries: catalog.entries.map((entry) => ({
        ...entry,
        installed_checkpoint_id: "blue-falcon-all-cups-v1",
      })),
      installed_checkpoints: [installedCheckpointFixture()],
    };

    render(
      <CheckpointsPanel
        catalog={installedCatalog}
        error={null}
        onDeleteCheckpoint={vi.fn()}
        onGlobalError={vi.fn()}
        onInstallCheckpoint={vi.fn()}
        onOpenCheckpoint={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("button", {
        name: "Download checkpoint: 72 x 96 IMPALA - like: Blue Falcon All Cups V1",
      }),
    ).toBeDisabled();
    expect(screen.getByText("installed")).toBeInTheDocument();
  });
});
