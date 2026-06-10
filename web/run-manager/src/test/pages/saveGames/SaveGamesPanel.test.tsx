// web/run-manager/src/test/pages/saveGames/SaveGamesPanel.test.tsx
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SaveGamesPanel } from "@/pages/saveGames/SaveGamesPanel";
import { saveGameFixture } from "@/test/fixtures";
import { cleanup, fireEvent, render, screen } from "@/test/render";

describe("SaveGamesPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("opens a new save-game workspace instead of rendering an inline form", async () => {
    const user = userEvent.setup();
    const onCreateSaveGame = vi.fn();

    render(
      <SaveGamesPanel
        saveGames={[]}
        onCreateSaveGame={onCreateSaveGame}
        onOpenSaveGame={vi.fn()}
      />,
    );

    expect(screen.queryByRole("textbox", { name: "Save game name" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Create career save" }));

    expect(onCreateSaveGame).toHaveBeenCalledOnce();
  });

  it("lists existing save games and opens them as workspaces", async () => {
    const saveGame = saveGameFixture({ name: "master unlock" });
    const onOpenSaveGame = vi.fn();

    render(
      <SaveGamesPanel
        saveGames={[saveGame]}
        onCreateSaveGame={vi.fn()}
        onOpenSaveGame={onOpenSaveGame}
      />,
    );

    expect(screen.getByText("master unlock")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "master unlock progress" })).toHaveAttribute(
      "aria-valuenow",
      "0",
    );

    const row = screen.getByRole("row", { name: /master unlock/i });
    fireEvent.click(row);

    expect(onOpenSaveGame).toHaveBeenCalledWith(saveGame);
  });
});
