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
        onDeleteSaveGame={vi.fn().mockResolvedValue(undefined)}
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
        onDeleteSaveGame={vi.fn().mockResolvedValue(undefined)}
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

  it("deletes a single save game through row action confirmation", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({ name: "master unlock" });
    const onDeleteSaveGame = vi.fn().mockResolvedValue(undefined);

    render(
      <SaveGamesPanel
        saveGames={[saveGame]}
        onCreateSaveGame={vi.fn()}
        onDeleteSaveGame={onDeleteSaveGame}
        onOpenSaveGame={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Delete career save master unlock" }));
    expect(screen.getByRole("dialog", { name: "Delete career save" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete career save" }));

    expect(onDeleteSaveGame).toHaveBeenCalledWith(saveGame);
  });

  it("selects and deletes multiple save games through one confirmation", async () => {
    const user = userEvent.setup();
    const firstSave = saveGameFixture({ id: "save-001", name: "first save" });
    const secondSave = saveGameFixture({ id: "save-002", name: "second save" });
    const onDeleteSaveGame = vi.fn().mockResolvedValue(undefined);

    render(
      <SaveGamesPanel
        saveGames={[firstSave, secondSave]}
        onCreateSaveGame={vi.fn()}
        onDeleteSaveGame={onDeleteSaveGame}
        onOpenSaveGame={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Delete selected" })).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: "Select career save first save" }));
    await user.click(screen.getByRole("checkbox", { name: "Select career save second save" }));
    await user.click(screen.getByRole("button", { name: "Delete selected (2)" }));

    expect(screen.getByRole("dialog", { name: "Delete career saves" })).toBeInTheDocument();
    expect(
      screen.getByText("Delete 2 selected career saves? This cannot be undone."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete career saves" }));

    expect(onDeleteSaveGame).toHaveBeenCalledTimes(2);
    expect(onDeleteSaveGame).toHaveBeenNthCalledWith(1, firstSave);
    expect(onDeleteSaveGame).toHaveBeenNthCalledWith(2, secondSave);
  });

  it("keeps the confirmation open and shows delete failures", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({ name: "running save" });
    const onDeleteSaveGame = vi.fn().mockRejectedValue(new Error("stop the runner first"));

    render(
      <SaveGamesPanel
        saveGames={[saveGame]}
        onCreateSaveGame={vi.fn()}
        onDeleteSaveGame={onDeleteSaveGame}
        onOpenSaveGame={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Delete career save running save" }));
    await user.click(screen.getByRole("button", { name: "Delete career save" }));

    expect(screen.getByRole("dialog", { name: "Delete career save" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("stop the runner first");
  });
});
