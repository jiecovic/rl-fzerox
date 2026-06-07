// src/rl_fzerox/apps/run_manager/web/src/test/features/save_games/SaveGameWorkspace.test.tsx
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SaveGameSession } from "@/app/workspace/types";
import { SaveGameWorkspace } from "@/features/save_games/SaveGameWorkspace";
import type { ManagedSaveGame } from "@/shared/api/contract";
import { configMetadataFixture, runFixture, saveGameFixture } from "@/test/fixtures";
import { cleanup, render, screen } from "@/test/render";

describe("SaveGameWorkspace", () => {
  afterEach(() => {
    cleanup();
  });

  it("creates a save game without a save identity seed", async () => {
    const user = userEvent.setup();
    const created = saveGameFixture({ name: "unlock save" });
    const onCreateSaveGame = vi.fn().mockResolvedValue(created);
    const onPatchSession = vi.fn();

    render(
      <StatefulNewSaveGameWorkspace
        onCreateSaveGame={onCreateSaveGame}
        onPatchSession={onPatchSession}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(onCreateSaveGame).toHaveBeenCalledWith("unlock save");
    expect(onPatchSession).toHaveBeenLastCalledWith("save-game:new", {
      nameText: "unlock save",
      saveGameId: "save-001",
      title: "unlock save",
    });
  });

  it("opens the save-game directory", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({ id: "save-002", name: "expert unlock" });
    const onOpenSaveGameDirectory = vi.fn().mockResolvedValue(undefined);

    renderSaveGameWorkspace({
      saveGame,
      onOpenSaveGameDirectory,
    });

    await user.click(screen.getByRole("button", { name: "Open folder" }));

    expect(onOpenSaveGameDirectory).toHaveBeenCalledWith("save-002");
  });

  it("stages broad course setup controls and saves course setups", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture();
    const run = runFixture({ id: "run-policy", name: "fast policy", status: "finished" });
    const onUpsertCourseSetup = vi.fn().mockResolvedValue(saveGame);

    renderSaveGameWorkspace({
      runs: [run],
      saveGame,
      onUpsertCourseSetup,
    });

    await user.selectOptions(screen.getByRole("combobox", { name: "Default policy" }), run.id);
    expect(screen.getByRole("button", { name: "Saved" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Apply to all courses" }));
    expect(onUpsertCourseSetup).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Save 24 changes" }));

    expect(onUpsertCourseSetup).toHaveBeenCalledTimes(24);
    expect(onUpsertCourseSetup).toHaveBeenCalledWith({
      courseId: "mute_city",
      cupId: "jack",
      difficulty: null,
      engineSettingRawValue: 50,
      policyArtifact: "best",
      policyRunId: "run-policy",
      saveGameId: "save-001",
      scope: "course",
      vehicleId: "blue_falcon",
    });
    expect(onUpsertCourseSetup).not.toHaveBeenCalledWith(
      expect.objectContaining({ scope: "global" }),
    );
    expect(onUpsertCourseSetup).not.toHaveBeenCalledWith(expect.objectContaining({ scope: "cup" }));
  });

  it("counts one dirty change for one course engine override", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: [
        {
          ...globalCourseSetupFixture(),
          id: "assignment-jack-cup",
          course_id: null,
          cup_id: "jack",
          scope: "cup",
        },
      ],
    });
    const onUpsertCourseSetup = vi.fn().mockResolvedValue(saveGame);

    renderSaveGameWorkspace({
      saveGame,
      onUpsertCourseSetup,
    });

    const muteCityEngine = screen.getByRole("spinbutton", { name: "Mute City engine" });
    await user.clear(muteCityEngine);
    await user.type(muteCityEngine, "40");

    await user.click(screen.getByRole("button", { name: "Save 1 change" }));

    expect(onUpsertCourseSetup).toHaveBeenCalledTimes(1);
    expect(onUpsertCourseSetup).toHaveBeenCalledWith({
      courseId: "mute_city",
      cupId: "jack",
      difficulty: null,
      engineSettingRawValue: 40,
      policyArtifact: "best",
      policyRunId: "run-policy",
      saveGameId: "save-001",
      scope: "course",
      vehicleId: "blue_falcon",
    });
  });

  it("starts the visible career runner for the next target", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: [globalCourseSetupFixture()],
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");
    const onRefresh = vi.fn().mockResolvedValue(undefined);

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
      onRefresh,
    });

    await user.selectOptions(screen.getByLabelText("Career Mode device"), "cpu");
    await user.selectOptions(
      screen.getByLabelText("Career Mode initial policy mode"),
      "stochastic",
    );
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStartCareerMode).toHaveBeenCalledWith(
      "save-001",
      "cpu",
      "gliden64",
      "123",
      "stochastic",
      null,
    );
    expect(onRefresh).toHaveBeenCalled();
    expect(await screen.findByText("Runner started.")).toBeInTheDocument();
  });

  it("starts the visible career runner for a clicked unlock target", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: [globalCourseSetupFixture()],
    });
    const selectedTarget = saveGame.unlock_progress?.targets[1];
    if (selectedTarget === undefined) {
      throw new Error("fixture is missing the selected target");
    }
    const onStartCareerMode = vi.fn().mockResolvedValue("started");

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
    });

    await user.click(screen.getByRole("button", { name: "Start Clear Novice Queen Cup" }));

    expect(onStartCareerMode).toHaveBeenCalledWith(
      "save-001",
      "cuda",
      "gliden64",
      "123",
      "deterministic",
      selectedTarget,
    );
  });

  it("starts a cup target from a cup setup", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: [
        {
          ...globalCourseSetupFixture(),
          id: "assignment-king-1",
          course_id: null,
          cup_id: "king",
          scope: "cup",
        },
      ],
      unlock_progress: {
        inspection_status: "inspected",
        completed_count: 2,
        total_count: 16,
        unlocked_vehicle_count: 6,
        unlocked_vehicle_ids: [
          "blue_falcon",
          "golden_fox",
          "wild_goose",
          "fire_stingray",
          "white_cat",
          "red_gazelle",
        ],
        next_target: {
          sequence_index: 2,
          kind: "clear_gp_cup",
          status: "pending",
          label: "Clear Novice King Cup",
          difficulty: "novice",
          cup_id: "king",
          course_id: null,
        },
        targets: [
          {
            sequence_index: 2,
            kind: "clear_gp_cup",
            status: "pending",
            label: "Clear Novice King Cup",
            difficulty: "novice",
            cup_id: "king",
            course_id: null,
          },
        ],
      },
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
    });

    expect(screen.queryByText("Choose a policy for the next cup.")).not.toBeInTheDocument();
    expect(screen.getByText("Clear Novice King Cup is ready.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStartCareerMode).toHaveBeenCalled();
  });

  it("shows start for a paused save without real progress or terminal attempts", () => {
    renderSaveGameWorkspace({
      saveGame: saveGameFixture({
        status: "paused",
        course_setups: [globalCourseSetupFixture()],
      }),
    });

    expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
  });

  it("shows continue after save progress exists", () => {
    renderSaveGameWorkspace({
      saveGame: saveGameFixture({
        status: "paused",
        course_setups: [globalCourseSetupFixture()],
        unlock_progress: {
          inspection_status: "inspected",
          completed_count: 1,
          total_count: 2,
          unlocked_vehicle_count: 6,
          unlocked_vehicle_ids: [
            "blue_falcon",
            "golden_fox",
            "wild_goose",
            "fire_stingray",
            "white_cat",
            "red_gazelle",
          ],
          next_target: {
            sequence_index: 1,
            kind: "clear_gp_cup",
            status: "pending",
            label: "Clear Novice Queen Cup",
            difficulty: "novice",
            cup_id: "queen",
            course_id: null,
          },
          targets: [
            {
              sequence_index: 0,
              kind: "clear_gp_cup",
              status: "succeeded",
              label: "Clear Novice Jack Cup",
              difficulty: "novice",
              cup_id: "jack",
              course_id: null,
            },
            {
              sequence_index: 1,
              kind: "clear_gp_cup",
              status: "pending",
              label: "Clear Novice Queen Cup",
              difficulty: "novice",
              cup_id: "queen",
              course_id: null,
            },
          ],
        },
      }),
    });

    expect(screen.getByRole("button", { name: "Continue" })).toBeInTheDocument();
  });

  it("does not show continue after old race-truncation artifacts", () => {
    renderSaveGameWorkspace({
      saveGame: saveGameFixture({
        status: "paused",
        course_setups: [globalCourseSetupFixture()],
        attempts: [
          {
            id: "attempt-001",
            course_id: null,
            cup_id: "jack",
            difficulty: "novice",
            failure_reason: "race truncated",
            finish_position: 1,
            finish_time_s: null,
            finished_at: "2026-06-02T10:40:00+00:00",
            policy_artifact: "best",
            policy_run_id: "run-policy",
            save_game_id: "save-001",
            started_at: "2026-06-02T10:30:00+00:00",
            status: "failed",
            target_kind: "clear_gp_cup",
          },
        ],
      }),
    });

    expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
  });

  it("renders the cup matrix without manual completion buttons", () => {
    renderSaveGameWorkspace({
      saveGame: saveGameFixture(),
    });

    expect(screen.getByText("Unlock path")).toBeInTheDocument();
    expect(screen.getAllByText("Jack Cup").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Queen Cup").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /mark succeeded/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /mark failed/i })).not.toBeInTheDocument();
  });
});

function renderSaveGameWorkspace({
  onOpenSaveGameDirectory = vi.fn(),
  onRefresh = vi.fn(),
  onUpsertCourseSetup = vi.fn(),
  onStartCareerMode = vi.fn(),
  runs = [],
  saveGame,
}: {
  onOpenSaveGameDirectory?: (saveGameId: string) => Promise<void>;
  onRefresh?: () => Promise<void>;
  onUpsertCourseSetup?: Parameters<typeof SaveGameWorkspace>[0]["onUpsertCourseSetup"];
  onStartCareerMode?: Parameters<typeof SaveGameWorkspace>[0]["onStartCareerMode"];
  runs?: Parameters<typeof SaveGameWorkspace>[0]["runs"];
  saveGame: ManagedSaveGame;
}) {
  return render(
    <StatefulExistingSaveGameWorkspace
      onOpenSaveGameDirectory={onOpenSaveGameDirectory}
      onRefresh={onRefresh}
      onStartCareerMode={onStartCareerMode}
      onUpsertCourseSetup={onUpsertCourseSetup}
      runs={runs}
      saveGame={saveGame}
    />,
  );
}

function globalCourseSetupFixture() {
  return {
    id: "assignment-001",
    course_id: null,
    cup_id: null,
    difficulty: null,
    policy_artifact: "best" as const,
    policy_run_id: "run-policy",
    save_game_id: "save-001",
    scope: "global" as const,
    vehicle_id: "blue_falcon",
    engine_setting_raw_value: 50,
    created_at: "2026-06-02T10:30:00+00:00",
    updated_at: "2026-06-02T10:30:00+00:00",
  };
}

function newSaveGameSession(): SaveGameSession {
  return {
    nameText: "unlock save",
    attemptSeedText: "123",
    policyMode: "deterministic",
    runnerDevice: "cuda",
    runnerRenderer: "gliden64",
    saveGameId: null,
    sessionId: "save-game:new",
    title: "unlock save",
  };
}

function StatefulNewSaveGameWorkspace({
  onCreateSaveGame,
  onPatchSession,
}: {
  onCreateSaveGame: (name: string) => Promise<ManagedSaveGame>;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
}) {
  const [session, setSession] = useState(newSaveGameSession());
  return (
    <SaveGameWorkspace
      metadata={configMetadataFixture}
      runs={[]}
      saveGame={null}
      session={session}
      onCreateSaveGame={onCreateSaveGame}
      onOpenSaveGameDirectory={vi.fn()}
      onPatchSession={(sessionId, patch) => {
        onPatchSession(sessionId, patch);
        setSession((current) =>
          current.sessionId === sessionId ? { ...current, ...patch } : current,
        );
      }}
      onRefresh={vi.fn()}
      onRenameSaveGame={vi.fn()}
      onUpsertCourseSetup={vi.fn()}
      onStartCareerMode={vi.fn()}
    />
  );
}

function StatefulExistingSaveGameWorkspace({
  onOpenSaveGameDirectory,
  onRefresh,
  onStartCareerMode,
  onUpsertCourseSetup,
  runs,
  saveGame,
}: {
  onOpenSaveGameDirectory: (saveGameId: string) => Promise<void>;
  onRefresh: () => Promise<void>;
  onStartCareerMode: Parameters<typeof SaveGameWorkspace>[0]["onStartCareerMode"];
  onUpsertCourseSetup: Parameters<typeof SaveGameWorkspace>[0]["onUpsertCourseSetup"];
  runs: Parameters<typeof SaveGameWorkspace>[0]["runs"];
  saveGame: ManagedSaveGame;
}) {
  const [session, setSession] = useState(existingSaveGameSession(saveGame.id));
  return (
    <SaveGameWorkspace
      metadata={configMetadataFixture}
      runs={runs}
      saveGame={saveGame}
      session={session}
      onCreateSaveGame={vi.fn()}
      onOpenSaveGameDirectory={onOpenSaveGameDirectory}
      onPatchSession={(sessionId, patch) => {
        setSession((current) =>
          current.sessionId === sessionId ? { ...current, ...patch } : current,
        );
      }}
      onRefresh={onRefresh}
      onRenameSaveGame={vi.fn()}
      onUpsertCourseSetup={onUpsertCourseSetup}
      onStartCareerMode={onStartCareerMode}
    />
  );
}

function existingSaveGameSession(saveGameId: string): SaveGameSession {
  return {
    nameText: "expert unlock",
    attemptSeedText: "123",
    policyMode: "deterministic",
    runnerDevice: "cuda",
    runnerRenderer: "gliden64",
    saveGameId,
    sessionId: `save-game:${saveGameId}`,
    title: "expert unlock",
  };
}
