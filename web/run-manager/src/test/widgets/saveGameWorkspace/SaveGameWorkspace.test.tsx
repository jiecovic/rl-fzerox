// web/run-manager/src/test/widgets/saveGameWorkspace/SaveGameWorkspace.test.tsx
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SaveGameSession } from "@/app/workspace/types";
import type { ManagedSaveGame } from "@/shared/api/contract";
import { configMetadataFixture, runFixture, saveGameFixture } from "@/test/fixtures";
import { cleanup, render, screen } from "@/test/render";
import { SaveGameWorkspace } from "@/widgets/saveGameWorkspace/SaveGameWorkspace";

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
    expect(onPatchSession).toHaveBeenLastCalledWith(
      "save-game:new",
      expect.objectContaining({
        nameText: "unlock save",
        saveGameId: "save-001",
        title: "unlock save",
      }),
    );
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

  it("saves runner launch settings", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture();
    const onUpdateRunnerSettings = vi.fn(async () =>
      saveGameFixture({
        runner_settings: {
          ...saveGame.runner_settings,
          attempt_seed: 456,
        },
      }),
    );

    renderSaveGameWorkspace({
      saveGame,
      onUpdateRunnerSettings,
    });

    const runtimeSeedInput = screen.getByRole("textbox", {
      name: "Career Mode runtime seed",
    });
    await user.clear(runtimeSeedInput);
    await user.type(runtimeSeedInput, "456");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onUpdateRunnerSettings).toHaveBeenCalledWith({
      attemptSeed: "456",
      device: "cuda",
      policyMode: "deterministic",
      recordingEnabled: false,
      recordingInputHudEnabled: false,
      recordingPath: null,
      renderer: "gliden64",
      saveGameId: "save-001",
    });
    expect(await screen.findByText("Runner settings saved.")).toBeInTheDocument();
  });

  it("stages broad course setup controls and saves course setups", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture();
    const run = runFixture({
      id: "run-policy",
      name: "fast policy",
      status: "finished",
      vehicle_setup: {
        engine_mode: "fixed",
        engine_setting_max_raw_value: 70,
        engine_setting_min_raw_value: 70,
        engine_setting_raw_value: 70,
        selected_vehicle_ids: ["blue_falcon"],
        selection_mode: "fixed",
      },
    });
    const onUpsertCourseSetup = vi.fn().mockResolvedValue(saveGame);
    const onUpsertCupSetup = vi.fn().mockResolvedValue(saveGame);

    renderSaveGameWorkspace({
      runs: [run],
      saveGame,
      onUpsertCourseSetup,
      onUpsertCupSetup,
    });

    await user.selectOptions(screen.getByRole("combobox", { name: "Policy" }), run.id);
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
    });
    expect(onUpsertCupSetup).not.toHaveBeenCalled();
  });

  it("imports learned engines from the selected bulk policy without saving", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      cup_setups: [cupSetupFixture({ cup_id: "jack", vehicle_id: "fire_stingray" })],
    });
    const run = runFixture({
      id: "run-adaptive",
      name: "adaptive policy",
      status: "finished",
      vehicle_setup: {
        engine_mode: "adaptive_tuner",
        engine_setting_max_raw_value: 100,
        engine_setting_min_raw_value: 0,
        engine_setting_raw_value: 50,
        selected_vehicle_ids: ["blue_falcon"],
        selection_mode: "fixed",
      },
    });
    const onImportEngineTuning = vi.fn().mockResolvedValue([
      {
        cup_id: "jack",
        course_id: "mute_city",
        difficulty: null,
        engine_setting_raw_value: 73,
        finish_count: 6,
        mean_score: -92_100,
        vehicle_id: "fire_stingray",
      },
      {
        cup_id: "jack",
        course_id: "silence",
        difficulty: null,
        engine_setting_raw_value: 88,
        finish_count: 4,
        mean_score: -90_400,
        vehicle_id: "fire_stingray",
      },
    ]);
    const onRefreshStatus = vi.fn();
    const onUpsertCourseSetup = vi.fn().mockResolvedValue(saveGame);

    renderSaveGameWorkspace({
      runs: [run],
      saveGame,
      onImportEngineTuning,
      onRefreshStatus,
      onUpsertCourseSetup,
    });

    await user.selectOptions(screen.getByRole("combobox", { name: "Policy" }), run.id);
    await user.click(screen.getByRole("button", { name: "Import learned engines" }));

    expect(onImportEngineTuning).toHaveBeenCalledWith({
      courseSetups: expect.arrayContaining([
        {
          courseId: "mute_city",
          cupId: "jack",
          difficulty: null,
          vehicleId: "fire_stingray",
        },
        {
          courseId: "silence",
          cupId: "jack",
          difficulty: null,
          vehicleId: "fire_stingray",
        },
      ]),
      policyArtifact: "best",
      policyRunId: "run-adaptive",
      saveGameId: "save-001",
    });
    expect(onUpsertCourseSetup).not.toHaveBeenCalled();
    expect(onRefreshStatus).not.toHaveBeenCalled();
    expect(await screen.findByRole("textbox", { name: "Mute City engine" })).toHaveValue("73");
    expect(screen.getByRole("textbox", { name: "Silence engine" })).toHaveValue("88");
    expect(screen.getByRole("button", { name: "Save 2 changes" })).toBeEnabled();
  });

  it("counts one dirty change for one course engine override", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: [courseSetupFixture({ course_id: "mute_city", cup_id: "jack" })],
    });
    const onUpsertCourseSetup = vi.fn().mockResolvedValue(saveGame);

    renderSaveGameWorkspace({
      saveGame,
      onUpsertCourseSetup,
    });

    const muteCityEngine = screen.getByRole("textbox", { name: "Mute City engine" });
    await user.clear(muteCityEngine);
    await user.type(muteCityEngine, "40");
    await user.tab();

    expect(screen.getByRole("textbox", { name: "Silence engine" })).toHaveValue("50");

    await user.click(await screen.findByRole("button", { name: "Save 1 change" }));

    expect(onUpsertCourseSetup).toHaveBeenCalledTimes(1);
    expect(onUpsertCourseSetup).toHaveBeenCalledWith({
      courseId: "mute_city",
      cupId: "jack",
      difficulty: null,
      engineSettingRawValue: 40,
      policyArtifact: "best",
      policyRunId: "run-policy",
      saveGameId: "save-001",
    });
  });

  it("starts the visible career runner for the next target", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: courseSetupsForCup("jack"),
      cup_setups: [cupSetupFixture({ cup_id: "jack" })],
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");
    const onRefreshStatus = vi.fn().mockResolvedValue(undefined);

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
      onRefreshStatus,
    });

    await user.selectOptions(screen.getByLabelText("Career Mode device"), "cpu");
    await user.selectOptions(
      screen.getByLabelText("Career Mode initial policy mode"),
      "stochastic",
    );
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStartCareerMode).toHaveBeenCalledWith({
      attemptSeed: "123",
      device: "cpu",
      policyMode: "stochastic",
      recordingEnabled: false,
      recordingInputHudEnabled: false,
      recordingPath: null,
      renderer: "gliden64",
      saveGameId: "save-001",
      singleTarget: false,
      target: null,
    });
    expect(onRefreshStatus).toHaveBeenCalledWith(saveGame.id);
    expect(await screen.findByText("Runner started.")).toBeInTheDocument();
  });

  it("starts with the default unlocked vehicle when no cup setup is saved", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: courseSetupsForCup("jack"),
      cup_setups: [],
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
    });

    expect(screen.queryByText("Choose a vehicle for the next cup.")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStartCareerMode).toHaveBeenCalled();
  });

  it("starts the career runner with MKV recording enabled", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: courseSetupsForCup("jack"),
      cup_setups: [cupSetupFixture({ cup_id: "jack" })],
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
    });

    await user.click(screen.getByLabelText("Record video"));
    await user.click(screen.getByLabelText("Input HUD"));
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStartCareerMode).toHaveBeenCalledWith({
      attemptSeed: "123",
      device: "cuda",
      policyMode: "deterministic",
      recordingEnabled: true,
      recordingInputHudEnabled: true,
      recordingPath: null,
      renderer: "gliden64",
      saveGameId: "save-001",
      singleTarget: false,
      target: null,
    });
  });

  it("starts the visible career runner for a clicked unlock target", async () => {
    const user = userEvent.setup();
    const baseSaveGame = saveGameFixture({
      course_setups: courseSetupsForCup("queen"),
      cup_setups: [cupSetupFixture({ cup_id: "queen" })],
    });
    const baseProgress = baseSaveGame.unlock_progress;
    if (baseProgress === null) {
      throw new Error("fixture is missing unlock progress");
    }
    const selectedTarget = baseProgress.targets[1];
    if (selectedTarget === undefined) {
      throw new Error("fixture is missing the selected target");
    }
    const launchableSelectedTarget = { ...selectedTarget, status: "pending" } as const;
    const saveGame = saveGameFixture({
      course_setups: courseSetupsForCup("queen"),
      cup_setups: [cupSetupFixture({ cup_id: "queen" })],
      unlock_progress: {
        ...baseProgress,
        inspection_status: "inspected",
        targets: [...baseProgress.targets.slice(0, 1), launchableSelectedTarget],
      },
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
    });

    await user.click(screen.getByRole("button", { name: "Start Clear Novice Queen Cup" }));

    expect(onStartCareerMode).toHaveBeenCalledWith({
      attemptSeed: "123",
      device: "cuda",
      policyMode: "deterministic",
      recordingEnabled: false,
      recordingInputHudEnabled: false,
      recordingPath: null,
      renderer: "gliden64",
      saveGameId: "save-001",
      singleTarget: true,
      target: launchableSelectedTarget,
    });
  });

  it("starts a clicked succeeded unlock target as a single-target replay", async () => {
    const user = userEvent.setup();
    const baseSaveGame = saveGameFixture({
      course_setups: courseSetupsForCup("jack"),
      cup_setups: [cupSetupFixture({ cup_id: "jack" })],
    });
    const baseProgress = baseSaveGame.unlock_progress;
    if (baseProgress === null) {
      throw new Error("fixture is missing unlock progress");
    }
    const firstTarget = baseProgress.targets[0];
    if (firstTarget === undefined) {
      throw new Error("fixture is missing the selected target");
    }
    const selectedTarget = {
      ...firstTarget,
      status: "succeeded",
    } as const;
    const saveGame = saveGameFixture({
      course_setups: courseSetupsForCup("jack"),
      cup_setups: [cupSetupFixture({ cup_id: "jack" })],
      unlock_progress: {
        ...baseProgress,
        completed_count: 1,
        inspection_status: "inspected",
        targets: [selectedTarget, ...baseProgress.targets.slice(1)],
      },
    });
    const onStartCareerMode = vi.fn().mockResolvedValue("started");

    renderSaveGameWorkspace({
      saveGame,
      onStartCareerMode,
    });

    await user.click(screen.getByRole("button", { name: "Start Clear Novice Jack Cup" }));

    expect(onStartCareerMode).toHaveBeenCalledWith(
      expect.objectContaining({
        saveGameId: "save-001",
        singleTarget: true,
        target: selectedTarget,
      }),
    );
  });

  it("starts a cup target from a cup setup", async () => {
    const user = userEvent.setup();
    const saveGame = saveGameFixture({
      course_setups: courseSetupsForCup("king"),
      cup_setups: [cupSetupFixture({ cup_id: "king" })],
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
        course_setups: courseSetupsForCup("jack"),
        cup_setups: [cupSetupFixture({ cup_id: "jack" })],
      }),
    });

    expect(screen.getByRole("button", { name: "Start" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
  });

  it("shows continue after save progress exists", () => {
    renderSaveGameWorkspace({
      saveGame: saveGameFixture({
        status: "paused",
        course_setups: courseSetupsForCup("queen"),
        cup_setups: [cupSetupFixture({ cup_id: "queen" })],
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
        course_setups: courseSetupsForCup("jack"),
        cup_setups: [cupSetupFixture({ cup_id: "jack" })],
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
  onImportEngineTuning = vi.fn(),
  onOpenSaveGameDirectory = vi.fn(),
  onRefreshStatus = vi.fn(),
  onUpdateRunnerSettings,
  onUpsertCourseSetup = vi.fn(),
  onUpsertCupSetup = vi.fn(),
  onStartCareerMode = vi.fn(),
  runs = [],
  saveGame,
}: {
  onImportEngineTuning?: Parameters<typeof SaveGameWorkspace>[0]["onImportEngineTuning"];
  onOpenSaveGameDirectory?: (saveGameId: string) => Promise<void>;
  onRefreshStatus?: Parameters<typeof SaveGameWorkspace>[0]["onRefreshStatus"];
  onUpdateRunnerSettings?: Parameters<typeof SaveGameWorkspace>[0]["onUpdateRunnerSettings"];
  onUpsertCourseSetup?: Parameters<typeof SaveGameWorkspace>[0]["onUpsertCourseSetup"];
  onUpsertCupSetup?: Parameters<typeof SaveGameWorkspace>[0]["onUpsertCupSetup"];
  onStartCareerMode?: Parameters<typeof SaveGameWorkspace>[0]["onStartCareerMode"];
  runs?: Parameters<typeof SaveGameWorkspace>[0]["runs"];
  saveGame: ManagedSaveGame;
}) {
  const updateRunnerSettings = onUpdateRunnerSettings ?? vi.fn(async () => saveGame);
  return render(
    <StatefulExistingSaveGameWorkspace
      onOpenSaveGameDirectory={onOpenSaveGameDirectory}
      onImportEngineTuning={onImportEngineTuning}
      onRefreshStatus={onRefreshStatus}
      onUpdateRunnerSettings={updateRunnerSettings}
      onStartCareerMode={onStartCareerMode}
      onUpsertCourseSetup={onUpsertCourseSetup}
      onUpsertCupSetup={onUpsertCupSetup}
      runs={runs}
      saveGame={saveGame}
    />,
  );
}

function courseSetupsForCup(cupId: string): ManagedSaveGame["course_setups"] {
  return configMetadataFixture.built_in_courses
    .filter((course) => course.cup === cupId)
    .map((course) =>
      courseSetupFixture({
        id: `assignment-${course.id}`,
        cup_id: cupId,
        course_id: course.id,
      }),
    );
}

function courseSetupFixture(
  overrides: Partial<ManagedSaveGame["course_setups"][number]> = {},
): ManagedSaveGame["course_setups"][number] {
  return {
    id: "assignment-001",
    course_id: "mute_city",
    cup_id: "jack",
    difficulty: null,
    policy_artifact: "best" as const,
    policy_run_id: "run-policy",
    save_game_id: "save-001",
    engine_setting_raw_value: 50,
    created_at: "2026-06-02T10:30:00+00:00",
    updated_at: "2026-06-02T10:30:00+00:00",
    ...overrides,
  };
}

function cupSetupFixture(
  overrides: Partial<ManagedSaveGame["cup_setups"][number]> = {},
): ManagedSaveGame["cup_setups"][number] {
  return {
    id: "cup-setup-001",
    cup_id: "jack",
    difficulty: null,
    save_game_id: "save-001",
    vehicle_id: "blue_falcon",
    created_at: "2026-06-02T10:30:00+00:00",
    updated_at: "2026-06-02T10:30:00+00:00",
    ...overrides,
  };
}

function newSaveGameSession(): SaveGameSession {
  return {
    nameText: "unlock save",
    attemptSeedText: "123",
    policyMode: "deterministic",
    recordingEnabled: false,
    recordingInputHudEnabled: false,
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
      onGlobalError={vi.fn()}
      onImportEngineTuning={vi.fn()}
      onOpenSaveGameDirectory={vi.fn()}
      onPatchSession={(sessionId, patch) => {
        onPatchSession(sessionId, patch);
        setSession((current) =>
          current.sessionId === sessionId ? { ...current, ...patch } : current,
        );
      }}
      onRefreshStatus={vi.fn()}
      onRenameSaveGame={vi.fn()}
      onUpdateRunnerSettings={vi.fn()}
      onUpsertCourseSetup={vi.fn()}
      onUpsertCupSetup={vi.fn()}
      onStartCareerMode={vi.fn()}
    />
  );
}

function StatefulExistingSaveGameWorkspace({
  onOpenSaveGameDirectory,
  onImportEngineTuning,
  onRefreshStatus,
  onUpdateRunnerSettings,
  onStartCareerMode,
  onUpsertCourseSetup,
  onUpsertCupSetup,
  runs,
  saveGame,
}: {
  onOpenSaveGameDirectory: (saveGameId: string) => Promise<void>;
  onImportEngineTuning: Parameters<typeof SaveGameWorkspace>[0]["onImportEngineTuning"];
  onRefreshStatus: Parameters<typeof SaveGameWorkspace>[0]["onRefreshStatus"];
  onUpdateRunnerSettings: Parameters<typeof SaveGameWorkspace>[0]["onUpdateRunnerSettings"];
  onStartCareerMode: Parameters<typeof SaveGameWorkspace>[0]["onStartCareerMode"];
  onUpsertCourseSetup: Parameters<typeof SaveGameWorkspace>[0]["onUpsertCourseSetup"];
  onUpsertCupSetup: Parameters<typeof SaveGameWorkspace>[0]["onUpsertCupSetup"];
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
      onGlobalError={vi.fn()}
      onImportEngineTuning={onImportEngineTuning}
      onOpenSaveGameDirectory={onOpenSaveGameDirectory}
      onPatchSession={(sessionId, patch) => {
        setSession((current) =>
          current.sessionId === sessionId ? { ...current, ...patch } : current,
        );
      }}
      onRefreshStatus={onRefreshStatus}
      onRenameSaveGame={vi.fn()}
      onUpdateRunnerSettings={onUpdateRunnerSettings}
      onUpsertCourseSetup={onUpsertCourseSetup}
      onUpsertCupSetup={onUpsertCupSetup}
      onStartCareerMode={onStartCareerMode}
    />
  );
}

function existingSaveGameSession(saveGameId: string): SaveGameSession {
  return {
    nameText: "expert unlock",
    attemptSeedText: "123",
    policyMode: "deterministic",
    recordingEnabled: false,
    recordingInputHudEnabled: false,
    runnerDevice: "cuda",
    runnerRenderer: "gliden64",
    saveGameId,
    sessionId: `save-game:${saveGameId}`,
    title: "expert unlock",
  };
}
