// web/run-manager/src/app/workspace/actions/saveGames.ts
import type { Dispatch, SetStateAction } from "react";

import { saveGameSessionId, upsertSaveGame } from "@/app/workspace/model";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import {
  createSaveGame,
  deleteSaveGame,
  importSaveEngineTuning,
  openSaveGameDirectory,
  renameSaveGame,
  startCareerModeRunner,
  updateSaveGameRunnerSettings,
  upsertSaveCourseSetup,
  upsertSaveCupSetup,
} from "@/shared/api/client";
import type {
  CareerModeRunnerLaunchRequest,
  ManagedSaveGame,
  SaveEngineTuningCourseSetupRecommendation,
  SaveGameRunnerSettingsUpdateRequest,
  SavePolicyArtifact,
  SavePolicySourceKind,
} from "@/shared/api/contract";

interface WorkspaceSaveGameActionsOptions {
  reloadManagerData: (options?: { showLoading?: boolean }) => Promise<void>;
  sessions: WorkspaceSessions;
  setSaveGames: Dispatch<SetStateAction<ManagedSaveGame[]>>;
}

export interface WorkspaceSaveGameActions {
  createManagedSaveGame: (name: string) => Promise<ManagedSaveGame>;
  importManagedSaveEngineTuning: (request: {
    courseSetups: readonly {
      courseId: string;
      cupId: string;
      difficulty?: string | null;
      vehicleId: string;
    }[];
    policyArtifact: SavePolicyArtifact;
    policySourceId: string;
    policySourceKind: SavePolicySourceKind;
    saveGameId: string;
  }) => Promise<readonly SaveEngineTuningCourseSetupRecommendation[]>;
  openManagedSaveGameDirectory: (saveGameId: string) => Promise<void>;
  removeSaveGame: (saveGame: ManagedSaveGame) => Promise<void>;
  renameManagedSaveGame: (saveGameId: string, name: string) => Promise<void>;
  startManagedCareerMode: (
    request: CareerModeRunnerLaunchRequest,
  ) => Promise<"started" | "already_running">;
  updateManagedSaveRunnerSettings: (
    request: SaveGameRunnerSettingsUpdateRequest,
  ) => Promise<ManagedSaveGame>;
  upsertManagedSaveCourseSetup: (request: {
    engineSettingRawValue: number;
    policyArtifact: SavePolicyArtifact;
    policySourceId: string;
    policySourceKind: SavePolicySourceKind;
    saveGameId: string;
    courseId?: string | null;
    cupId?: string | null;
    difficulty?: string | null;
  }) => Promise<ManagedSaveGame>;
  upsertManagedSaveCupSetup: (request: {
    cupId: string;
    saveGameId: string;
    vehicleId: string;
    difficulty?: string | null;
  }) => Promise<ManagedSaveGame>;
}

export function workspaceSaveGameActions({
  reloadManagerData,
  sessions,
  setSaveGames,
}: WorkspaceSaveGameActionsOptions): WorkspaceSaveGameActions {
  async function createManagedSaveGame(name: string) {
    try {
      const saveGame = await createSaveGame(name);
      setSaveGames((current) => upsertSaveGame(current, saveGame));
      return saveGame;
    } catch (caught) {
      await reloadManagerData();
      throw caught;
    }
  }

  async function upsertManagedSaveCourseSetup(request: {
    engineSettingRawValue: number;
    policyArtifact: SavePolicyArtifact;
    policySourceId: string;
    policySourceKind: SavePolicySourceKind;
    saveGameId: string;
    courseId?: string | null;
    cupId?: string | null;
    difficulty?: string | null;
  }) {
    const saveGame = await upsertSaveCourseSetup(request);
    setSaveGames((current) => upsertSaveGame(current, saveGame));
    return saveGame;
  }

  async function upsertManagedSaveCupSetup(request: {
    cupId: string;
    saveGameId: string;
    vehicleId: string;
    difficulty?: string | null;
  }) {
    const saveGame = await upsertSaveCupSetup(request);
    setSaveGames((current) => upsertSaveGame(current, saveGame));
    return saveGame;
  }

  async function importManagedSaveEngineTuning(request: {
    courseSetups: readonly {
      courseId: string;
      cupId: string;
      difficulty?: string | null;
      vehicleId: string;
    }[];
    policyArtifact: SavePolicyArtifact;
    policySourceId: string;
    policySourceKind: SavePolicySourceKind;
    saveGameId: string;
  }) {
    return await importSaveEngineTuning(request);
  }

  async function removeSaveGame(saveGame: ManagedSaveGame) {
    await deleteSaveGame(saveGame.id);
    setSaveGames((current) => current.filter((candidate) => candidate.id !== saveGame.id));
    sessions.closeWorkspaceTab(saveGameSessionId(saveGame.id));
  }

  async function renameManagedSaveGame(saveGameId: string, name: string) {
    const saveGame = await renameSaveGame(saveGameId, name);
    setSaveGames((current) => upsertSaveGame(current, saveGame));
    sessions.patchSaveGameSession(saveGameSessionId(saveGameId), {
      nameText: saveGame.name,
      title: saveGame.name,
    });
  }

  async function updateManagedSaveRunnerSettings(request: SaveGameRunnerSettingsUpdateRequest) {
    const saveGame = await updateSaveGameRunnerSettings(request);
    setSaveGames((current) => upsertSaveGame(current, saveGame));
    return saveGame;
  }

  async function openManagedSaveGameDirectory(saveGameId: string) {
    await openSaveGameDirectory(saveGameId);
  }

  async function startManagedCareerMode(
    request: CareerModeRunnerLaunchRequest,
  ): Promise<"started" | "already_running"> {
    const status = await startCareerModeRunner(request);
    await reloadManagerData();
    return status;
  }

  return {
    createManagedSaveGame,
    importManagedSaveEngineTuning,
    openManagedSaveGameDirectory,
    removeSaveGame,
    renameManagedSaveGame,
    startManagedCareerMode,
    updateManagedSaveRunnerSettings,
    upsertManagedSaveCourseSetup,
    upsertManagedSaveCupSetup,
  };
}
