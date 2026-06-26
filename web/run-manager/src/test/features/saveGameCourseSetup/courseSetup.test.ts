// web/run-manager/src/test/features/saveGameCourseSetup/courseSetup.test.ts
import { describe, expect, it } from "vitest";

import {
  type CourseSetupDraftMap,
  courseSetupKey,
  resetCourseEngineDrafts,
  sharedPolicySelectionDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";

describe("resetCourseEngineDrafts", () => {
  it("resets selected course engines while preserving policy setup", () => {
    const muteCity = { cupId: "jack", courseId: "mute_city" };
    const silence = { cupId: "jack", courseId: "silence" };
    const current: CourseSetupDraftMap = {
      [courseSetupKey(muteCity)]: {
        ...muteCity,
        engineSettingRawValue: 83,
        policyArtifact: "latest",
        policySourceId: "run-a",
        policySourceKind: "run",
        vehicleId: "blue_falcon",
      },
      [courseSetupKey(silence)]: {
        ...silence,
        engineSettingRawValue: 12,
        policyArtifact: "best",
        policySourceId: "run-b",
        policySourceKind: "run",
        vehicleId: "wild_goose",
      },
    };

    const next = resetCourseEngineDrafts(current, [muteCity]);

    expect(next[courseSetupKey(muteCity)]).toEqual({
      ...muteCity,
      engineSettingRawValue: 64,
      policyArtifact: "latest",
      policySourceId: "run-a",
      policySourceKind: "run",
      vehicleId: "blue_falcon",
    });
    expect(next[courseSetupKey(silence)]).toBe(current[courseSetupKey(silence)]);
  });

  it("creates neutral empty drafts for previously untouched courses", () => {
    const portTown = { cupId: "queen", courseId: "port_town" };

    const next = resetCourseEngineDrafts({}, [portTown]);

    expect(next[courseSetupKey(portTown)]).toEqual({
      ...portTown,
      engineSettingRawValue: 64,
      policyArtifact: "best",
      policySourceId: "",
      policySourceKind: "run",
      vehicleId: "blue_falcon",
    });
  });
});

describe("sharedPolicySelectionDraft", () => {
  it("keeps a shared policy selection when course engines differ", () => {
    expect(
      sharedPolicySelectionDraft(
        [
          {
            courseId: "mute_city",
            cupId: "jack",
            engineSettingRawValue: 80,
            policyArtifact: "latest",
            policySourceId: "run-a",
            policySourceKind: "run",
            vehicleId: "blue_falcon",
          },
          {
            courseId: "silence",
            cupId: "jack",
            engineSettingRawValue: 103,
            policyArtifact: "latest",
            policySourceId: "run-a",
            policySourceKind: "run",
            vehicleId: "blue_falcon",
          },
        ],
        2,
      ),
    ).toEqual({
      policyArtifact: "latest",
      policySourceId: "run-a",
      policySourceKind: "run",
    });
  });

  it("returns null when the cup has mixed policy selections", () => {
    expect(
      sharedPolicySelectionDraft(
        [
          {
            courseId: "mute_city",
            cupId: "jack",
            engineSettingRawValue: 80,
            policyArtifact: "latest",
            policySourceId: "run-a",
            policySourceKind: "run",
            vehicleId: "blue_falcon",
          },
          {
            courseId: "silence",
            cupId: "jack",
            engineSettingRawValue: 80,
            policyArtifact: "latest",
            policySourceId: "run-b",
            policySourceKind: "run",
            vehicleId: "blue_falcon",
          },
        ],
        2,
      ),
    ).toBeNull();
  });
});
