// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/ConfiguratorTracks.test.tsx

import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Configurator } from "@/features/configurator/Configurator";
import type { ManagedRunConfig } from "@/shared/api/contract";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
  runFixture,
} from "@/test/fixtures";
import { cleanup, render, screen, waitFor } from "@/test/render";

const fetchPolicyPreviewMock = vi.fn();

function launchRunMock() {
  return vi.fn().mockResolvedValue(runFixture({ name: "run" }));
}

vi.mock("@/shared/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/client")>("@/shared/api/client");
  return {
    ...actual,
    fetchPolicyPreview: (...args: Parameters<typeof actual.fetchPolicyPreview>) =>
      fetchPolicyPreviewMock(...args),
  };
});

describe("Configurator", () => {
  beforeEach(() => {
    fetchPolicyPreviewMock.mockReset();
    fetchPolicyPreviewMock.mockResolvedValue(policyPreviewFixture);
  });

  afterEach(() => {
    vi.useRealTimers();
    cleanup();
  });

  it("lets you edit the selected course pool from the tracks tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Tracks" }));

    const muteCity = screen.getByRole("button", { name: "Mute City" });
    const silence = screen.getByRole("button", { name: "Silence" });
    expect(muteCity).toHaveAttribute("aria-pressed", "true");
    expect(silence).toHaveAttribute("aria-pressed", "true");

    await user.click(silence);
    expect(silence).toHaveAttribute("aria-pressed", "false");

    await user.click(screen.getByRole("button", { name: "Select all" }));
    expect(screen.getByRole("button", { name: "Silence" })).toHaveAttribute("aria-pressed", "true");
  });

  it("enables generated X Cup courses for GP race pools", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Tracks" }));
    expect(screen.getByRole("button", { name: "Enable X Cup" })).toBeDisabled();

    await user.click(screen.getByRole("radio", { name: "GP Race" }));
    await user.click(screen.getByRole("button", { name: "Enable X Cup" }));

    expect(screen.getByRole("button", { name: "Disable X Cup" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("radio", { name: "GP Race" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("textbox", { name: "Generated courses" })).toHaveValue("6");
    expect(
      screen.getByText("Deterministic GP X Cup baselines materialized at training start."),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Auto regenerate" }));
    expect(screen.getByRole("textbox", { name: "Completion threshold" })).toHaveValue("0.9");
    expect(screen.getByRole("textbox", { name: "Min episodes" })).toHaveValue("24");
    expect(screen.getByRole("button", { name: "Episode cap" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    await user.click(screen.getByRole("button", { name: "Episode cap" }));
    expect(screen.getByRole("textbox", { name: "Max episodes" })).toHaveValue("100");
    expect(screen.getByRole("button", { name: "Mute City" })).toBeInTheDocument();
  });

  it("clears X Cup auto regeneration when X Cup is disabled", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Tracks" }));
    await user.click(screen.getByRole("radio", { name: "GP Race" }));
    await user.click(screen.getByRole("button", { name: "Enable X Cup" }));
    await user.click(screen.getByRole("button", { name: "Auto regenerate" }));
    await user.click(screen.getByRole("button", { name: "Disable X Cup" }));
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() => expect(onSaveDraft).toHaveBeenCalled());
    const savedConfig = onSaveDraft.mock.calls[0]?.[1] as ManagedRunConfig;
    expect(savedConfig.tracks.include_x_cup).toBe(false);
    expect(savedConfig.tracks.x_cup_auto_regeneration.enabled).toBe(false);
  });

  it("enables GP difficulty selection only in GP race mode", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Tracks" }));

    expect(screen.getByRole("button", { name: "Master" })).toBeDisabled();

    await user.click(screen.getByRole("radio", { name: "GP Race" }));

    const masterDifficulty = screen.getByRole("button", { name: "Master" });
    expect(masterDifficulty).toBeEnabled();
    await user.click(masterDifficulty);
    expect(masterDifficulty).toHaveAttribute("aria-pressed", "true");
  });

  it("supports pooled vehicle selection and random engine ranges in the vehicle tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Vehicle" }));
    await user.click(screen.getByRole("button", { name: "Select all" }));
    await user.click(screen.getByRole("button", { name: "Golden Fox" }));
    await user.click(screen.getByRole("checkbox", { name: "Random range" }));

    expect(screen.getByRole("button", { name: "Golden Fox" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByRole("button", { name: "Wild Goose" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("textbox", { name: "Engine range minimum" })).toHaveValue("20");
    expect(screen.getByRole("textbox", { name: "Engine range maximum" })).toHaveValue("80");
  });

  it("lets you edit episode bounds from the environment tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Environment" }));

    const maxEpisodeSteps = screen.getByRole("textbox", { name: "Max episode steps" });
    await user.clear(maxEpisodeSteps);
    await user.type(maxEpisodeSteps, "18000");
    maxEpisodeSteps.blur();

    await waitFor(() => expect(maxEpisodeSteps).toHaveValue("18,000"));

    const stallToggle = screen.getByRole("button", { name: "Enable no-progress truncation" });
    expect(stallToggle).toHaveAttribute("aria-pressed", "true");

    await user.click(stallToggle);
    expect(stallToggle).toHaveAttribute("aria-pressed", "false");
  });
});
