import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Configurator } from "@/features/configurator/Configurator";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
} from "@/test/fixtures";

const fetchPolicyPreviewMock = vi.fn();

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
    cleanup();
  });

  it("loads an opened draft into the configurator and updates it in place", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn();
    const onUpdateDraft = vi.fn().mockResolvedValue(draftFixture());
    const loadedDraft = draftFixture({ name: "ppo_allcups_recurrent" });

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={loadedDraft}
        metadata={configMetadataFixture}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={onUpdateDraft}
      />,
    );

    const runName = screen.getByLabelText("Run name");
    await waitFor(() => expect(runName).toHaveValue("ppo_allcups_recurrent"));

    await user.clear(runName);
    await user.type(runName, "ppo_allcups_recurrent_v2");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onUpdateDraft).toHaveBeenCalledWith(
        "draft-001",
        "ppo_allcups_recurrent_v2",
        managedRunConfigFixture,
      ),
    );
    expect(onSaveDraft).not.toHaveBeenCalled();
  });

  it("does not show a duplicate-name error when reopening an unchanged draft", async () => {
    const loadedDraft = draftFixture({ name: "ppo_allcups_recurrent" });

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={["ppo_allcups_recurrent", "other_open_editor"]}
        loadedDraft={loadedDraft}
        metadata={configMetadataFixture}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await waitFor(() =>
      expect(screen.getByRole("textbox", { name: "Run name" })).toHaveValue(
        "ppo_allcups_recurrent",
      ),
    );

    expect(
      screen.queryByText("This name is already used by another draft, run, or open editor."),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save draft" })).toBeEnabled();
  });

  it("blocks saving a duplicate draft name before hitting the API", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn();
    const onUpdateDraft = vi.fn();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={["ppo_allcups_recurrent"]}
        initialDraftName="ppo_allcups_recurrent"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={onUpdateDraft}
      />,
    );

    expect(
      screen.getByText("This name is already used by another draft, run, or open editor."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save draft" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Save draft" }));
    expect(onSaveDraft).not.toHaveBeenCalled();
  });

  it("lets you edit the selected course pool from the tracks tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
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

  it("switches the tracks tab into X Cup mode", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Tracks" }));
    await user.click(screen.getByRole("button", { name: "X Cup" }));

    expect(screen.getByRole("button", { name: "GP Race" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Time Attack" })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(screen.getByText("X Cup random generator")).toBeInTheDocument();
    expect(
      screen.getByText(
        "The game generates six courses at runtime, so there is no fixed roster to toggle here.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Mute City" })).not.toBeInTheDocument();
  });

  it("supports pooled vehicle selection and random engine ranges in the vehicle tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
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
    expect(screen.getByRole("spinbutton", { name: "Engine range minimum" })).toHaveValue(20);
    expect(screen.getByRole("spinbutton", { name: "Engine range maximum" })).toHaveValue(80);
  });

  it("lets you edit episode bounds from the environment tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
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

  it("separates head presence from runtime masking in the action tab", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Action" }));
    const steeringDiscreteButton = screen
      .getAllByRole("button", { name: "Discrete" })
      .find((button) => button.getAttribute("aria-pressed") === "false");
    if (steeringDiscreteButton === undefined) {
      throw new Error("Missing discrete steering button");
    }
    await user.click(steeringDiscreteButton);

    expect(screen.getByRole("checkbox", { name: "Pitch in output" })).toBeChecked();

    await user.click(screen.getByRole("checkbox", { name: "Boost enabled" }));
    expect(screen.getByRole("checkbox", { name: "Boost enabled" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Boost in output" })).toBeChecked();

    await user.click(screen.getByRole("checkbox", { name: "Pitch in output" }));
    expect(screen.getByRole("checkbox", { name: "Pitch in output" })).not.toBeChecked();

    await user.click(screen.getByRole("checkbox", { name: "Force full throttle" }));
    expect(screen.getByRole("checkbox", { name: "Force full throttle" })).toBeChecked();
  });

  it("lets you switch image features from auto to a custom width", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Policy" }));
    await user.click(screen.getByRole("button", { name: "Custom" }));

    const imageFeatures = screen.getByRole("textbox", { name: "Image features" });
    expect(imageFeatures).toHaveValue("512");

    await user.clear(imageFeatures);
    await user.type(imageFeatures, "640");
    imageFeatures.blur();

    await waitFor(() =>
      expect(screen.getByRole("textbox", { name: "Image features" })).toHaveValue("640"),
    );
  });
});
