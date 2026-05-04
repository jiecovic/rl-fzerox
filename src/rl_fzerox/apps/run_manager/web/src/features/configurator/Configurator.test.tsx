import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Configurator } from "@/features/configurator/Configurator";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
  runFixture,
} from "@/test/fixtures";

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
        onLaunchRun={launchRunMock()}
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
        onLaunchRun={launchRunMock()}
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
      screen.queryByText("This draft name is already used by another draft or open editor."),
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
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={onUpdateDraft}
      />,
    );

    expect(
      screen.getByText("This draft name is already used by another draft or open editor."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save draft" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Train" })).toBeEnabled();

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

  it("switches the tracks tab into X Cup mode", async () => {
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

  it("separates head presence from runtime masking in the action tab", async () => {
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

    await user.click(screen.getByRole("button", { name: "Action" }));
    await user.click(screen.getByText("Auxiliary branches"));

    await user.click(screen.getByRole("checkbox", { name: "Boost enabled" }));
    expect(screen.getByRole("checkbox", { name: "Boost enabled" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Boost in output" })).toBeChecked();

    await user.click(screen.getByRole("checkbox", { name: "Force full throttle" }));
    expect(screen.getByRole("checkbox", { name: "Force full throttle" })).toBeChecked();
  });

  it("lets you switch image features from auto to a custom width", async () => {
    const user = userEvent.setup();

    const { container } = render(
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

    await user.click(screen.getByRole("button", { name: "Policy" }));
    await user.click(
      within(screen.getByRole("group", { name: "Image features mode" })).getByRole("button", {
        name: "Custom",
      }),
    );

    const imageFeatures = container.querySelector(".feature-dim-input");
    if (!(imageFeatures instanceof HTMLInputElement)) {
      throw new Error("Missing feature-dim input");
    }
    expect(imageFeatures).toHaveValue("512");

    await user.clear(imageFeatures);
    await user.type(imageFeatures, "640");
    imageFeatures.blur();

    await waitFor(() => expect(imageFeatures).toHaveValue("640"));
  });

  it("locks only checkpoint-incompatible fork controls", async () => {
    const user = userEvent.setup();
    const loadedDraft = draftFixture({
      source_artifact: "latest",
      source_run_id: "run-001",
      source_num_timesteps: 123_456,
    });

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={loadedDraft}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    expect(screen.getByRole("combobox", { name: "Input resolution" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Resize filter" })).toBeEnabled();

    await user.click(screen.getByText("control history"));
    const controlHistoryPanel = screen.getByText("control history").closest("details");
    if (!(controlHistoryPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing control history panel");
    }
    expect(
      within(controlHistoryPanel).getByRole("checkbox", { name: "category enabled" }),
    ).toBeDisabled();
    const gasRow = within(controlHistoryPanel).getByText("Gas at time t-1").closest("tr");
    if (!(gasRow instanceof HTMLTableRowElement)) {
      throw new Error("Missing gas row");
    }
    const gasToggle = within(gasRow).getByRole("checkbox", { name: "entry enabled" });
    expect(gasToggle).toBeEnabled();
    expect(gasToggle).toBeChecked();
    await user.click(gasToggle);
    expect(gasToggle).not.toBeChecked();

    await user.click(screen.getByText("track position"));
    const trackPositionPanel = screen.getByText("track position").closest("details");
    if (!(trackPositionPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing track position panel");
    }
    const lapProgressRow = within(trackPositionPanel).getByText("Progress scalar").closest("tr");
    if (!(lapProgressRow instanceof HTMLTableRowElement)) {
      throw new Error("Missing lap progress row");
    }
    const progressSourceGroup = within(lapProgressRow).getByRole("group", {
      name: "Progress scalar source",
    });
    await user.click(within(progressSourceGroup).getByRole("button", { name: "Lap segment" }));
    expect(within(progressSourceGroup).getByRole("button", { name: "Lap segment" })).toHaveClass(
      "active",
    );

    await user.click(screen.getByRole("button", { name: "Policy" }));
    expect(screen.getByRole("combobox", { name: "CNN profile" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Activation" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Action" }));
    await user.click(screen.getByText("Control family"));
    await user.click(screen.getByText("Auxiliary branches"));

    expect(
      screen.getByRole("button", { name: configMetadataFixture.drive_modes[0].label }),
    ).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Force full throttle" })).toBeEnabled();
    expect(screen.getByRole("checkbox", { name: "Boost in output" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Boost enabled" })).toBeEnabled();
  });
});
