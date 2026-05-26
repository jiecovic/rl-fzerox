// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/Configurator.test.tsx
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Configurator } from "@/features/configurator/Configurator";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";
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

function reorderConfigKeys(config: ManagedRunConfig): ManagedRunConfig {
  return {
    reward: config.reward,
    policy: config.policy,
    observation: config.observation,
    train: config.train,
    tracks: config.tracks,
    vehicle: config.vehicle,
    action: config.action,
    environment: config.environment,
    preset_name: config.preset_name,
    seed: config.seed,
    version: config.version,
  };
}

function impalaLargePreviewFixture(): PolicyArchitecturePreview {
  const convLayers: PolicyArchitecturePreview["conv_layers"] = [
    {
      name: "conv1",
      kind: "conv",
      in_channels: 6,
      out_channels: 16,
      kernel_size: 3,
      stride: 1,
      padding: 1,
      post_activation: false,
      input_height: 72,
      input_width: 96,
      output_height: 72,
      output_width: 96,
      dropped_height: 0,
      dropped_width: 0,
      params: 880,
    },
    {
      name: "pool2",
      kind: "maxpool",
      in_channels: 16,
      out_channels: 16,
      kernel_size: 3,
      stride: 2,
      padding: 1,
      post_activation: true,
      input_height: 72,
      input_width: 96,
      output_height: 36,
      output_width: 48,
      dropped_height: 0,
      dropped_width: 0,
      params: 0,
    },
    {
      name: "res-pre3",
      kind: "residual_pre",
      in_channels: 16,
      out_channels: 16,
      kernel_size: 3,
      stride: 1,
      padding: 1,
      post_activation: true,
      input_height: 36,
      input_width: 48,
      output_height: 36,
      output_width: 48,
      dropped_height: 0,
      dropped_width: 0,
      params: 4640,
    },
  ];
  return {
    ...policyPreviewFixture,
    conv_layers: convLayers,
    flatten_dim: 27_648,
    image_features_dim: 27_648,
    image_shape: { height: 72, width: 96, channels: 6 },
  };
}

describe("Configurator", () => {
  beforeEach(() => {
    fetchPolicyPreviewMock.mockReset();
    fetchPolicyPreviewMock.mockResolvedValue(policyPreviewFixture);
  });

  afterEach(() => {
    vi.useRealTimers();
    cleanup();
  });

  it("debounces policy preview updates while editing config", async () => {
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

    await waitFor(() => expect(fetchPolicyPreviewMock).toHaveBeenCalledTimes(1));
    fetchPolicyPreviewMock.mockClear();

    fireEvent.click(screen.getByRole("button", { name: "Training" }));
    const clipRangeInput = screen.getByRole("spinbutton", { name: "Clip range" });
    fireEvent.change(clipRangeInput, { target: { value: "0.17" } });
    fireEvent.change(clipRangeInput, { target: { value: "0.18" } });

    expect(fetchPolicyPreviewMock).not.toHaveBeenCalled();
    await waitFor(() => expect(fetchPolicyPreviewMock).toHaveBeenCalledTimes(1));
    expect(fetchPolicyPreviewMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        train: expect.objectContaining({ clip_range: 0.18 }),
      }),
      expect.objectContaining({ signal: expect.any(Object) }),
    );
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

  it("does not mark a saved draft dirty when only config key order changes", async () => {
    const loadedDraft = draftFixture({
      config: managedRunConfigFixture,
      name: "ppo_allcups_recurrent",
    });
    const { rerender } = render(
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

    expect(screen.getByRole("button", { name: "Save draft" })).not.toHaveClass(
      "dirty-action-button",
    );

    rerender(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        loadedDraft={draftFixture({
          ...loadedDraft,
          config: reorderConfigKeys(managedRunConfigFixture),
        })}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Save draft" })).not.toHaveClass(
      "dirty-action-button",
    );
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

    await user.click(screen.getByRole("button", { name: "GP Race" }));
    await user.click(screen.getByRole("button", { name: "Enable X Cup" }));

    expect(screen.getByRole("button", { name: "Disable X Cup" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "GP Race" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("textbox", { name: "Generated courses" })).toHaveValue("6");
    expect(
      screen.getByText("Deterministic GP X Cup baselines materialized at training start."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mute City" })).toBeInTheDocument();
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

    const masterDifficulty = screen.getByRole("button", { name: "Master" });
    expect(masterDifficulty).toHaveAttribute("aria-disabled", "true");

    await user.click(screen.getByRole("button", { name: "GP Race" }));

    expect(masterDifficulty).toHaveAttribute("aria-disabled", "false");
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

  it("lets you configure episode-scoped state-feature dropout from the observation tab", async () => {
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

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.click(screen.getByText("track position"));

    const trackPositionPanel = screen.getByText("track position").closest("details");
    if (!(trackPositionPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing track position panel");
    }
    const edgeRatioInput = within(trackPositionPanel).getByRole("spinbutton", {
      name: "Edge ratio episode dropout",
    });
    expect(edgeRatioInput).toBeEnabled();
    expect(edgeRatioInput).toHaveValue(1);

    const edgeRatioRow = within(trackPositionPanel).getByText("Edge ratio").closest("tr");
    if (!(edgeRatioRow instanceof HTMLTableRowElement)) {
      throw new Error("Missing edge ratio row");
    }
    const edgeRatioIncludedToggle = within(edgeRatioRow).getByRole("checkbox", {
      name: "use entry as policy input",
    });
    const edgeRatioValueToggle = within(edgeRatioRow).getByRole("checkbox", {
      name: "Edge ratio uses real value",
    });

    expect(edgeRatioIncludedToggle).toBeEnabled();
    expect(edgeRatioIncludedToggle).toBeChecked();
    expect(edgeRatioValueToggle).toBeEnabled();
    expect(edgeRatioValueToggle).not.toBeChecked();

    await user.click(edgeRatioValueToggle);

    await waitFor(() => {
      expect(edgeRatioIncludedToggle).toBeChecked();
      expect(edgeRatioValueToggle).toBeChecked();
      expect(edgeRatioInput).toHaveValue(0);
    });

    await user.clear(edgeRatioInput);
    await user.type(edgeRatioInput, "0.25");
    edgeRatioInput.blur();

    await waitFor(() => expect(edgeRatioInput).toHaveValue(0.25));

    await user.clear(edgeRatioInput);
    await user.type(edgeRatioInput, "1");
    edgeRatioInput.blur();

    await waitFor(() => {
      expect(edgeRatioInput).toBeEnabled();
      expect(edgeRatioInput).toHaveValue(1);
      expect(edgeRatioValueToggle).not.toBeChecked();
    });

    await user.click(edgeRatioValueToggle);

    await waitFor(() => {
      expect(edgeRatioInput).toBeEnabled();
      expect(edgeRatioInput).toHaveValue(0);
      expect(edgeRatioValueToggle).toBeChecked();
    });
  });

  it.each([
    "four_way_categorical",
    "independent_buttons",
  ] as const)("shows split lean history rows for %s lean output", async (leanOutputMode) => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={{
          ...managedRunConfigFixture,
          action: {
            ...managedRunConfigFixture.action,
            lean_output_mode: leanOutputMode,
          },
        }}
        existingNames={[]}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.click(screen.getByText("control history"));

    const controlHistoryPanel = screen.getByText("control history").closest("details");
    if (!(controlHistoryPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing control history panel");
    }

    expect(within(controlHistoryPanel).getByText("Lean left at time t-1")).toBeInTheDocument();
    expect(within(controlHistoryPanel).getByText("Lean right at time t-1")).toBeInTheDocument();
    expect(within(controlHistoryPanel).queryByText("Lean at time t-1")).not.toBeInTheDocument();
    expect(within(controlHistoryPanel).getByText("7 entries")).toBeInTheDocument();
  });

  it("persists progress-scalar dropout without requiring a blur before saving", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="progress dropout draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.click(screen.getByText("track position"));

    const trackPositionPanel = screen.getByText("track position").closest("details");
    if (!(trackPositionPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing track position panel");
    }
    const lapProgressInput = within(trackPositionPanel).getByRole("spinbutton", {
      name: "Progress scalar episode dropout",
    });

    await user.clear(lapProgressInput);
    await user.type(lapProgressInput, "0.1");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "progress dropout draft",
        expect.objectContaining({
          observation: expect.objectContaining({
            state_feature_dropouts: expect.arrayContaining([
              { dropout_prob: 0.1, name: "track_position.lap_progress" },
            ]),
          }),
        }),
      ),
    );
  });

  it("persists auxiliary state loss settings from the observation table", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="aux loss draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.click(screen.getByText("track position"));

    const trackPositionPanel = screen.getByText("track position").closest("details");
    if (!(trackPositionPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing track position panel");
    }

    await user.click(
      within(trackPositionPanel).getByRole("checkbox", {
        name: "Edge ratio auxiliary loss enabled",
      }),
    );
    const weightInput = within(trackPositionPanel).getByRole("spinbutton", {
      name: "Edge ratio auxiliary loss weight",
    });
    await user.clear(weightInput);
    await user.type(weightInput, "0.25");
    await user.click(
      within(trackPositionPanel).getByRole("checkbox", {
        name: "grounded only",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "aux loss draft",
        expect.objectContaining({
          policy: expect.objectContaining({
            auxiliary_state_enabled: true,
            auxiliary_state_losses: expect.arrayContaining([
              {
                name: "track_position.edge_ratio",
                weight: 0.25,
                grounded_only: true,
              },
            ]),
          }),
        }),
      ),
    );
  });

  it("persists ground-height inclusion from the observation table", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="ground height draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.click(screen.getByText("track position"));

    const groundHeightRow = screen.getByText("Ground height").closest("tr");
    if (!(groundHeightRow instanceof HTMLTableRowElement)) {
      throw new Error("Missing ground height row");
    }

    await user.click(
      within(groundHeightRow).getByRole("checkbox", {
        name: "use entry as policy input",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "ground height draft",
        expect.objectContaining({
          observation: expect.objectContaining({
            state_components: expect.arrayContaining([
              expect.objectContaining({
                name: "track_position",
                included_features: expect.arrayContaining([
                  "track_position.height_above_ground_norm",
                ]),
              }),
            ]),
          }),
        }),
      ),
    );
  });

  it("persists image features without requiring a blur before saving", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="feature dim draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Policy" }));
    await user.click(screen.getByRole("button", { name: "Custom" }));

    const imageFeaturesInput = screen.getByRole("textbox", { name: "Image features" });
    await user.clear(imageFeaturesInput);
    await user.type(imageFeaturesInput, "1024");
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Image proj activation" }),
      "gelu",
    );
    await user.selectOptions(screen.getByRole("combobox", { name: "State activation" }), "gelu");
    await user.selectOptions(screen.getByRole("combobox", { name: "Fusion activation" }), "tanh");
    await user.selectOptions(screen.getByRole("combobox", { name: "Post-LN activation" }), "relu");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "feature dim draft",
        expect.objectContaining({
          policy: expect.objectContaining({
            features_dim: 1024,
            image_projection_activation: "gelu",
            state_activation: "gelu",
            fusion_activation: "tanh",
            layer_norm_activation: "relu",
          }),
        }),
      ),
    );
  });

  it("persists custom observation resolution through the observation tab", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="custom resolution draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.selectOptions(screen.getByLabelText("Resolution source"), "custom");

    const heightInput = screen.getByRole("textbox", { name: "Custom height" });
    const widthInput = screen.getByRole("textbox", { name: "Custom width" });
    await user.clear(heightInput);
    await user.type(heightInput, "72");
    await user.clear(widthInput);
    await user.type(widthInput, "96");

    expect(screen.getByText("208 x 296")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "custom resolution draft",
        expect.objectContaining({
          observation: expect.objectContaining({
            resolution: { mode: "custom", height: 72, width: 96 },
          }),
        }),
      ),
    );
  });

  it("persists original crop as a resolution source instead of a preset", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="source crop draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    await user.selectOptions(screen.getByLabelText("Resolution source"), "source_crop");
    expect(screen.getAllByText("208 x 296").length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "source crop draft",
        expect.objectContaining({
          observation: expect.objectContaining({
            resolution: { mode: "source_crop" },
          }),
        }),
      ),
    );
  });

  it("sets the IMPALA image geometry when selecting the large IMPALA CNN profile", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="impala large draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Policy" }));
    await user.selectOptions(screen.getByLabelText("CNN profile"), "impala_large");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "impala large draft",
        expect.objectContaining({
          observation: expect.objectContaining({
            resolution: { mode: "preset", preset: "crop_72x96" },
          }),
          policy: expect.objectContaining({
            conv_profile: "impala_large",
          }),
        }),
      ),
    );
  });

  it("copies preset CNN rows into a custom profile when edited", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());
    fetchPolicyPreviewMock.mockResolvedValue(impalaLargePreviewFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="edited preset cnn"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Policy" }));
    await user.selectOptions(screen.getByLabelText("CNN profile"), "impala_large");
    expect(await screen.findByLabelText("custom CNN layer 1 activation")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Edit as custom" }));
    const firstActivation = await screen.findByLabelText("custom CNN layer 1 activation");
    await user.selectOptions(firstActivation, "relu");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "edited preset cnn",
        expect.objectContaining({
          policy: expect.objectContaining({
            conv_profile: "custom",
            custom_cnn_final_relu: true,
            custom_conv_layers: expect.arrayContaining([
              expect.objectContaining({
                kind: "conv",
                post_activation: true,
              }),
              expect.objectContaining({
                kind: "residual_pre",
              }),
            ]),
          }),
        }),
      ),
    );
  });

  it("persists recent retention without requiring a blur before saving", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="recent retention draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Logging" }));
    await user.click(screen.getByRole("button", { name: "Keep recent" }));

    const retentionInput = screen.getByRole("textbox", { name: "Recent retention" });
    await user.clear(retentionInput);
    await user.type(retentionInput, "12");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "recent retention draft",
        expect.objectContaining({
          train: expect.objectContaining({
            recent_checkpoint_limit: 12,
            save_recent_checkpoints: true,
          }),
        }),
      ),
    );
  });

  it("persists PPO clip range without requiring a blur before saving", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="clip range draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Training" }));

    const clipRangeInput = screen.getByRole("spinbutton", { name: "Clip range" });
    await user.clear(clipRangeInput);
    await user.type(clipRangeInput, "0.17");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "clip range draft",
        expect.objectContaining({
          train: expect.objectContaining({
            clip_range: 0.17,
          }),
        }),
      ),
    );
  });

  it("launches with edited PPO clip range from an unsaved fork draft", async () => {
    const user = userEvent.setup();
    const onLaunchRun = vi.fn().mockResolvedValue(runFixture({ name: "run" }));

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        forkSourceArtifact="latest"
        forkSourceRunLabel="source run"
        initialDraftName="clip range fork"
        initialConfig={managedRunConfigFixture}
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={onLaunchRun}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Training" }));

    const clipRangeInput = screen.getByRole("spinbutton", { name: "Clip range" });
    await user.clear(clipRangeInput);
    await user.type(clipRangeInput, "0.17");
    await user.click(screen.getByRole("button", { name: "Train" }));

    await waitFor(() =>
      expect(onLaunchRun).toHaveBeenCalledWith(
        "clip range fork",
        expect.objectContaining({
          train: expect.objectContaining({
            clip_range: 0.17,
          }),
        }),
        null,
      ),
    );
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
    expect(screen.getByRole("checkbox", { name: "Force full throttle" })).toBeEnabled();
  });

  it("only allows spin output with 3-way lean", async () => {
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

    const spinOutput = screen.getByRole("checkbox", { name: "Spin in output" });
    const spinEnabled = screen.getByRole("checkbox", { name: "Spin enabled" });

    expect(spinOutput).toBeEnabled();
    expect(spinOutput).not.toBeChecked();
    await user.click(spinOutput);
    expect(spinOutput).toBeChecked();
    expect(spinEnabled).toBeChecked();

    await user.click(screen.getByRole("button", { name: "4-way categorical" }));
    await waitFor(() => {
      expect(spinOutput).toBeDisabled();
      expect(spinOutput).not.toBeChecked();
      expect(spinEnabled).toBeDisabled();
      expect(spinEnabled).not.toBeChecked();
    });

    await user.click(screen.getByRole("button", { name: "Independent buttons" }));
    await waitFor(() => {
      expect(spinOutput).toBeDisabled();
      expect(spinOutput).not.toBeChecked();
      expect(spinEnabled).toBeDisabled();
      expect(spinEnabled).not.toBeChecked();
    });

    await user.click(screen.getByRole("button", { name: "3-way axis" }));
    await waitFor(() => {
      expect(spinOutput).toBeEnabled();
      expect(spinOutput).not.toBeChecked();
      expect(spinEnabled).toBeDisabled();
    });
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
      config: {
        ...managedRunConfigFixture,
        policy: {
          ...managedRunConfigFixture.policy,
          conv_profile: "custom",
        },
      },
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
    const gasToggle = within(gasRow).getByRole("checkbox", {
      name: "use entry as policy input",
    });
    expect(gasToggle).toBeDisabled();
    expect(gasToggle).toBeChecked();
    expect(
      within(gasRow).getByRole("spinbutton", { name: "Gas at time t-1 episode dropout" }),
    ).toBeEnabled();
    expect(
      within(gasRow).getByRole("checkbox", { name: "Gas at time t-1 uses real value" }),
    ).toBeEnabled();

    await user.click(screen.getByText("track position"));
    const trackPositionPanel = screen.getByText("track position").closest("details");
    if (!(trackPositionPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing track position panel");
    }
    expect(
      screen.getByRole("checkbox", { name: "auxiliary state supervision enabled" }),
    ).toBeEnabled();
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
    const edgeRatioAuxToggle = within(trackPositionPanel).getByRole("checkbox", {
      name: "Edge ratio auxiliary loss enabled",
    });
    expect(edgeRatioAuxToggle).toBeEnabled();
    const groundHeightRow = within(trackPositionPanel).getByText("Ground height").closest("tr");
    if (!(groundHeightRow instanceof HTMLTableRowElement)) {
      throw new Error("Missing ground height row");
    }
    expect(
      within(groundHeightRow).getByRole("checkbox", { name: "use entry as policy input" }),
    ).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Policy" }));
    expect(screen.getByRole("combobox", { name: "CNN profile" })).toBeDisabled();
    expect(screen.getByLabelText("custom CNN layer 1 output channels")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add custom CNN conv layer" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Activation" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Action" }));
    await user.click(screen.getByText("Control family"));
    await user.click(screen.getByText("Auxiliary branches"));

    const throttleModeGroup = screen.getByRole("group", { name: "Throttle mode" });
    expect(
      within(throttleModeGroup).getByRole("button", {
        name: configMetadataFixture.drive_modes[0].label,
      }),
    ).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Force full throttle" })).toBeEnabled();
    expect(screen.getByRole("checkbox", { name: "Boost in output" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Boost enabled" })).toBeEnabled();
  });

  it("locks checkpoint-incompatible controls before a fork draft is saved", async () => {
    const user = userEvent.setup();

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        forkSourceArtifact="latest"
        forkSourceRunLabel="source run"
        initialConfig={managedRunConfigFixture}
        initialDraftName="source run fork"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={vi.fn()}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Observation" }));
    expect(screen.getByRole("combobox", { name: "Input resolution" })).toBeDisabled();

    await user.click(screen.getByText("track position"));
    const trackPositionPanel = screen.getByText("track position").closest("details");
    if (!(trackPositionPanel instanceof HTMLDetailsElement)) {
      throw new Error("Missing track position panel");
    }
    const edgeRatioRow = within(trackPositionPanel).getByText("Edge ratio").closest("tr");
    if (!(edgeRatioRow instanceof HTMLTableRowElement)) {
      throw new Error("Missing edge ratio row");
    }

    expect(
      within(edgeRatioRow).getByRole("checkbox", { name: "use entry as policy input" }),
    ).toBeDisabled();
    expect(
      within(edgeRatioRow).getByRole("checkbox", { name: "Edge ratio uses real value" }),
    ).toBeEnabled();
  });
});
