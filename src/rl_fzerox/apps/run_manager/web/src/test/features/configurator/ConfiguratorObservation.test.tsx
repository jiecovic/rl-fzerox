// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/ConfiguratorObservation.test.tsx

import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Configurator } from "@/features/configurator/Configurator";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
  runFixture,
} from "@/test/fixtures";
import { cleanup, render, screen, waitFor, within } from "@/test/render";

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
    {
      name: "act4",
      kind: "activation",
      in_channels: 16,
      out_channels: 16,
      kernel_size: 1,
      stride: 1,
      padding: 0,
      post_activation: true,
      activation: "relu",
      input_height: 36,
      input_width: 48,
      output_height: 36,
      output_width: 48,
      dropped_height: 0,
      dropped_width: 0,
      params: 0,
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
    await user.click(screen.getByRole("radio", { name: "Custom" }));

    const imageFeaturesInput = screen.getByRole("textbox", { name: "Image features" });
    await user.clear(imageFeaturesInput);
    await user.type(imageFeaturesInput, "1024");
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Projection activation" }),
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
            custom_conv_layers: expect.arrayContaining([
              expect.objectContaining({
                kind: "conv",
                post_activation: true,
              }),
              expect.objectContaining({
                kind: "residual_pre",
              }),
              expect.objectContaining({
                kind: "activation",
                activation: "relu",
              }),
            ]),
          }),
        }),
      ),
    );
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
      within(screen.getByRole("group", { name: "Image features mode" })).getByRole("radio", {
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
});
