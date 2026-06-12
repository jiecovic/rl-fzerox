// web/run-manager/src/test/widgets/configurator/ConfiguratorTrainingAction.test.tsx

import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ManagedRunConfig } from "@/shared/api/contract";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
  runFixture,
} from "@/test/fixtures";
import { cleanup, render, screen, waitFor, within } from "@/test/render";
import { Configurator } from "@/widgets/configurator/Configurator";

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

    const clipRangeInput = screen.getByRole("textbox", { name: "Clip range" });
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

    const clipRangeInput = screen.getByRole("textbox", { name: "Clip range" });
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

  it("stores action entropy group weights from the action tab", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        initialDraftName="entropy group draft"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Training" }));
    expect(screen.queryByText("Entropy groups")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Action" }));
    const entropyPanel = screen.getByText("Entropy groups").closest("section");
    if (!(entropyPanel instanceof HTMLElement)) {
      throw new Error("Missing entropy groups panel");
    }

    const pitchGroup = within(entropyPanel).getByRole("region", {
      name: "Pitch entropy group",
    });
    expect(within(pitchGroup).getByText("Effective coeff")).toBeInTheDocument();
    expect(within(pitchGroup).getByText("1.00e-2")).toBeInTheDocument();

    const pitchEntropy = within(pitchGroup).getByRole("button", { name: "Pitch" });
    expect(pitchEntropy).toHaveAttribute("aria-pressed", "true");
    await user.click(pitchEntropy);
    expect(pitchEntropy).toHaveAttribute("aria-pressed", "false");
    expect(within(pitchGroup).getByText("0")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "entropy group draft",
        expect.objectContaining({
          train: expect.objectContaining({
            entropy_group_weights: expect.objectContaining({ pitch: 0 }),
          }),
        }),
      ),
    );
  });

  it("allows grounded pitch actor loss edits on checkpoint forks", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn().mockResolvedValue(draftFixture());
    const forkConfig: ManagedRunConfig = {
      ...managedRunConfigFixture,
      action: {
        ...managedRunConfigFixture.action,
        include_pitch: true,
        pitch_mode: "continuous",
      },
    };

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        existingNames={[]}
        forkSourceArtifact="latest"
        forkSourceRunLabel="source run"
        initialConfig={forkConfig}
        initialDraftName="pitch loss fork"
        loadedDraft={null}
        metadata={configMetadataFixture}
        onLaunchRun={launchRunMock()}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Action" }));
    await user.click(screen.getByText("Auxiliary branches"));

    const pitchPanel = screen.getByText("Pitch control").closest("section");
    if (!(pitchPanel instanceof HTMLElement)) {
      throw new Error("Missing pitch control panel");
    }
    expect(within(pitchPanel).getByRole("radio", { name: "Continuous" })).toBeDisabled();

    const enableLoss = within(pitchPanel).getByRole("button", {
      name: "Grounded pitch neutral loss",
    });
    expect(enableLoss).toBeEnabled();
    expect(enableLoss).toHaveAttribute("aria-pressed", "false");
    await user.click(enableLoss);
    expect(enableLoss).toHaveAttribute("aria-pressed", "true");

    const lossWeight = within(pitchPanel).getByRole("textbox", { name: "Mean loss weight" });
    expect(lossWeight).toBeEnabled();
    expect(lossWeight).toHaveValue("0.01");

    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() =>
      expect(onSaveDraft).toHaveBeenCalledWith(
        "pitch loss fork",
        expect.objectContaining({
          train: expect.objectContaining({
            actor_regularization: expect.objectContaining({
              grounded_pitch_neutral_loss_weight: 0.01,
            }),
          }),
        }),
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
    await user.click(screen.getByText("Control family"));
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

    await user.click(screen.getByRole("radio", { name: "4-way categorical" }));
    await waitFor(() => {
      expect(spinOutput).toBeDisabled();
      expect(spinOutput).not.toBeChecked();
      expect(spinEnabled).toBeDisabled();
      expect(spinEnabled).not.toBeChecked();
    });

    await user.click(screen.getByRole("radio", { name: "Independent buttons" }));
    await waitFor(() => {
      expect(spinOutput).toBeDisabled();
      expect(spinOutput).not.toBeChecked();
      expect(spinEnabled).toBeDisabled();
      expect(spinEnabled).not.toBeChecked();
    });

    await user.click(screen.getByRole("radio", { name: "3-way axis" }));
    await waitFor(() => {
      expect(spinOutput).toBeEnabled();
      expect(spinOutput).not.toBeChecked();
      expect(spinEnabled).toBeDisabled();
    });
  });

  it("updates the no-spin logit probability note while typing", async () => {
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
    await user.click(screen.getByRole("checkbox", { name: "Spin in output" }));

    expect(screen.getByText("logit 0 -> idle 33.3%, left/right 33.3% each")).toBeVisible();

    const noSpinLogit = screen.getByRole("textbox", { name: "No-spin logit" });
    await user.clear(noSpinLogit);
    await user.type(noSpinLogit, "0.5");

    expect(screen.getByText("logit +0.5 -> idle 45.2%, left/right 27.4% each")).toBeVisible();
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
      within(gasRow).getByRole("textbox", { name: "Gas at time t-1 episode dropout" }),
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
    await user.click(within(progressSourceGroup).getByRole("radio", { name: "Lap segment" }));
    expect(within(progressSourceGroup).getByRole("radio", { name: "Lap segment" })).toHaveAttribute(
      "aria-checked",
      "true",
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
    expect(screen.getByRole("combobox", { name: "Aux/policy/value activation" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Action" }));
    await user.click(screen.getByText("Control family"));
    await user.click(screen.getByText("Auxiliary branches"));

    const throttleModeGroup = screen.getByRole("group", { name: "Throttle mode" });
    expect(
      within(throttleModeGroup).getByRole("radio", {
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
