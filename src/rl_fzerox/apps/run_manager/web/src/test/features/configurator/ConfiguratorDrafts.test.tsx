// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/ConfiguratorDrafts.test.tsx

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
import { cleanup, fireEvent, render, screen, waitFor } from "@/test/render";

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
});
