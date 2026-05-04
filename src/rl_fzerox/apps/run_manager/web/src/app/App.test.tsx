import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
  runFixture,
  runMetricSampleFixture,
} from "@/test/fixtures";

const loadManagerDataMock = vi.fn();
const createDraftWithSourceMock = vi.fn();
const updateDraftWithSourceMock = vi.fn();
const deleteDraftMock = vi.fn();
const deleteRunMock = vi.fn();
const fetchRunsMock = vi.fn();
const fetchRunMetricsMock = vi.fn();
const fetchRunTrackSamplingStateMock = vi.fn();
const fetchPolicyPreviewMock = vi.fn();
const launchRunMock = vi.fn();
const openRunDirectoryMock = vi.fn();
const renameRunMock = vi.fn();
const resumeRunMock = vi.fn();
const stopRunMock = vi.fn();

vi.mock("@/app/managerData", () => ({
  loadManagerData: () => loadManagerDataMock(),
}));

vi.mock("@/shared/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/client")>("@/shared/api/client");
  return {
    ...actual,
    createDraftWithSource: (
      name: string,
      config: typeof managedRunConfigFixture,
      sourceRunId: string | null,
      sourceArtifact: "latest" | "best" | null,
    ) => createDraftWithSourceMock(name, config, sourceRunId, sourceArtifact),
    deleteDraft: (id: string) => deleteDraftMock(id),
    deleteRun: (id: string) => deleteRunMock(id),
    fetchRuns: () => fetchRunsMock(),
    fetchRunMetrics: (runId: string) => fetchRunMetricsMock(runId),
    fetchRunTrackSamplingState: (runId: string) => fetchRunTrackSamplingStateMock(runId),
    fetchPolicyPreview: (config: typeof managedRunConfigFixture) => fetchPolicyPreviewMock(config),
    launchRun: (...args: unknown[]) => launchRunMock(...args),
    openRunDirectory: (runId: string) => openRunDirectoryMock(runId),
    renameRun: (runId: string, name: string) => renameRunMock(runId, name),
    resumeRun: (runId: string) => resumeRunMock(runId),
    stopRun: (runId: string) => stopRunMock(runId),
    updateDraftWithSource: (
      id: string,
      name: string,
      config: typeof managedRunConfigFixture,
      sourceRunId: string | null,
      sourceArtifact: "latest" | "best" | null,
    ) => updateDraftWithSourceMock(id, name, config, sourceRunId, sourceArtifact),
  };
});

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    loadManagerDataMock.mockResolvedValue({
      drafts: [draftFixture()],
      metadata: configMetadataFixture,
      runs: [],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });
    createDraftWithSourceMock.mockImplementation(
      async (
        name: string,
        _config: typeof managedRunConfigFixture,
        sourceRunId: string | null,
        sourceArtifact: "latest" | "best" | null,
      ) =>
        draftFixture({
          id: name,
          name,
          source_artifact: sourceArtifact,
          source_run_id: sourceRunId,
        }),
    );
    updateDraftWithSourceMock.mockImplementation(async (id: string, name: string) =>
      draftFixture({ id, name }),
    );
    deleteDraftMock.mockResolvedValue(undefined);
    deleteRunMock.mockResolvedValue(undefined);
    fetchRunsMock.mockResolvedValue([]);
    fetchRunMetricsMock.mockResolvedValue([runMetricSampleFixture()]);
    fetchRunTrackSamplingStateMock.mockResolvedValue(null);
    fetchPolicyPreviewMock.mockResolvedValue(policyPreviewFixture);
    launchRunMock.mockResolvedValue(runFixture());
    openRunDirectoryMock.mockResolvedValue(undefined);
    renameRunMock.mockResolvedValue(runFixture({ name: "renamed run" }));
    resumeRunMock.mockResolvedValue(runFixture({ status: "running", pending_command: null }));
    stopRunMock.mockResolvedValue(runFixture({ pending_command: "stop" }));
  });

  afterEach(() => {
    cleanup();
  });

  it("keeps multiple draft editors open as closable workspace tabs", async () => {
    const user = userEvent.setup();

    render(<App />);

    await screen.findByRole("button", { name: "Create draft" });

    await user.click(screen.getByRole("button", { name: "Create draft" }));
    expect(await screen.findByRole("textbox", { name: "Run name" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Draft · ppo_allcups_recurrent 2" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Drafts" }));
    await user.click(screen.getByRole("button", { name: /50,000,000 steps/i }));

    expect(await screen.findByRole("textbox", { name: "Run name" })).toHaveValue(
      "ppo_allcups_recurrent",
    );
    expect(
      screen.getByRole("button", { name: "Draft · ppo_allcups_recurrent" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Draft · ppo_allcups_recurrent 2" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Drafts" }));
    expect(screen.getByRole("button", { name: "Create draft" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Draft · ppo_allcups_recurrent" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Draft · ppo_allcups_recurrent 2" }),
    ).toBeInTheDocument();
  });

  it("opens a fork as a draft editor instead of launching immediately", async () => {
    const user = userEvent.setup();
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [runFixture({ id: "run-001", name: "ppo_test_1" })],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    await user.click(screen.getByRole("button", { name: "Open run ppo_test_1" }));
    await user.click(await screen.findByRole("button", { name: "Fork latest checkpoint" }));

    expect(launchRunMock).not.toHaveBeenCalled();
    expect(createDraftWithSourceMock).toHaveBeenCalledWith(
      "ppo_test_1 fork",
      managedRunConfigFixture,
      "run-001",
      "latest",
    );
    expect(await screen.findByText(/Forked from/i)).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Run name" })).toHaveValue("ppo_test_1 fork");
    await user.click(within(workspaceTabs).getByRole("button", { name: "Drafts" }));
    expect(screen.getByRole("button", { name: /^ppo_test_1 fork/i })).toBeInTheDocument();
  });

  it("shows forked sim game time with lineage step offset included", async () => {
    const user = userEvent.setup();
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("run fixture must include runtime data");
    }
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [
        runFixture({
          id: "run-001",
          status: "stopped",
          lineage_step_offset: 180_000,
          stopped_at: "2026-05-03T19:05:24+00:00",
          config: {
            ...managedRunConfigFixture,
            action: {
              ...managedRunConfigFixture.action,
              action_repeat: 2,
            },
          },
          runtime: {
            ...runtime,
            num_timesteps: 60_000,
          },
        }),
      ],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    await user.click(screen.getByRole("button", { name: "Open run ppo_test_1" }));

    const simMetric = (await screen.findByText("Sim game time")).closest(".run-runtime-metric");
    expect(simMetric?.textContent).toContain("2h 13m");
    const ratioMetric = screen.getByText("Sim / wall").closest(".run-runtime-metric");
    expect(ratioMetric?.textContent).toContain("2.52x");
    expect(screen.getByText(/lineage steps ·/i).textContent).toContain(
      "240,000 lineage steps · 60,000 / 50,000,000 local fork steps",
    );
  });
});
