// src/rl_fzerox/apps/run_manager/web/src/test/app/App.test.tsx
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import { ApiSchemaMismatchError } from "@/shared/api/client";
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
const fetchRunMock = vi.fn();
const fetchRunsMock = vi.fn();
const fetchRunMetricsMock = vi.fn();
const fetchRunTrackSamplingStateMock = vi.fn();
const fetchPolicyPreviewMock = vi.fn();
const launchRunMock = vi.fn();
const openRunDirectoryMock = vi.fn();
const renameRunMock = vi.fn();
const resumeRunMock = vi.fn();
const stopRunMock = vi.fn();
const subscribeRunLiveUpdatesMock = vi.fn();
const watchRunMock = vi.fn();

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
    fetchRun: (runId: string) => fetchRunMock(runId),
    fetchRuns: () => fetchRunsMock(),
    fetchRunMetrics: (runId: string) => fetchRunMetricsMock(runId),
    fetchRunTrackSamplingState: (runId: string) => fetchRunTrackSamplingStateMock(runId),
    fetchPolicyPreview: (config: typeof managedRunConfigFixture) => fetchPolicyPreviewMock(config),
    launchRun: (...args: unknown[]) => launchRunMock(...args),
    openRunDirectory: (runId: string) => openRunDirectoryMock(runId),
    renameRun: (runId: string, name: string) => renameRunMock(runId, name),
    resumeRun: (runId: string) => resumeRunMock(runId),
    stopRun: (runId: string) => stopRunMock(runId),
    subscribeRunLiveUpdates: (options: unknown) => subscribeRunLiveUpdatesMock(options),
    watchRun: (runId: string, artifact: "latest" | "best") => watchRunMock(runId, artifact),
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
        config: typeof managedRunConfigFixture,
        sourceRunId: string | null,
        sourceArtifact: "latest" | "best" | null,
      ) =>
        draftFixture({
          config,
          id: name,
          name,
          source_artifact: sourceArtifact,
          source_run_id: sourceRunId,
        }),
    );
    updateDraftWithSourceMock.mockImplementation(
      async (id: string, name: string, config: typeof managedRunConfigFixture) =>
        draftFixture({ config, id, name }),
    );
    deleteDraftMock.mockResolvedValue(undefined);
    deleteRunMock.mockResolvedValue(undefined);
    fetchRunMock.mockResolvedValue(runFixture());
    fetchRunsMock.mockResolvedValue([]);
    fetchRunMetricsMock.mockResolvedValue([runMetricSampleFixture()]);
    fetchRunTrackSamplingStateMock.mockResolvedValue(null);
    fetchPolicyPreviewMock.mockResolvedValue(policyPreviewFixture);
    launchRunMock.mockResolvedValue(runFixture());
    openRunDirectoryMock.mockResolvedValue(undefined);
    renameRunMock.mockResolvedValue(runFixture({ name: "renamed run" }));
    resumeRunMock.mockResolvedValue(runFixture({ status: "running", pending_command: null }));
    stopRunMock.mockResolvedValue(runFixture({ pending_command: "stop" }));
    subscribeRunLiveUpdatesMock.mockReturnValue(() => undefined);
    watchRunMock.mockResolvedValue("started");
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
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);
    await user.click(await screen.findByRole("button", { name: "Fork latest checkpoint" }));

    expect(launchRunMock).not.toHaveBeenCalled();
    expect(createDraftWithSourceMock).not.toHaveBeenCalled();
    expect(await screen.findByText(/Forked from/i)).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Run name" })).toHaveValue("ppo_test_1 fork");
    await user.click(within(workspaceTabs).getByRole("button", { name: "Drafts" }));
    expect(
      screen.getByText("No drafts yet. Create one to open the configurator."),
    ).toBeInTheDocument();
  });

  it("saves and launches an unsaved fork using its in-memory fork source", async () => {
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
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);
    await user.click(await screen.findByRole("button", { name: "Fork latest checkpoint" }));
    await user.click(await screen.findByRole("button", { name: "Save draft" }));

    expect(createDraftWithSourceMock).toHaveBeenCalledWith(
      "ppo_test_1 fork",
      managedRunConfigFixture,
      "run-001",
      "latest",
    );

    await user.click(screen.getByRole("button", { name: "Train" }));

    expect(launchRunMock).toHaveBeenCalledWith(
      "ppo_test_1 fork",
      managedRunConfigFixture,
      "ppo_test_1 fork",
      "run-001",
      "latest",
    );
  });

  it("launches an unsaved fork with edited PPO clip range", async () => {
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
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);
    await user.click(await screen.findByRole("button", { name: "Fork latest checkpoint" }));
    await user.click(screen.getByRole("button", { name: "Training" }));

    const clipRangeInput = screen.getByRole("spinbutton", { name: "Clip range" });
    await user.clear(clipRangeInput);
    await user.type(clipRangeInput, "0.17");
    await user.click(screen.getByRole("button", { name: "Train" }));

    expect(launchRunMock).toHaveBeenCalledWith(
      "ppo_test_1 fork",
      expect.objectContaining({
        train: expect.objectContaining({
          clip_range: 0.17,
        }),
      }),
      null,
      "run-001",
      "latest",
    );
  });

  it("launches a fork with edited course pool", async () => {
    const user = userEvent.setup();
    const sourceConfig = {
      ...managedRunConfigFixture,
      tracks: {
        ...managedRunConfigFixture.tracks,
        selected_course_ids: [
          "mute_city",
          "silence",
          "sand_ocean",
          "devils_forest",
          "sector_alpha",
        ],
      },
    };
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [runFixture({ id: "run-001", name: "ppo_test_1", config: sourceConfig })],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);
    await user.click(await screen.findByRole("button", { name: "Fork latest checkpoint" }));
    await user.click(screen.getByRole("button", { name: "Big Blue" }));
    await user.click(screen.getByRole("button", { name: "Port Town" }));
    await user.click(screen.getByRole("button", { name: "Train" }));

    const launchedConfig = launchRunMock.mock.calls[0]?.[1] as typeof managedRunConfigFixture;
    expect(launchedConfig.tracks.selected_course_ids).toEqual([
      "mute_city",
      "silence",
      "sand_ocean",
      "devils_forest",
      "big_blue",
      "port_town",
      "sector_alpha",
    ]);
  });

  it("launches a saved draft with the edited PPO clip range", async () => {
    const user = userEvent.setup();
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Create draft" }));
    await user.click(screen.getByRole("button", { name: "Training" }));

    fireEvent.change(screen.getByRole("spinbutton", { name: "Clip range" }), {
      target: { value: "0.17" },
    });
    await user.click(screen.getByRole("button", { name: "Save draft" }));
    await waitFor(() => expect(createDraftWithSourceMock).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "Train" }));

    expect(launchRunMock).toHaveBeenCalledWith(
      "ppo_allcups_recurrent",
      expect.objectContaining({
        train: expect.objectContaining({
          clip_range: 0.17,
        }),
      }),
      "ppo_allcups_recurrent",
      null,
      null,
    );
  });

  it("creates a normal editable draft from a run without fork lineage", async () => {
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
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);
    await user.click(screen.getByRole("button", { name: "Create editable draft from run" }));

    expect(launchRunMock).not.toHaveBeenCalled();
    expect(createDraftWithSourceMock).toHaveBeenCalledWith(
      "ppo_test_1 draft",
      managedRunConfigFixture,
      null,
      null,
    );
    expect(screen.queryByText(/Forked from/i)).not.toBeInTheDocument();
    expect(await screen.findByRole("textbox", { name: "Run name" })).toHaveValue(
      "ppo_test_1 draft",
    );
    await user.click(within(workspaceTabs).getByRole("button", { name: "Drafts" }));
    expect(screen.getByRole("button", { name: /^ppo_test_1 draft/i })).toBeInTheDocument();
  });

  it("shows a generic restart message for backend schema mismatch", async () => {
    loadManagerDataMock.mockRejectedValueOnce(new ApiSchemaMismatchError());

    render(<App />);

    expect(
      await screen.findByText("Run-manager backend is outdated. Restart run-manager."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/\[\s*\{/)).not.toBeInTheDocument();
  });

  it("shows forked sim game time with lineage step offset included", async () => {
    const user = userEvent.setup();
    const runtime = runFixture().runtime;
    if (runtime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const parentRun = runFixture({
      id: "run-parent",
      status: "stopped",
      started_at: "2026-05-03T17:35:24+00:00",
      stopped_at: "2026-05-03T18:35:24+00:00",
      runtime: {
        ...runtime,
        num_timesteps: 180_000,
        updated_at: "2026-05-03T18:35:24+00:00",
        fps: null,
      },
    });
    const childRun = runFixture({
      id: "run-001",
      status: "stopped",
      lineage_step_offset: 180_000,
      parent_run_id: "run-parent",
      started_at: "2026-05-03T18:35:24+00:00",
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
        updated_at: "2026-05-03T19:05:24+00:00",
        fps: null,
      },
    });
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [parentRun, childRun],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);

    const wallLocalMetric = (await screen.findByText("Wall time · local")).closest(
      ".run-runtime-metric",
    );
    expect(wallLocalMetric?.textContent).toMatch(/29m 59s|30m 0s/);
    const wallTotalMetric = screen.getByText("Wall time · total").closest(".run-runtime-metric");
    expect(wallTotalMetric?.textContent).toMatch(/1h 29m 59s|1h 30m 0s/);
    const simLocalMetric = (await screen.findByText("Sim game time · local")).closest(
      ".run-runtime-metric",
    );
    expect(simLocalMetric?.textContent).toContain("33m 20s");
    const simTotalMetric = screen.getByText("Sim game time · total").closest(".run-runtime-metric");
    expect(simTotalMetric?.textContent).toContain("2h 13m 20s");
    const ratioLocalMetric = screen.getByText("Sim / wall · local").closest(".run-runtime-metric");
    expect(ratioLocalMetric?.textContent).toContain("1.11x");
    const ratioTotalMetric = screen.getByText("Sim / wall · total").closest(".run-runtime-metric");
    expect(ratioTotalMetric?.textContent).toContain("1.48x");
    expect(screen.getByText(/lineage steps ·/i).textContent).toContain(
      "240,000 lineage steps · 60,000 / 50,000,000 local fork steps",
    );
  });

  it("anchors watch launch failures to the watch control", async () => {
    const user = userEvent.setup();
    const run = runFixture({ id: "run-001", name: "ppo_test_1" });
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [run],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });
    watchRunMock.mockRejectedValueOnce(
      new Error(
        "Saved train config is not compatible with the current schema: /tmp/run. Restart the run with the current config schema.",
      ),
    );

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    const runOpenButtons = screen.getAllByRole("button", { name: "Open run ppo_test_1" });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);

    const watchButton = await screen.findByRole("button", { name: "Watch latest checkpoint" });
    await user.click(watchButton);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Saved train config is not compatible with the current schema");
    expect(watchButton.closest(".run-watch-control")).toContainElement(alert);
  });

  it("derives local wall time from active runtime instead of stale initial launch timestamps", async () => {
    const user = userEvent.setup();
    const baseRuntime = runFixture().runtime;
    if (baseRuntime === null) {
      throw new Error("run fixture must include runtime data");
    }
    const staleStartedRun = runFixture({
      id: "run-001",
      name: "ppo_test_1 - all tracks",
      status: "running",
      created_at: "2026-05-04T15:35:04+00:00",
      started_at: "2026-05-04T15:35:04+00:00",
      lineage_step_offset: 22_079_560,
      parent_run_id: "run-parent",
      config: {
        ...managedRunConfigFixture,
        action: {
          ...managedRunConfigFixture.action,
          action_repeat: 2,
        },
      },
      runtime: {
        ...baseRuntime,
        num_timesteps: 600,
        fps: 60,
        updated_at: "2026-05-04T17:00:10+00:00",
      },
    });
    const parentRun = runFixture({
      id: "run-parent",
      status: "stopped",
      runtime: null,
      started_at: "2026-05-03T10:00:00+00:00",
      stopped_at: "2026-05-03T11:00:00+00:00",
    });
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [parentRun, staleStartedRun],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    const runOpenButtons = screen.getAllByRole("button", {
      name: "Open run ppo_test_1 - all tracks",
    });
    const openRunButton = runOpenButtons.at(-1);
    if (openRunButton === undefined) {
      throw new Error("expected at least one open-run button");
    }
    await user.click(openRunButton);

    const wallLocalMetric = (await screen.findByText("Wall time · local")).closest(
      ".run-runtime-metric",
    );
    expect(wallLocalMetric?.textContent).toContain("10s");
    const ratioLocalMetric = screen.getByText("Sim / wall · local").closest(".run-runtime-metric");
    expect(ratioLocalMetric?.textContent).toContain("2.00x");
  });

  it("shows lineage step fallback for failed forks without runtime samples", async () => {
    const user = userEvent.setup();
    const failedForkRun = runFixture({
      id: "20260504-153504-ed51f7e7",
      name: "ppo_test_1 - all tracks",
      status: "failed",
      lineage_step_offset: 20_304_180,
      source_num_timesteps: 20_304_180,
      runtime: null,
    });
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [failedForkRun],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    await user.click(screen.getByRole("button", { name: "Open run ppo_test_1 - all tracks" }));

    const lineageStepsMetric = await screen.findByText("Lineage steps");
    expect(lineageStepsMetric.closest(".run-runtime-metric")?.textContent).toContain("20,304,180");
    expect(screen.getByText("20260504-153504-ed51f7e7")).toBeInTheDocument();
  });

  it("shows the latest failed event detail instead of the last startup step", async () => {
    const user = userEvent.setup();
    const failedRun = runFixture({
      id: "run-failed",
      name: "ppo_test_failed",
      status: "failed",
      runtime: null,
      recent_events: [
        {
          created_at: "2026-05-06T12:48:30+00:00",
          kind: "failed",
          message: "training failed: FileNotFoundError: missing fork source config",
        },
        {
          created_at: "2026-05-06T12:48:29+00:00",
          kind: "startup_resume",
          message: "Loading latest checkpoint",
        },
      ],
    });
    loadManagerDataMock.mockResolvedValueOnce({
      drafts: [],
      metadata: configMetadataFixture,
      runs: [failedRun],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });

    render(<App />);

    const workspaceTabs = await screen.findByRole("navigation", { name: "Run manager sections" });
    await user.click(within(workspaceTabs).getByRole("button", { name: "Runs" }));
    await user.click(screen.getByRole("button", { name: "Open run ppo_test_failed" }));

    expect(
      await screen.findByText("training failed: FileNotFoundError: missing fork source config"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Loading latest checkpoint")).not.toBeInTheDocument();
  });
});
