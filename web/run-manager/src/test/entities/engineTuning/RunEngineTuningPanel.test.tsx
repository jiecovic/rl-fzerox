// web/run-manager/src/test/entities/engineTuning/RunEngineTuningPanel.test.tsx
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunEngineTuningPanel } from "@/entities/engineTuning/ui/RunEngineTuningPanel";
import type {
  EngineTuningRuntimeCandidate,
  EngineTuningRuntimeCandidateEstimate,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";
import { configMetadataFixture } from "@/test/fixtures";
import { cleanup, fireEvent, render, screen, within } from "@/test/render";

describe("RunEngineTuningPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders readable context labels and engine probabilities", () => {
    render(
      <RunEngineTuningPanel
        artifact="latest"
        canReset={true}
        enabled={true}
        expanded={true}
        isResetting={false}
        metadata={configMetadataFixture}
        state={engineTuningStateFixture()}
        onExpandedChange={() => undefined}
        onReset={() => undefined}
      />,
    );

    expect(screen.getByText("Big Blue 2 · Blue Falcon")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /X Cup · Blue Falcon/ })).toBeInTheDocument();
    expect(screen.getByText(hasExactText("y: probability 0-42.0%"))).toBeInTheDocument();
    expect(screen.getByText("deterministic greedy")).toBeInTheDocument();
    expect(screen.queryByText("big_blue_2 · blue_falcon")).not.toBeInTheDocument();
  });

  it("hides chart details while collapsed", () => {
    const onExpandedChange = vi.fn();

    render(
      <RunEngineTuningPanel
        artifact="latest"
        canReset={true}
        enabled={true}
        expanded={false}
        isResetting={false}
        metadata={configMetadataFixture}
        state={engineTuningStateFixture()}
        onExpandedChange={onExpandedChange}
        onReset={() => undefined}
      />,
    );

    expect(screen.getByText("Engine tuning")).toBeInTheDocument();
    expect(screen.queryByText("Big Blue 2 · Blue Falcon")).not.toBeInTheDocument();
    const summary = screen.getByText("Engine tuning").closest("summary");
    if (!(summary instanceof HTMLElement)) {
      throw new Error("engine tuning summary not found");
    }
    const details = summary.closest("details");
    if (!(details instanceof HTMLDetailsElement)) {
      throw new Error("engine tuning details not found");
    }
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(onExpandedChange).toHaveBeenCalledWith(true);
  });

  it("confirms before resetting tuner sidecars", () => {
    const onReset = vi.fn();

    render(
      <RunEngineTuningPanel
        artifact="latest"
        canReset={true}
        enabled={true}
        expanded={true}
        isResetting={false}
        metadata={configMetadataFixture}
        state={engineTuningStateFixture()}
        onExpandedChange={() => undefined}
        onReset={onReset}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Reset tuner" }));

    expect(screen.getByRole("dialog", { name: "Reset engine tuner" })).toBeInTheDocument();
    expect(onReset).not.toHaveBeenCalled();

    fireEvent.click(
      within(screen.getByRole("dialog", { name: "Reset engine tuner" })).getByRole("button", {
        name: "Reset tuner",
      }),
    );

    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("orders equal-mean bandit buckets by best finish time before finish count", () => {
    render(
      <RunEngineTuningPanel
        artifact="latest"
        canReset={true}
        enabled={true}
        expanded={true}
        isResetting={false}
        metadata={configMetadataFixture}
        state={engineTuningStateFixture({
          contexts: [
            engineTuningContextFixture({
              candidates: [
                engineTuningCandidateEstimateFixture({
                  best_finish_time_ms: 90_000,
                  engine_setting_raw_value: 25,
                  estimated_finish_time_ms: 95_000,
                  finish_count: 2,
                }),
                engineTuningCandidateEstimateFixture({
                  best_finish_time_ms: 94_000,
                  engine_setting_raw_value: 90,
                  estimated_finish_time_ms: 95_000,
                  finish_count: 3,
                }),
              ],
              recommended_engine_setting_raw_value: 25,
            }),
          ],
          model_backend: "bandit",
        })}
        onExpandedChange={() => undefined}
        onReset={() => undefined}
      />,
    );

    const table = screen.getByRole("table", { name: "Measured bandit bucket finishes" });
    const rows = within(table).getAllByRole("row").slice(1);

    expect(within(rows[0] as HTMLElement).getByText("ENG 19.5")).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByText("ENG 70.3")).toBeInTheDocument();
  });
});

function hasExactText(expected: string) {
  return (_content: string, element: Element | null) => element?.textContent === expected;
}

function engineTuningStateFixture(
  overrides: Partial<EngineTuningRuntimeState> = {},
): EngineTuningRuntimeState {
  return {
    candidates: [
      engineTuningCandidateFixture({ course_key: "big_blue_2", vehicle_id: "blue_falcon" }),
      engineTuningCandidateFixture({
        context_key: "x_cup|blue_falcon",
        course_key: "x_cup",
        engine_setting_raw_value: 10,
        vehicle_id: "blue_falcon",
      }),
    ],
    contexts: [
      engineTuningContextFixture({
        candidates: [
          engineTuningCandidateEstimateFixture({
            engine_setting_raw_value: 55,
            selection_probability: 0.42,
          }),
          engineTuningCandidateEstimateFixture({
            engine_setting_raw_value: 60,
            selection_probability: 0.25,
          }),
        ],
        course_key: "big_blue_2",
      }),
      engineTuningContextFixture({
        candidates: [engineTuningCandidateEstimateFixture({ engine_setting_raw_value: 10 })],
        context_key: "x_cup|blue_falcon",
        course_key: "x_cup",
        recommended_engine_setting_raw_value: 10,
      }),
    ],
    model_backend: "gaussian_process",
    update_count: 10,
    version: 1,
    ...overrides,
  };
}

function engineTuningCandidateFixture(
  overrides: Partial<EngineTuningRuntimeCandidate> = {},
): EngineTuningRuntimeCandidate {
  return {
    best_score: 2.25,
    best_finish_time_ms: 92_000,
    context_key: "big_blue_2|blue_falcon",
    course_key: "big_blue_2",
    engine_setting_raw_value: 55,
    finish_count: 1,
    mean_score: 2.25,
    raw_mean_score: 2.25,
    vehicle_id: "blue_falcon",
    ...overrides,
  };
}

function engineTuningContextFixture(
  overrides: Partial<EngineTuningRuntimeContext> = {},
): EngineTuningRuntimeContext {
  return {
    candidates: [engineTuningCandidateEstimateFixture()],
    context_key: "big_blue_2|blue_falcon",
    course_key: "big_blue_2",
    finish_count: 1,
    model_ready: true,
    observed_candidate_count: 1,
    recommended_engine_setting_raw_value: 55,
    warmup_remaining: 0,
    warmup_successes: 0,
    vehicle_id: "blue_falcon",
    ...overrides,
  };
}

function engineTuningCandidateEstimateFixture(
  overrides: Partial<EngineTuningRuntimeCandidateEstimate> = {},
): EngineTuningRuntimeCandidateEstimate {
  return {
    engine_setting_raw_value: 55,
    best_finish_time_ms: 92_000,
    estimated_finish_time_ms: 95_000,
    finish_count: 1,
    mean_score: 2.25,
    selection_probability: 0.5,
    uncertainty_score: 0.25,
    ...overrides,
  };
}
