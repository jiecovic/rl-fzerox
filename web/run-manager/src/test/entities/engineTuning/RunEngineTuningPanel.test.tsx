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

    const table = screen.getByRole("table", { name: "Measured bandit bucket samples" });
    const rows = within(table).getAllByRole("row").slice(1);

    expect(within(rows[0] as HTMLElement).getByText("ENG 19.5")).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByText("ENG 70.3")).toBeInTheDocument();
  });

  it("sorts bandit bucket rows by selected metric", () => {
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
                  best_finish_time_ms: 94_000,
                  engine_setting_raw_value: 25,
                  estimated_finish_time_ms: 95_000,
                  finish_count: 4,
                  selection_probability: 0.1,
                }),
                engineTuningCandidateEstimateFixture({
                  best_finish_time_ms: 90_000,
                  engine_setting_raw_value: 55,
                  estimated_finish_time_ms: 97_000,
                  finish_count: 4,
                  selection_probability: 0.2,
                }),
                engineTuningCandidateEstimateFixture({
                  best_finish_time_ms: 93_000,
                  engine_setting_raw_value: 90,
                  estimated_finish_time_ms: 96_000,
                  finish_count: 4,
                  selection_probability: 0.7,
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

    const table = screen.getByRole("table", { name: "Measured bandit bucket samples" });

    fireEvent.click(within(table).getByRole("button", { name: /prob/i }));
    expect(firstBucketLabel(table)).toBe("ENG 70.3");

    fireEvent.click(within(table).getByRole("button", { name: /^mean time/i }));
    expect(firstBucketLabel(table)).toBe("ENG 19.5");

    fireEvent.click(within(table).getByRole("button", { name: /^best time/i }));
    expect(firstBucketLabel(table)).toBe("ENG 43.0");
  });

  it("shows collected return columns for finish-time bandit buckets", () => {
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
                  best_return_score: 21.75,
                  engine_setting_raw_value: 25,
                  mean_finish_time_ms: 95_000,
                  mean_return_score: 18.5,
                }),
              ],
              recommended_engine_setting_raw_value: 25,
            }),
          ],
          model_backend: "bandit",
          objective: "finish_time",
        })}
        onExpandedChange={() => undefined}
        onReset={() => undefined}
      />,
    );

    const table = screen.getByRole("table", { name: "Measured bandit bucket samples" });

    expect(within(table).getByRole("columnheader", { name: /mean return/i })).toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: /best return/i })).toBeInTheDocument();
    expect(within(table).getByText("18.5")).toBeInTheDocument();
    expect(within(table).getByText("21.8")).toBeInTheDocument();
  });

  it("marks the best bandit bucket value in each metric column", () => {
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
                  best_finish_time_ms: 94_000,
                  best_return_score: 15,
                  engine_setting_raw_value: 25,
                  finish_count: 1,
                  mean_finish_time_ms: 95_000,
                  mean_return_score: 10,
                  selection_probability: 0.1,
                }),
                engineTuningCandidateEstimateFixture({
                  best_finish_time_ms: 92_000,
                  best_return_score: 14,
                  engine_setting_raw_value: 55,
                  finish_count: 3,
                  mean_finish_time_ms: 90_000,
                  mean_return_score: 9,
                  selection_probability: 0.8,
                }),
                engineTuningCandidateEstimateFixture({
                  best_finish_time_ms: 88_000,
                  best_return_score: 20,
                  engine_setting_raw_value: 90,
                  finish_count: 2,
                  mean_finish_time_ms: 93_000,
                  mean_return_score: 12,
                  selection_probability: 0.2,
                }),
              ],
              recommended_engine_setting_raw_value: 55,
            }),
          ],
          model_backend: "bandit",
        })}
        onExpandedChange={() => undefined}
        onReset={() => undefined}
      />,
    );

    const table = screen.getByRole("table", { name: "Measured bandit bucket samples" });
    const row25 = rowForBucket(table, "ENG 19.5");
    const row55 = rowForBucket(table, "ENG 43.0");
    const row90 = rowForBucket(table, "ENG 70.3");

    expect(cellAt(row55, 1)).toHaveClass("font-semibold");
    expect(cellAt(row90, 2)).toHaveClass("font-semibold");
    expect(cellAt(row90, 3)).toHaveClass("font-semibold");
    expect(cellAt(row90, 4)).toHaveClass("font-semibold");
    expect(cellAt(row55, 5)).toHaveClass("font-semibold");
    expect(cellAt(row55, 6)).toHaveClass("font-semibold");
    expect(cellAt(row25, 1)).not.toHaveClass("font-semibold");
  });
});

function hasExactText(expected: string) {
  return (_content: string, element: Element | null) => element?.textContent === expected;
}

function firstBucketLabel(table: HTMLElement) {
  const firstRow = within(table).getAllByRole("row")[1];
  if (!(firstRow instanceof HTMLElement)) {
    throw new Error("expected one bandit bucket row");
  }
  return firstRow.querySelector("td")?.textContent ?? "";
}

function rowForBucket(table: HTMLElement, bucketLabel: string) {
  const rows = within(table).getAllByRole("row").slice(1);
  const row = rows.find((candidateRow) => {
    if (!(candidateRow instanceof HTMLElement)) {
      return false;
    }
    return within(candidateRow).queryByText(bucketLabel) !== null;
  });
  if (!(row instanceof HTMLElement)) {
    throw new Error(`expected bandit bucket row ${bucketLabel}`);
  }
  return row;
}

function cellAt(row: HTMLElement, index: number) {
  const cell = within(row).getAllByRole("cell")[index];
  if (!(cell instanceof HTMLElement)) {
    throw new Error(`expected cell ${index}`);
  }
  return cell;
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
    objective: "finish_time",
    reward_fingerprint: null,
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
    best_completion_score: 1.0,
    best_return_score: 2.25,
    context_key: "big_blue_2|blue_falcon",
    course_key: "big_blue_2",
    engine_setting_raw_value: 55,
    episode_count: 1,
    failure_rate: 0.0,
    finish_count: 1,
    finish_rate: 1.0,
    mean_completion_score: 1.0,
    mean_finish_score: 2.25,
    mean_return_score: null,
    mean_score: 2.25,
    raw_mean_score: 2.25,
    return_count: 0,
    score_count: 1,
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
    episode_count: 1,
    finish_count: 1,
    model_ready: true,
    observed_candidate_count: 1,
    recommended_engine_setting_raw_value: 55,
    score_count: 1,
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
    best_completion_score: 1.0,
    best_return_score: 2.25,
    best_score: 2.25,
    episode_count: 1,
    estimated_finish_time_ms: 95_000,
    failure_rate: 0.0,
    finish_count: 1,
    finish_rate: 1.0,
    mean_completion_score: 1.0,
    mean_finish_time_ms: 95_000,
    mean_return_score: 2.25,
    mean_score: 2.25,
    score_count: 1,
    return_count: 1,
    selection_probability: 0.5,
    uncertainty_score: 0.25,
    ...overrides,
  };
}
