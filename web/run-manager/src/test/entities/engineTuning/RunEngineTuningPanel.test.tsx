// web/run-manager/src/test/entities/engineTuning/RunEngineTuningPanel.test.tsx
import { afterEach, describe, expect, it } from "vitest";
import { RunEngineTuningPanel } from "@/entities/engineTuning/ui/RunEngineTuningPanel";
import type {
  EngineTuningRuntimeArm,
  EngineTuningRuntimeBin,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";
import { configMetadataFixture } from "@/test/fixtures";
import { cleanup, render, screen } from "@/test/render";

describe("RunEngineTuningPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders readable context labels and engine probabilities", () => {
    render(
      <RunEngineTuningPanel
        artifact="latest"
        enabled={true}
        metadata={configMetadataFixture}
        state={engineTuningStateFixture()}
      />,
    );

    expect(screen.getByText("Big Blue 2 · Blue Falcon")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /X Cup · Blue Falcon/ })).toBeInTheDocument();
    expect(screen.getByText("42.0%")).toBeInTheDocument();
    expect(screen.queryByText("big_blue_2 · blue_falcon")).not.toBeInTheDocument();
  });

  it("reports stale API payloads that still contain only raw arms", () => {
    render(
      <RunEngineTuningPanel
        artifact="latest"
        enabled={true}
        metadata={configMetadataFixture}
        state={{
          arms: [engineTuningArmFixture()],
          contexts: [],
          update_count: 10,
          version: 1,
        }}
      />,
    );

    expect(screen.getByText(/samples exist/i)).toBeInTheDocument();
    expect(screen.getByText(/no distribution projection/i)).toBeInTheDocument();
  });
});

function engineTuningStateFixture(): EngineTuningRuntimeState {
  return {
    arms: [
      engineTuningArmFixture({ course_key: "big_blue_2", vehicle_id: "blue_falcon" }),
      engineTuningArmFixture({
        context_key: "x_cup|blue_falcon",
        course_key: "x_cup",
        engine_setting_raw_value: 10,
        vehicle_id: "blue_falcon",
      }),
    ],
    contexts: [
      engineTuningContextFixture({
        bins: [
          engineTuningBinFixture({ engine_setting_raw_value: 55, selection_probability: 0.42 }),
          engineTuningBinFixture({ engine_setting_raw_value: 60, selection_probability: 0.25 }),
        ],
        course_key: "big_blue_2",
      }),
      engineTuningContextFixture({
        bins: [engineTuningBinFixture({ engine_setting_raw_value: 10 })],
        context_key: "x_cup|blue_falcon",
        course_key: "x_cup",
        recommended_engine_setting_raw_value: 10,
      }),
    ],
    update_count: 10,
    version: 1,
  };
}

function engineTuningArmFixture(
  overrides: Partial<EngineTuningRuntimeArm> = {},
): EngineTuningRuntimeArm {
  return {
    attempts: 1,
    best_score: 2.25,
    context_key: "big_blue_2|blue_falcon",
    course_key: "big_blue_2",
    engine_setting_raw_value: 55,
    finished_attempts: 1,
    finish_rate: 1,
    mean_completion: 1,
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
    attempts: 1,
    bins: [engineTuningBinFixture()],
    context_key: "big_blue_2|blue_falcon",
    course_key: "big_blue_2",
    observed_arm_count: 1,
    recommended_engine_setting_raw_value: 55,
    vehicle_id: "blue_falcon",
    ...overrides,
  };
}

function engineTuningBinFixture(
  overrides: Partial<EngineTuningRuntimeBin> = {},
): EngineTuningRuntimeBin {
  return {
    attempts: 1,
    engine_setting_raw_value: 55,
    finish_rate: 1,
    mean_completion: 1,
    posterior_mean: 2.25,
    selection_probability: 0.5,
    ...overrides,
  };
}
