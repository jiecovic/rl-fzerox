// web/run-manager/src/test/widgets/runWorkspace/RunTrackPoolPanel.test.tsx

import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunTrackPoolPanel } from "@/entities/trackPool/ui/RunTrackPoolPanel";
import type { TrackSamplingRuntimeState } from "@/shared/api/contract";
import { configMetadataFixture, managedRunConfigFixture, runFixture } from "@/test/fixtures";
import { cleanup, render, screen, within } from "@/test/render";

function enabledResetButton() {
  const button = screen
    .getAllByRole("button", { name: "Reset stats" })
    .find((candidate) => !candidate.hasAttribute("disabled"));
  if (button === undefined) {
    throw new Error("expected one enabled Reset stats button");
  }
  return button;
}

const emptyGenerationStats = {
  generation_episode_count: 0,
  generation_finished_episode_count: 0,
  generation_success_sample_count: 0,
  generation_success_rate: null,
  generation_ema_completion_fraction: null,
  generated_course_slot: null,
  generated_course_generation: null,
} as const;

const defaultDeficitBudgetState = {
  deficit_budget_difficulty_metric: "completion_ema",
  deficit_budget_warmup_min_episodes_per_course: 10,
} as const;

const emptySamplerSignal = {
  ema_finish_rate: null,
  current_problem_score: 0,
} as const;

const emptyCompletionStats = {
  completion_sample_count: 0,
  completion_fraction_total: 0,
  completion_rate: null,
} as const;

describe("RunTrackPoolPanel", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("resets the selected cup when the opened run changes", async () => {
    const user = userEvent.setup();
    const firstCup = configMetadataFixture.track_cups[0];
    const secondCup = configMetadataFixture.track_cups[1];
    if (
      firstCup?.course_ids[0] === undefined ||
      firstCup.course_ids[1] === undefined ||
      secondCup?.course_ids[0] === undefined
    ) {
      throw new Error(
        "fixture must provide at least two first-cup courses and one second-cup course",
      );
    }
    const initialCourseIds = [firstCup.course_ids[0], secondCup.course_ids[0]];
    const expandedCourseIds = [
      firstCup.course_ids[0],
      firstCup.course_ids[1],
      secondCup.course_ids[0],
    ];
    const state = trackSamplingStateForCourses(expandedCourseIds);

    const { rerender } = render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          id: "run-before",
          config: runConfigWithSelectedCourses(initialCourseIds),
        })}
        state={state}
      />,
    );

    await user.click(screen.getByRole("tab", { name: /queen/i }));
    expect(screen.getByText(courseDisplayName(secondCup.course_ids[0]))).toBeInTheDocument();

    rerender(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          id: "run-after",
          config: runConfigWithSelectedCourses(expandedCourseIds),
        })}
        state={state}
      />,
    );

    expect(screen.getByText(courseDisplayName(firstCup.course_ids[1]))).toBeInTheDocument();
    expect(screen.queryByText(courseDisplayName(secondCup.course_ids[0]))).not.toBeInTheDocument();
  });

  it("groups the pool into cup tabs and switches the visible course rows", async () => {
    const user = userEvent.setup();
    const firstCup = configMetadataFixture.track_cups[0];
    const secondCup = configMetadataFixture.track_cups[1];
    if (firstCup === undefined || secondCup === undefined) {
      throw new Error("fixture must provide at least two cups");
    }
    const selectedCourseIds = [
      firstCup.course_ids[0],
      firstCup.course_ids[1],
      secondCup.course_ids[0],
      secondCup.course_ids[1],
    ];
    if (selectedCourseIds.some((courseId) => courseId === undefined)) {
      throw new Error("fixture cups must provide at least two courses each");
    }
    const selectedCourses = configMetadataFixture.built_in_courses.filter((course) =>
      selectedCourseIds.includes(course.id),
    );
    const state: TrackSamplingRuntimeState = {
      sampling_mode: "step_balanced",
      action_repeat: 2,
      update_episodes: 50,
      ema_alpha: 0.1,
      max_weight_scale: 5,
      adaptive_completion_weight: 0.35,
      adaptive_target_completion: 0.9,
      adaptive_min_confidence_episodes: 24,
      adaptive_confidence_scale: 4,
      ...defaultDeficitBudgetState,
      update_count: 3,
      episodes_since_update: 17,
      entries: selectedCourses.map((course, index) => ({
        track_id: `${course.id}-track`,
        course_key: course.id,
        label: course.display_name,
        current_weight: 1,
        current_probability: 0.25,
        episode_count: index + 1,
        finished_episode_count: index,
        success_sample_count: index + 1,
        ...emptyCompletionStats,
        episode_share: 0.25,
        success_rate: index / Math.max(index + 1, 1),
        ...emptyGenerationStats,
        ...emptySamplerSignal,
        target_step_share: 0.25,
        completed_frames: (index + 1) * 600,
        completed_env_steps: (index + 1) * 300,
        measurement_env_steps: (index + 1) * 300,
        step_share: 0.25,
        ema_episode_frames: (index + 1) * 600,
        ema_completion_fraction: Math.min(0.95, 0.3 + index * 0.1),
      })),
    };

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              sampling_mode: "step_balanced",
              selected_course_ids: selectedCourseIds as string[],
            },
          },
        })}
        state={state}
      />,
    );

    expect(screen.getByRole("tab", { name: /jack/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /queen/i })).toBeInTheDocument();
    expect(screen.getByText(selectedCourses[0]?.display_name ?? "")).toBeInTheDocument();
    expect(screen.getByText("Completion")).toBeInTheDocument();
    expect(screen.getByText(/finish 33.3%/i)).toBeInTheDocument();
    expect(screen.queryByText(selectedCourses[2]?.display_name ?? "")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /queen/i }));

    expect(screen.getByText(selectedCourses[2]?.display_name ?? "")).toBeInTheDocument();
    expect(screen.queryByText(selectedCourses[0]?.display_name ?? "")).not.toBeInTheDocument();
  });

  it("supports keyboard navigation across cup tabs", async () => {
    const user = userEvent.setup();
    const firstCup = configMetadataFixture.track_cups[0];
    const secondCup = configMetadataFixture.track_cups[1];
    if (firstCup === undefined || secondCup === undefined) {
      throw new Error("fixture must provide at least two cups");
    }
    const selectedCourseIds = [
      firstCup.course_ids[0],
      firstCup.course_ids[1],
      secondCup.course_ids[0],
      secondCup.course_ids[1],
    ];
    if (selectedCourseIds.some((courseId) => courseId === undefined)) {
      throw new Error("fixture cups must provide at least two courses each");
    }
    const selectedCourses = configMetadataFixture.built_in_courses.filter((course) =>
      selectedCourseIds.includes(course.id),
    );

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: runConfigWithSelectedCourses(selectedCourseIds as string[]),
        })}
        state={trackSamplingStateForCourses(selectedCourseIds as string[])}
      />,
    );

    screen.getByRole("tab", { name: /jack/i }).focus();
    await user.keyboard("{ArrowRight}");

    expect(screen.getByRole("tab", { name: /queen/i })).toHaveFocus();
    expect(screen.getByText(selectedCourses[2]?.display_name ?? "")).toBeInTheDocument();

    await user.keyboard("{Home}");

    expect(screen.getByRole("tab", { name: /jack/i })).toHaveFocus();
    expect(screen.getByText(selectedCourses[0]?.display_name ?? "")).toBeInTheDocument();
  });

  it("requires confirmation before resetting track-pool stats", async () => {
    const user = userEvent.setup();
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];
    const selectedCourses = configMetadataFixture.built_in_courses.filter((course) =>
      selectedCourseIds.includes(course.id),
    );
    const state: TrackSamplingRuntimeState = {
      sampling_mode: "step_balanced",
      action_repeat: 2,
      update_episodes: 50,
      ema_alpha: 0.1,
      max_weight_scale: 5,
      adaptive_completion_weight: 0.35,
      adaptive_target_completion: 0.9,
      adaptive_min_confidence_episodes: 24,
      adaptive_confidence_scale: 4,
      ...defaultDeficitBudgetState,
      update_count: 1,
      episodes_since_update: 0,
      entries: selectedCourses.map((course) => ({
        track_id: course.id,
        course_key: course.id,
        label: course.display_name,
        current_weight: 1,
        current_probability: 0.5,
        episode_count: 1,
        finished_episode_count: 1,
        success_sample_count: 1,
        ...emptyCompletionStats,
        episode_share: 0.5,
        success_rate: 1,
        ...emptyGenerationStats,
        ...emptySamplerSignal,
        target_step_share: 0.5,
        completed_frames: 600,
        completed_env_steps: 300,
        measurement_env_steps: 300,
        step_share: 0.5,
        ema_episode_frames: 600,
        ema_completion_fraction: 1,
      })),
    };
    const onReset = vi.fn();

    render(
      <RunTrackPoolPanel
        canReset
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={onReset}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              sampling_mode: "step_balanced",
              selected_course_ids: selectedCourseIds as string[],
            },
          },
        })}
        state={state}
      />,
    );

    await user.click(enabledResetButton());

    expect(screen.getByRole("dialog", { name: "Reset track-pool stats" })).toBeInTheDocument();
    expect(onReset).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(
      screen.queryByRole("dialog", { name: "Reset track-pool stats" }),
    ).not.toBeInTheDocument();
    expect(onReset).not.toHaveBeenCalled();

    await user.click(enabledResetButton());
    await user.click(
      within(screen.getByRole("dialog", { name: "Reset track-pool stats" })).getByRole("button", {
        name: "Reset stats",
      }),
    );

    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("requires confirmation before clearing alt baselines", async () => {
    const user = userEvent.setup();
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];
    const onClearAltBaselines = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ baselines: [] }), {
          headers: { "Content-Type": "application/json" },
          status: 200,
        }),
      ),
    );

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={onClearAltBaselines}
        onReset={() => undefined}
        run={runFixture({
          active_alt_baseline_count: 2,
          config: runConfigWithSelectedCourses(selectedCourseIds),
        })}
        state={trackSamplingStateForCourses(selectedCourseIds)}
      />,
    );

    expect(screen.getByText(/2 alt baselines/i)).toBeInTheDocument();
    await user.click(screen.getByText("Alt baselines (2)"));
    await user.click(screen.getByRole("button", { name: "Clear all" }));

    expect(screen.getByRole("dialog", { name: "Clear alt baselines" })).toBeInTheDocument();
    expect(onClearAltBaselines).not.toHaveBeenCalled();

    await user.click(
      within(screen.getByRole("dialog", { name: "Clear alt baselines" })).getByRole("button", {
        name: "Clear alt baselines",
      }),
    );

    expect(onClearAltBaselines).toHaveBeenCalledTimes(1);
  });

  it("renders compact alt-baseline course rows with difficulty labels", async () => {
    const user = userEvent.setup();
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            baselines: [
              {
                id: "alt-001",
                course_key: firstCup.course_ids[0],
                reset_variant_key: "mode=gp_race|gp_difficulty=novice|vehicle=blue_falcon",
                source_entry_id: `${firstCup.course_ids[0]}_gp_race_novice_blue_falcon`,
                label: "frame 1200",
                state_path: "/tmp/alt-001.state",
                weight: 1,
                created_at: "2026-06-13T17:00:00+00:00",
                updated_at: "2026-06-13T17:00:00+00:00",
              },
              {
                id: "alt-002",
                course_key: firstCup.course_ids[1],
                reset_variant_key: "mode=gp_race|gp_difficulty=master|vehicle=blue_falcon",
                source_entry_id: `${firstCup.course_ids[1]}_gp_race_master_blue_falcon`,
                label: "frame 2400",
                state_path: "/tmp/alt-002.state",
                weight: 1,
                created_at: "2026-06-13T17:05:00+00:00",
                updated_at: "2026-06-13T17:05:00+00:00",
              },
            ],
          }),
          {
            headers: { "Content-Type": "application/json" },
            status: 200,
          },
        ),
      ),
    );

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          active_alt_baseline_count: 2,
          config: runConfigWithSelectedCourses(selectedCourseIds),
        })}
        state={trackSamplingStateForCourses(selectedCourseIds)}
      />,
    );

    await user.click(screen.getByText("Alt baselines (2)"));

    expect(screen.queryByRole("button", { name: /refresh/i })).not.toBeInTheDocument();
    expect(
      await screen.findByText("Novice · frame 1200 · 2026-06-13 17:00:00 UTC"),
    ).toBeInTheDocument();
    expect(screen.getByText("Master · frame 2400 · 2026-06-13 17:05:00 UTC")).toBeInTheDocument();
  });

  it("renders the adaptive step-balanced mode label", () => {
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              sampling_mode: "adaptive_step_balanced",
              selected_course_ids: selectedCourseIds as string[],
            },
          },
        })}
        state={{
          ...trackSamplingStateForCourses(selectedCourseIds),
          sampling_mode: "adaptive_step_balanced",
        }}
      />,
    );

    expect(screen.getByText(/adaptive step-balanced/i)).toBeInTheDocument();
  });

  it("shows generated X Cup courses from runtime sampling state", () => {
    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              race_mode: "gp_race",
              gp_difficulties: ["novice"],
              include_x_cup: true,
              x_cup_course_count: 2,
              sampling_mode: "step_balanced",
              selected_course_ids: [],
            },
          },
        })}
        state={xCupTrackSamplingState()}
      />,
    );

    expect(screen.getByText("X Cup")).toBeInTheDocument();
    expect(screen.getByText("X Cup abcd1234")).toBeInTheDocument();
    expect(screen.getByText("X Cup ef567890")).toBeInTheDocument();
    expect(screen.getByText(/slot 1 · generated 1 · sampling:/i)).toBeInTheDocument();
    expect(screen.getByText(/slot 2 · generated 2 · sampling:/i)).toBeInTheDocument();
  });

  it("shows the target env-step share in the env-steps bar tooltip", () => {
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              sampling_mode: "adaptive_step_balanced",
              selected_course_ids: selectedCourseIds as string[],
            },
          },
        })}
        state={{
          ...trackSamplingStateForCourses(selectedCourseIds),
          sampling_mode: "adaptive_step_balanced",
          entries: trackSamplingStateForCourses(selectedCourseIds).entries.map((entry, index) => ({
            ...entry,
            step_share: index === 0 ? 0.4 : 0.6,
            target_step_share: index === 0 ? 0.7 : 0.3,
          })),
        }}
      />,
    );

    expect(screen.getByText(/step target/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /target 70\.0%/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /target 30\.0%/i })).toBeInTheDocument();
  });

  it("hides step targets for fixed env assignment", () => {
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              sampling_mode: "fixed_env",
              selected_course_ids: selectedCourseIds as string[],
            },
          },
        })}
        state={{
          ...trackSamplingStateForCourses(selectedCourseIds),
          sampling_mode: "fixed_env",
        }}
      />,
    );

    expect(screen.queryByText(/step target/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /target/i })).not.toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /completed episode env steps/i }).length,
    ).toBeGreaterThan(0);
  });

  it("shows track-pool stats for deficit budget sampling", () => {
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];

    render(
      <RunTrackPoolPanel
        canReset={false}
        isClearingAltBaselines={false}
        isResetting={false}
        metadata={configMetadataFixture}
        onClearAltBaselines={() => undefined}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              sampling_mode: "deficit_budget",
              selected_course_ids: selectedCourseIds as string[],
            },
          },
        })}
        state={{
          ...trackSamplingStateForCourses(selectedCourseIds),
          sampling_mode: "deficit_budget",
          entries: trackSamplingStateForCourses(selectedCourseIds).entries.map((entry, index) => ({
            ...entry,
            current_probability: index === 0 ? 0 : 1,
            target_step_share: 0.5,
          })),
        }}
      />,
    );

    expect(screen.getByText("Track pool")).toBeInTheDocument();
    expect(screen.getByText(/deficit budget/i)).toBeInTheDocument();
    expect(screen.getByText(/Target step share/i)).toBeInTheDocument();
    expect(screen.getByText("Completion")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sample 0.0%" })).not.toBeInTheDocument();
    expect(screen.getByText(/step target/i)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /env steps .* target/i }).length).toBeGreaterThan(
      0,
    );
  });
});

function runConfigWithSelectedCourses(courseIds: readonly string[]) {
  return {
    ...managedRunConfigFixture,
    tracks: {
      ...managedRunConfigFixture.tracks,
      sampling_mode: "step_balanced" as const,
      selected_course_ids: [...courseIds],
    },
  };
}

function trackSamplingStateForCourses(courseIds: readonly string[]): TrackSamplingRuntimeState {
  const selectedCourses = configMetadataFixture.built_in_courses.filter((course) =>
    courseIds.includes(course.id),
  );
  const probability = selectedCourses.length === 0 ? 0 : 1 / selectedCourses.length;
  return {
    sampling_mode: "step_balanced",
    action_repeat: 2,
    update_episodes: 50,
    ema_alpha: 0.1,
    max_weight_scale: 5,
    adaptive_completion_weight: 0.35,
    adaptive_target_completion: 0.9,
    adaptive_min_confidence_episodes: 24,
    adaptive_confidence_scale: 4,
    ...defaultDeficitBudgetState,
    update_count: 1,
    episodes_since_update: 0,
    entries: selectedCourses.map((course) => ({
      track_id: course.id,
      course_key: course.id,
      label: course.display_name,
      current_weight: 1,
      current_probability: probability,
      episode_count: 1,
      finished_episode_count: 0,
      success_sample_count: 1,
      ...emptyCompletionStats,
      episode_share: probability,
      success_rate: 0,
      ...emptyGenerationStats,
      ...emptySamplerSignal,
      target_step_share: probability,
      completed_frames: 600,
      completed_env_steps: 300,
      measurement_env_steps: 300,
      step_share: probability,
      ema_episode_frames: 600,
      ema_completion_fraction: 0,
    })),
  };
}

function xCupTrackSamplingState(): TrackSamplingRuntimeState {
  return {
    sampling_mode: "step_balanced",
    action_repeat: 2,
    update_episodes: 50,
    ema_alpha: 0.1,
    max_weight_scale: 5,
    adaptive_completion_weight: 0.35,
    adaptive_target_completion: 0.9,
    adaptive_min_confidence_episodes: 24,
    adaptive_confidence_scale: 4,
    ...defaultDeficitBudgetState,
    update_count: 1,
    episodes_since_update: 0,
    entries: ["abcd1234", "ef567890"].map((hash, index) => ({
      track_id: `x_cup_${hash}`,
      course_key: `x_cup_${hash}`,
      label: `X Cup ${hash}`,
      current_weight: 1,
      current_probability: 0.5,
      episode_count: index + 1,
      finished_episode_count: index,
      success_sample_count: index + 1,
      ...emptyCompletionStats,
      episode_share: 0.5,
      success_rate: index / Math.max(index + 1, 1),
      ...emptyGenerationStats,
      ...emptySamplerSignal,
      generated_course_slot: index,
      generated_course_generation: index + 1,
      target_step_share: 0.5,
      completed_frames: (index + 1) * 600,
      completed_env_steps: (index + 1) * 300,
      measurement_env_steps: (index + 1) * 300,
      step_share: 0.5,
      ema_episode_frames: (index + 1) * 600,
      ema_completion_fraction: Math.min(0.95, 0.3 + index * 0.1),
    })),
  };
}

function courseDisplayName(courseId: string): string {
  const course = configMetadataFixture.built_in_courses.find(
    (candidate) => candidate.id === courseId,
  );
  if (course === undefined) {
    throw new Error(`unknown fixture course: ${courseId}`);
  }
  return course.display_name;
}
