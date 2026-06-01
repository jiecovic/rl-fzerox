// src/rl_fzerox/apps/run_manager/web/src/test/features/runs/RunTrackPoolPanel.test.tsx

import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunTrackPoolPanel } from "@/features/runs/RunTrackPoolPanel";
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

describe("RunTrackPoolPanel", () => {
  afterEach(() => {
    cleanup();
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
        isResetting={false}
        metadata={configMetadataFixture}
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
        isResetting={false}
        metadata={configMetadataFixture}
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
        episode_share: 0.25,
        success_rate: index / Math.max(index + 1, 1),
        ...emptyGenerationStats,
        target_step_share: 0.25,
        completed_frames: (index + 1) * 600,
        completed_env_steps: (index + 1) * 300,
        step_share: 0.25,
        ema_episode_frames: (index + 1) * 600,
        ema_completion_fraction: Math.min(0.95, 0.3 + index * 0.1),
      })),
    };

    render(
      <RunTrackPoolPanel
        canReset={false}
        isResetting={false}
        metadata={configMetadataFixture}
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
        isResetting={false}
        metadata={configMetadataFixture}
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
        episode_share: 0.5,
        success_rate: 1,
        ...emptyGenerationStats,
        target_step_share: 0.5,
        completed_frames: 600,
        completed_env_steps: 300,
        step_share: 0.5,
        ema_episode_frames: 600,
        ema_completion_fraction: 1,
      })),
    };
    const onReset = vi.fn();

    render(
      <RunTrackPoolPanel
        canReset
        isResetting={false}
        metadata={configMetadataFixture}
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

  it("renders the adaptive step-balanced mode label", () => {
    const firstCup = configMetadataFixture.track_cups[0];
    if (firstCup?.course_ids[0] === undefined || firstCup.course_ids[1] === undefined) {
      throw new Error("fixture cup must provide at least two courses");
    }
    const selectedCourseIds = [firstCup.course_ids[0], firstCup.course_ids[1]];

    render(
      <RunTrackPoolPanel
        canReset={false}
        isResetting={false}
        metadata={configMetadataFixture}
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
        isResetting={false}
        metadata={configMetadataFixture}
        onReset={() => undefined}
        run={runFixture({
          config: {
            ...managedRunConfigFixture,
            tracks: {
              ...managedRunConfigFixture.tracks,
              race_mode: "gp_race",
              gp_difficulty: "novice",
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
        isResetting={false}
        metadata={configMetadataFixture}
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
      episode_share: probability,
      success_rate: 0,
      ...emptyGenerationStats,
      target_step_share: probability,
      completed_frames: 600,
      completed_env_steps: 300,
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
    update_count: 1,
    episodes_since_update: 0,
    entries: ["abcd1234", "ef567890"].map((hash, index) => ({
      track_id: `x_cup_${hash}_gp_race_novice_blue_falcon_balanced`,
      course_key: `x_cup_${hash}`,
      label: `X Cup ${hash}`,
      current_weight: 1,
      current_probability: 0.5,
      episode_count: index + 1,
      finished_episode_count: index,
      success_sample_count: index + 1,
      episode_share: 0.5,
      success_rate: index / Math.max(index + 1, 1),
      ...emptyGenerationStats,
      generated_course_slot: index,
      generated_course_generation: index + 1,
      target_step_share: 0.5,
      completed_frames: (index + 1) * 600,
      completed_env_steps: (index + 1) * 300,
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
