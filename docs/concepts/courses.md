# Courses and Race Modes

This project treats a "course" as a reset target for training, evaluation, and
watch playback. A target is more than a visible track name: it also includes the
race mode, GP difficulty when applicable, vehicle, engine setting, and the
run-local baseline savestate used to enter the race.

## Fixed Courses

The fixed F-Zero X course pool is the set of built-in courses exposed by the run
manager. In the UI these are grouped by cup:

- Jack Cup
- Queen Cup
- King Cup
- Joker Cup

The selector currently treats those 24 fixed courses as ordinary reusable
training/evaluation targets. X-Cup is handled separately because the game
generates the course layout at runtime.

## Race Modes

`gp_race` is the normal Grand Prix race mode. It has opponents, a vehicle, and
one GP difficulty. In managed evaluations the preset should choose exactly one
GP difficulty for a GP target. A multi-difficulty benchmark would be a different
benchmark definition, not one ambiguous preset.

`time_attack` is a single-course time-trial mode. It has no opponent grid and no
GP difficulty. Time Attack baselines are therefore simpler and do not need GP
opponent-grid variants.

## Track Sampling Entries

Training can sample among multiple reset targets. Each
`TrackSamplingEntryConfig` is one reset candidate with stable identity and
materialized baseline information:

- `course_id` is the logical fixed or generated course id when known.
- `runtime_course_key` is the sampling key used by runtime statistics.
- `course_index` is the game/menu course index used during materialization.
- `mode` is usually `gp_race` or `time_attack`.
- `gp_difficulty` applies only to GP race mode.
- `baseline_state_path` points to the run-local savestate after materialization.
- `source_*` fields preserve the original materialization target after a
  run-local baseline file has been written.

The `runtime_course_key` matters for generated courses and for stable statistics.
It lets a generated X-Cup slot keep its runtime identity even when the generated
course id changes after rotation.

## Evaluation Targets

Evaluation presets describe benchmark environment targets: mode, selected
courses or cups, GP difficulty, renderer, seed, repeats, and runtime settings.
They should not own the source checkpoint, checkpoint artifact such as `latest`,
policy mode, device, or copied model files. Those belong to the evaluation
snapshot/run.

For cross-course evaluation summaries, course-level statistics are usually more
useful than global "best time" or total env steps. Finish distribution, weakest
courses, completion, and per-course position/return are the meaningful signals.

