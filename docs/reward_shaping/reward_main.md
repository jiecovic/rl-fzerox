# Reward Main

`reward_main` is the current canonical reward profile.

It is a spline-frontier reward: the agent receives progress only when it moves
the episode frontier forward into new distance buckets. This keeps reward tied
to newly covered track progress instead of repeatedly paying the same local
position.

## Main Terms

The profile combines:

- frontier progress bucket rewards
- optional time, reverse, and slow-speed penalties
- lap completion and race-position bonuses
- KO star rewards for GP races
- progress multipliers for energy refill, dirt, and ice surfaces
- boost request and boost-pad event rewards
- lean, air-brake, grounded-pitch, impact, and energy-change shaping
- airborne landing reward support
- failure and truncation penalties
- optional per-step reward clipping

Energy gain reward is proportional to energy gained in the step. Energy-refill
progress rewards remain progress-gated so refill behavior cannot be farmed by
going backward over the same local region.

Impact penalty is frame-based and covers native impact/recoil signals exposed by
the emulator summary. It replaces the older separate damage streak/ramp style.

## Configuration

The public runtime schema is `RewardConfig` in
`src/rl_fzerox/core/runtime_spec/schema/env.py`. Manager-owned run specs project
their reward section into that schema before launch.

Course-specific overrides are available through `reward.course_overrides`. Each
override inherits the base `reward_main` weights and replaces only the fields it
sets.

## Ownership

Rust/native stepping provides the frame summary and telemetry used by reward
calculation. Python owns the reward terms, episode-local frontier tracking, and
the final Gym reward/info breakdown.
