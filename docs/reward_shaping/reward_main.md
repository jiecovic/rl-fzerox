# Reward Main

The reward uses F-Zero X's **race progress from RAM**. The game advances that
value along the **course spline/centerline**.

The reward tracker turns that RAM progress into **one-way progress buckets**:

- each bucket can pay at most once per episode
- driving backward or oscillating over the same section does not pay again
- progress can be suspended while the machine is too far outside track bounds

Main signal: **gated progress reward**.

For a fixed course, the total available progress reward is mostly fixed: finish
the same course and you cross the same progress buckets. PPO still prefers
getting that reward earlier because **discounting** makes future rewards worth
less than immediate rewards.

`time_penalty_per_frame` can add extra **time pressure**. Use it carefully: if
it is too strong, the policy can learn to avoid long episodes instead of
learning to finish the race.

## Speed Multiplier

The **speed multiplier curve** can scale progress reward by current speed. Keep
it moderate: if it dominates, the policy may chase speed instead of clean course
progress.

## Other Shaping Knobs

The run manager exposes additional reward knobs around the main progress signal,
including:

- speed and race-position progress multipliers
- per-frame time penalty
- lap completion and finish-position bonuses
- off-track recovery reward
- dirt, ice, and energy-refill progress multipliers
- energy gain and energy loss terms
- manual boost request reward
- boost-pad reward
- air-brake, spin, and lean request penalties
- impact and grounded-pitch penalties
- airborne landing reward
- KO star reward in GP races
- failure, truncation, and per-step clipping

These are **secondary shaping terms**. They should help the policy learn useful
driving behavior without replacing progress as the core objective.

## Overrides

**Course-specific reward overrides** can adjust individual fields for tracks
that need different shaping. Unset fields inherit the base `reward_main` values.
