# `race_v2`

`race_v2` is the current reward profile for training race completion instead of
short-horizon driving tricks.

## Plan

- Remove dense per-step frontier shaping.
- Use coarse one-time milestones instead of trusting `race_distance` as a
  precise local line-quality signal.
- Reward lap completion with increasing bonuses.
- Reward better placement at lap boundaries and more strongly on final finish.
- Penalize time on every internal emulator frame so earlier completion stays
  better than slower completion.
- Keep energy loss negative and energy refill positive, but smaller than the
  loss penalty.
- Make death and truncation penalties dynamic:
  - base penalty
  - plus remaining-step pressure derived from `time_penalty_per_frame`
  - plus remaining-lap penalty based on live game telemetry

## Runtime assumptions

- `race_v2` does not carry hidden race-format constants anymore.
- Total laps and total racers are read from live emulator RAM through the native
  telemetry path.
- The reverse-engineered RAM layout currently targets the US ROM build.

## Terms

- `time_penalty_per_frame`
  - Small negative reward per internal emulator frame.
- `milestone_distance`
  - Absolute `race_distance` spacing for one-time milestone rewards.
- `milestone_bonus`
  - One-time reward paid for each newly crossed milestone bucket.
- `lap_1_completion_bonus`
  - Reward for first completed lap.
- `lap_2_completion_bonus`
  - Reward for second completed lap.
- `final_lap_completion_bonus`
  - Reward for the final completed lap that actually finishes the race.
- `lap_position_scale`
  - Small placement bonus scale applied when a lap is completed.
- `finish_position_scale`
  - Larger placement bonus scale applied on actual race finish.
- `remaining_lap_penalty`
  - Extra penalty per unfinished lap on death or truncation.
- `energy_loss_penalty_scale`
  - Penalty scale applied to `energy_loss_total`.
- `energy_gain_reward_scale`
  - Smaller reward scale applied to `energy_gain_total`.

## Current shape

Ordinary step:

- negative time reward
- one-time milestone reward
- lap completion reward if a new lap was crossed
- lap placement reward if a new lap was crossed
- energy loss penalty
- smaller energy refill reward
- collision / spin entry penalties

Finish:

- final lap completion reward
- final placement reward

Crash / retire / off-track / truncation:

- base failure penalty
- plus remaining-step penalty from skipped future time cost
- plus remaining-lap penalty from unfinished race progress
