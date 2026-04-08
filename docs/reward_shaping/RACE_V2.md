# `race_v2`

`race_v2` is the current reward profile for training race completion instead of
short-horizon driving tricks.

## Plan

- Keep dense frontier shaping only as a bootstrap signal until milestone 1.
- Use coarse one-time milestones after that instead of trusting `race_distance`
  as a precise local line-quality signal for the whole episode.
- Reward lap completion with increasing bonuses.
- Reward better placement at lap boundaries and more strongly on final finish.
- Penalize time on every internal emulator frame so earlier completion stays
  better than slower completion.
- Penalize reverse-driving frames more strongly by scaling the normal per-frame
  time penalty while the game reverse-warning timer is active.
- Penalize low-speed frames more strongly by scaling the normal per-frame time
  penalty whenever speed drops below the env stuck-speed threshold.
- Keep energy loss negative and energy refill positive, but smaller than the
  loss penalty.
- Make death and truncation penalties dynamic:
  - base penalty
  - plus explicit remaining-step pressure from
    `remaining_step_penalty_per_frame`
  - plus remaining-lap penalty based on live game telemetry

## Runtime assumptions

- `race_v2` does not carry hidden race-format constants anymore.
- Total laps and total racers are read from live emulator RAM through the native
  telemetry path.
- The reverse-engineered RAM layout currently targets the US ROM build.
- Reverse-driving punishment starts as soon as the game's live reverse timer is
  non-zero. The on-screen HUD warning still appears later at the game's hard
  threshold (`100`), and wrong-way truncation remains a separate env limit
  based on that same reverse timer.

## Terms

- `time_penalty_per_frame`
  - Small negative reward per internal emulator frame.
- `milestone_distance`
  - Absolute `race_distance` spacing for one-time milestone rewards.
- `milestone_bonus`
  - One-time reward paid for each newly crossed milestone bucket.
- `bootstrap_progress_scale`
  - Dense new-best frontier reward used only before the first milestone is
    reached.
- `reverse_time_penalty_scale`
  - Multiplier applied to the normal time penalty on frames where the live game
    reverse timer is active.
- `low_speed_time_penalty_scale`
  - Multiplier applied to the normal time penalty on frames below the env's
    configured stuck-speed threshold.
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
- `remaining_step_penalty_per_frame`
  - Extra penalty per remaining internal frame on death or truncation.
- `energy_loss_penalty_scale`
  - Penalty scale applied to `energy_loss_total`.
- `energy_gain_reward_scale`
  - Smaller reward scale applied to `energy_gain_total`.

## Current shape

Ordinary step:

- negative time reward
- bootstrap frontier reward on new best progress before milestone 1 only
- one-time milestone reward
- lap completion reward if a new lap was crossed
- lap placement reward if a new lap was crossed
- extra reverse-driving time penalty while the live reverse timer is active
- extra low-speed time penalty while speed is below the stuck-speed threshold
- energy loss penalty
- smaller energy refill reward
- collision / spin entry penalties

Finish:

- final lap completion reward
- final placement reward

Crash / retire / off-track / truncation:

- base failure penalty
- plus remaining-step penalty from `remaining_step_penalty_per_frame`
- plus remaining-lap penalty from unfinished race progress
