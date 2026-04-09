# `race_v2`

`race_v2` is the current reward profile for training race completion instead of
short-horizon driving tricks.

## Plan

- Keep signed dense progress shaping only as a bootstrap signal for the first
  few configured completed laps, with an optional small race-position
  multiplier.
- Use coarse one-time milestones after that instead of trusting `race_distance`
  as a precise local line-quality signal for the whole episode.
- Optionally add a capped milestone speed bonus from average progress per frame
  since the previous milestone.
- Reward lap completion with increasing bonuses.
- Reward better placement at lap boundaries and more strongly on final finish.
- Penalize time on every internal emulator frame so earlier completion stays
  better than slower completion.
- Penalize reverse-driving frames more strongly by scaling the normal per-frame
  time penalty while the game reverse-warning timer is active.
- Penalize low-speed frames more strongly by scaling the normal per-frame time
  penalty whenever speed drops below the env stuck-speed threshold.
- Keep energy refill positive, but penalize energy loss only when post-step
  energy falls below the configured safe fraction.
- Suppress energy-refill reward for a short configurable cooldown after entering
  collision recoil so crashes into refill zones are not rewarded immediately.
- Make death and truncation penalties dynamic:
  - base penalty
  - plus explicit remaining-step pressure from
    `remaining_step_penalty_per_frame`
  - plus remaining-lap penalty based on live game telemetry

## Runtime assumptions

- `race_v2` does not carry hidden race-format constants anymore.
- Total laps and total racers are read from live emulator RAM through the native
  telemetry path.
- Completed race laps follow the HUD lap number (`lap - 1`), not the raw
  `laps_completed` RAM field, because that raw field increments on the initial
  start-line crossing while the HUD still shows lap 1/3.
- Env info and training logs use `race_laps_completed` for that HUD-aligned
  value; the raw RAM field is kept only as `raw_laps_completed` for debugging.
- The reverse-engineered RAM layout currently targets the US ROM build.
- Reverse-driving punishment starts as soon as the game's live reverse timer is
  non-zero. Wrong-way truncation uses the configured env timer limit directly.
- If `env.terminate_on_energy_depleted` is enabled, the env ends the episode as
  `energy_depleted` once in-race player energy reaches zero, even before the
  game sets its later crash/game-over flags.

## Terms

- `time_penalty_per_frame`
  - Small negative reward per internal emulator frame.
- `milestone_distance`
  - Episode-relative progress spacing for one-time milestone rewards, measured
    from the reset-time `race_distance` origin.
- `milestone_bonus`
  - One-time reward paid for each newly crossed milestone bucket.
- `milestone_speed_scale`
  - Optional extra reward scale for reaching milestones faster, computed from
    crossed race distance divided by internal frames since the previous
    milestone.
- `milestone_speed_bonus_cap`
  - Maximum extra speed reward paid per crossed milestone.
- `bootstrap_progress_scale`
  - Dense reward scale for positive race-distance delta during the configured
    bootstrap lap window.
- `bootstrap_regress_penalty_scale`
  - Dense penalty scale for negative race-distance delta during the configured
    bootstrap lap window. Keep this higher than `bootstrap_progress_scale` so
    back-and-forth oscillation is net negative.
- `bootstrap_position_multiplier_scale`
  - Max extra multiplier for signed dense bootstrap progress/regress based on
    race position: first place gets `1 + scale`, last place stays `1.0`.
- `bootstrap_lap_count`
  - Number of initial completed laps that still allow dense bootstrap progress
    reward before the profile switches fully to milestone-only progress shaping.
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
  - Maximum penalty scale applied to `energy_loss_total` when post-step energy
    is near zero.
- `energy_loss_safe_fraction`
  - Post-step energy fraction above which energy loss is not penalized.
- `energy_loss_danger_power`
  - Exponent for the danger curve below `energy_loss_safe_fraction`; higher
    values keep the penalty mild until energy gets lower.
- `energy_gain_reward_scale`
  - Smaller reward scale applied to `energy_gain_total`.
- `energy_gain_collision_cooldown_frames`
  - Internal-frame cooldown after entering collision recoil during which
    `energy_gain_total` is not rewarded.
- `boost_redundant_press_penalty`
  - Fixed penalty when the agent requests boost on an env step whose pre-step
    telemetry already had `boost_timer > 0`.

## Current shape

Ordinary step:

- negative time reward
- bootstrap signed progress reward or regress penalty during the configured
  initial lap warmup window
- one-time milestone reward
- optional milestone speed reward
- lap completion reward if a new lap was crossed
- lap placement reward if a new lap was crossed
- extra reverse-driving time penalty while the live reverse timer is active
- extra low-speed time penalty while speed is below the stuck-speed threshold
- danger-weighted energy loss penalty below the configured safe fraction
- smaller energy refill reward, unless collision cooldown is active
- fixed redundant-boost penalty when boost was already active before the step
- collision / spin entry penalties

Finish:

- final lap completion reward
- final placement reward

Crash / retire / off-track / energy-depleted / truncation:

- base failure penalty
- plus remaining-step penalty from `remaining_step_penalty_per_frame`
- plus remaining-lap penalty from unfinished race progress
