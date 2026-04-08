# docs/reward_shaping/RACE_V2.md

# Race V2 Reward

`race_v2` is the first simplified replacement for the original highly-shaped
reward draft.

## Goal

Optimize for:

- finishing the race
- finishing sooner
- avoiding crashes, spins, and off-track failures
- avoiding wasteful damage and reverse driving

It intentionally removes many special-case incentives so the objective is easier
to reason about and harder to exploit.

## Formula

Per emulated game frame:

```text
reward =
  time_penalty
  + progress_frontier_reward
  + reverse_progress_penalty
  + energy_loss_penalty
  + event_penalties
  + terminal_success_reward
```

### Terms

- `time_penalty`
  - fixed negative reward every emulated frame
  - pushes the policy to finish in less in-game time

- `progress_frontier_reward`
  - only rewards new best `race_distance`
  - does not reward local forward/backward yo-yo movement

- `reverse_progress_penalty`
  - penalizes meaningful backward movement

- `energy_loss_penalty`
  - penalizes losing energy, regardless of the cause

- `event_penalties`
  - entry-only penalties for:
    - collision recoil
    - spinning out
    - falling off track
    - crashing
    - retiring

- `terminal_success_reward`
  - finish bonus
  - plus placement bonus based on final race position

### Truncation penalties

Additional penalties apply on truncation:

- `stuck`
- `wrong_way`
- `timeout`

## Removed from v1

`race_v2` intentionally removes these shaping terms:

- checkpoint bonus
- dash-pad bonus
- refill bonus
- low-speed penalty
- low-energy manual-boost penalty
- stall penalty

The simplified design relies on:

- progress
- time pressure
- damage costs
- failure costs

instead of many overlapping special-case bonuses.

## Calibration rule

Failure should be worse than a bad but legitimate finish.

The intended tuning rule is:

- a slow ugly 3-lap finish should still beat crashing out early
- suicide should never be an efficient shortcut to save time

That should be checked against real run logs before treating the weights as
final.
