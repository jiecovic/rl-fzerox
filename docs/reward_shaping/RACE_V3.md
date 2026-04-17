# `race_v3`

`race_v3` is the canonical reward profile.

It rewards newly covered progress buckets on the game spline, then layers in
small event rewards and penalties that are hard to farm:

- one-time frontier progress buckets per episode
- per-lap completion and position bonuses
- boost-pad rewards gated by progress windows
- energy-refill progress multipliers and full-refill lap bonuses
- damage, collision recoil, failure, and truncation penalties
- optional landing reward when the car touches down after being airborne

The profile intentionally avoids the older progress reward variants.
Episode urgency should primarily come from discounting, truncation, clean
progress coverage, and optional time penalties in config.
