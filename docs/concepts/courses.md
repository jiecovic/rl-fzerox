# Training Targets and Race Modes

In the run manager you choose the **training targets** for a run. A target is
one concrete F-Zero X race start:

- course
- race mode
- GP difficulty, if GP Race is used
- selected ship/vehicle
- engine setting

For each target, the run manager creates or reuses a baseline savestate. That
file lets the emulator reset directly back to the exact race start.

## Built-In Courses

The normal F-Zero X courses are built into the game. In the UI they are grouped
by cup:

- Jack Cup
- Queen Cup
- King Cup
- Joker Cup

For training, evaluation, and watch playback, you can select any subset of those
24 tracks.

## X-Cup

X-Cup can also be part of the training targets. It behaves differently from the
built-in courses because F-Zero X generates the track layout at runtime.

See [Generated X-Cup Courses](x_cup.md) for the generated-course details.

## Race Modes

`gp_race` is the normal Grand Prix mode: opponents, one selected ship/vehicle,
engine setting, and one GP difficulty.

`time_attack` is a single-track time trial without opponent grid or GP
difficulty.

### GP Race Starts

F-Zero X GP Race is cup-based. The normal game flow plays each cup in order.

For training, the materializer creates a baseline savestate for the requested GP
target. It navigates to a GP race setup, writes the requested course/setup into
RAM, then forces the game to rebuild the race start. The result is a reset state
for one specific course, difficulty, selected ship/vehicle, and engine setting.

## Course Sampling

If a run has multiple training targets, the course sampler decides which target
the next reset uses.

### Equal

`equal` cycles through the selected targets evenly. It is useful when every
target should get roughly the same number of resets.

Note: `equal` balances reset count. A target with longer episodes can still
receive more total training frames.

### Step Balanced

`step_balanced` balances by training frames instead of reset count. That matters
because one target can produce much longer episodes than another.

The sampler tracks how many frames each target has already received and raises
the reset weight for targets that are behind their frame share.

### Fixed Env

`fixed_env` pins each parallel env worker to one target slot. This gives
deterministic per-rollout coverage, but it only fits when the number of envs can
cover the selected targets cleanly.

### Deficit Budget

`deficit_budget` assigns resets from step-budget accounts. Each rollout adds a
new budget, and targets build up deficit when they have received fewer env steps
than their current target share.

The budget is split into two lanes:

- **uniform lane**: keeps broad coverage across all selected targets
- **difficulty lane**: focuses targets with low completion or finish rate

`deficit_budget_uniform_fraction` controls the split. With the default `0.7`,
70% of the budget stays uniform and 30% can focus difficult targets.

The difficulty lane uses one selected metric based on recent EMA stats:

- `completion_ema`: focus targets with low completion
- `finish_ema`: focus targets with low finish rate
- `mixed`: use the worse of completion and finish rate

`deficit_budget_focus_sharpness` controls how aggressively the difficulty lane
focuses the weakest targets. Higher values concentrate more budget on the worst
targets.

`deficit_budget_warmup_min_episodes_per_course` keeps the sampler broad until
each target has enough samples. `deficit_budget_uniform_staleness_rotations`
prevents the uniform lane from ignoring a target for too long.
