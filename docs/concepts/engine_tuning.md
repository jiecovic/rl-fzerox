# Engine Tuning

F-Zero X lets the player tune each vehicle between **acceleration** and **top
speed** before a race. Lower engine settings favor acceleration, so the vehicle
gets back to speed faster after starts, turns, wall hits, and speed loss. Higher
engine settings favor maximum speed, so the vehicle can be faster on straights
but accelerates more slowly.

The game shows the setting as a rounded engine percentage in the setup screen.
Internally this project stores it as a raw slider step from `0` to `128`. A
policy can behave differently at different engine settings, and the best setting
can vary by course.

Training a separate policy for every engine value would be expensive. The tuner
samples a small bucket list during training and keeps statistics for the engine
values that were actually tried.

## Engine Modes

The Run Manager has three engine modes:

- `fixed`: train with one engine slider value.
- `random_range`: sample one slider value from the configured range on reset.
- `adaptive_tuner`: let the tuner choose from configured engine buckets.

For `random_range` and `adaptive_tuner`, keep
`machine_context.engine` enabled in the policy state input. It gives the policy
the current engine slider value. See [Policy inputs](policy_inputs.md).

## Training Progression

A useful progression is:

1. Start with `fixed` near the balanced 50% setup.
2. Move to `random_range` and widen the range over time.
3. Use `adaptive_tuner` near the end to search for the best bucket for the
   selected score or metric.

The final tuner phase works best after the policy can already finish enough
episodes for the bucket statistics to mean something.

## Goal

The tuner should answer this reset-time question:

```text
For this course and vehicle, which engine bucket should the next episode use?
```

The policy keeps learning while the tuner explores engine settings. Finished
episodes feed speed and reliability data back into the tuner.

## Contexts and Arms

An engine-tuning **context** is currently:

```text
(course_key, vehicle_id)
```

For built-in courses, `course_key` is the runtime course key or course id. For
generated X-Cup targets, the context uses `x_cup`.

An **arm** is currently one engine bucket inside that context:

```text
(course_key, vehicle_id, engine_setting_raw_value)
```

The default bucket list is centered around the neutral slider step. The same
model can later use a larger arm definition, for example
`(vehicle_id, engine_setting_raw_value)` when vehicle selection should be tuned
together with engine setup.

## Episode Data

The tuner records regular training episodes and ignores alternate-baseline
inspection samples.

Each recorded episode contributes:

- completion fraction
- finished or failed
- finish time when the race was completed
- finish position when available
- episode return when available

Failed episodes still update episode count, completion, finish rate, and return
statistics. Finish-time objectives use successful race times as their time
observations.

## Bandit Theory

The maintained backend is a discrete **multi-armed bandit** over engine buckets.
At reset time it chooses one bucket for the active context. This is close to
Thompson sampling: draw one plausible score for each bucket, then use the bucket
with the highest draw.

Selection works in this order:

1. `uniform_exploration` can choose any bucket from a flat distribution.
2. Buckets with no useful observations are sampled first.
3. Observed buckets are sampled with a Thompson-sampling-style draw.

Notation:

- `a` is one engine bucket, for example raw slider value `64`.
- `e_a` is the number of recorded episodes for bucket `a`.
- `f_a` is the number of finished episodes for bucket `a`.
- `n_a` is the active score count used by the current objective.
- `t` is finish time in seconds.
- `s` is the higher-is-better score used by the tuner.
- `mu_a` is the current mean score estimate for bucket `a`.
- `sigma_a` is the sampling uncertainty for bucket `a`.

### Finish Time

For a successful finish, lower time becomes a higher score:

```math
s = -t_\mathrm{seconds}
```

For `finish_time`, each bucket draws a sampled score:

```math
\tilde{s}_a = \mu_a + \sigma_a \epsilon,\quad \epsilon \sim \mathcal{N}(0, 1)
```

The uncertainty term shrinks as the bucket collects observations:

```math
\sigma_a =
\frac{\max(0.25,\ \mathrm{exploration\_seconds})}{\sqrt{\max(1, n_a)}}
```

For `finish_time`, `n_a` counts successful finish-time observations. Failed
episodes still update diagnostics, but they do not create a finish-time score.

### Finish Rate

Finish rate is a Bernoulli signal: an episode either finishes or it does not.
For `finish_rate`, each bucket draws a sampled finish probability:

```math
p_a \sim \mathrm{Beta}(1 + f_a,\; 1 + e_a - f_a)
```

`Beta(alpha, beta)` is a distribution over probabilities from `0` to `1`.
Here `alpha = 1 + f_a` and `beta = 1 + e_a - f_a`. The initial `1, 1` is a flat
prior. Each finish adds one success count. Each failed episode adds one failure
count.

The sampled `p_a` is not the next episode result. It is one plausible finish
rate for bucket `a` given the observed finishes and failures.

### Safe Finish Time

`safe_finish_time` combines both pieces:

1. Draw `p_a` from the Beta finish-rate model.
2. Draw a finish-time score from the Gaussian score model.
3. Keep buckets whose sampled `p_a` clears `safe_finish_rate_threshold` and that
   have at least one real finish.
4. Choose the fastest sampled finish-time score inside that safe set.

When no bucket clears the threshold, selection uses sampled finish rate
first and sampled finish time second. This keeps exploration focused on buckets
that still need reliability.

## Objectives

- `finish_time`: optimize successful finish time.
- `safe_finish_time`: require a minimum finish-rate sample, then optimize finish
  time inside that safe set.
- `finish_rate`: optimize finishing reliability.

`safe_finish_time` is the useful objective when the target is "finish at least
often enough, then get faster."

## Greedy Use

Training samples engine buckets. Evaluation and watch playback should use the
saved tuner state in greedy mode so the policy runs with the best known engine
setting instead of continuing exploration.

## Notes

The bandit backend is maintained. Gaussian-process and MLP-ensemble backends are
still loadable for old configs and experiments, but new run-manager work should
treat Bandit as the supported path.

The finish-rate model is the standard Beta-Bernoulli update used in Thompson
sampling for Bernoulli rewards. The finish-time model is lighter: it uses an
aggregate mean plus a Gaussian uncertainty draw, not a full Bayesian time model.

The code path for the maintained backend is
`src/rl_fzerox/core/engine_tuning/bandit_sampling.py`.

## References

- [Russo et al., A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038)
- [Kaufmann, Korda, and Munos, Thompson Sampling: An Asymptotically Optimal
  Finite-Time Analysis](https://arxiv.org/abs/1205.4217)
- [Agrawal and Goyal, Further Optimal Regret Bounds for Thompson
  Sampling](https://arxiv.org/abs/1209.3353)
- [Bubeck and Cesa-Bianchi, Regret Analysis of Stochastic and Nonstochastic
  Multi-armed Bandit Problems](https://arxiv.org/abs/1204.5721)
