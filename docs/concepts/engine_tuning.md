# Adaptive Engine Tuning

F-Zero X has an engine slider that trades acceleration against top speed. The
adaptive engine tuner samples engine settings during training and learns which
raw slider values work well for each context.

## Contexts

An engine-tuning context is currently `(course_key, vehicle_id)`. For fixed
courses, `course_key` is the runtime course key or course id. For generated
X-Cup entries, the course key is collapsed to `x_cup`, so X-Cup appears as one
engine-tuning context rather than exposing generated segments as separate
tracks.

This is intentional: the tuner should learn an engine preference for the reset
target family the policy sees, not for internal implementation details.

## Outcomes

The tuner records default-baseline episode outcomes:

- completion fraction
- finished or failed
- finish time when the race was completed
- finish position when available
- episode return when available

Failed episodes still update episode count, completion, finish-rate, and return
statistics. Finish-time objectives only use successful race times as time
observations.

## Bandit Backend

The main backend is a discrete multi-armed bandit over engine slider buckets.
The bucket list is centered by default and validated against the game slider
range.

For the `finish_time` objective, successful finish time is converted into a
higher-is-better score:

```math
s = -t_\mathrm{seconds}
```

The sampled score for one engine value is:

```math
\tilde{s}_a = \mu_a + \sigma_a \epsilon,\quad \epsilon \sim \mathcal{N}(0, 1)
```

where uncertainty decays with the number of observations:

```math
\sigma_a = \frac{\mathrm{exploration\_seconds}}{\sqrt{\max(1, n_a)}}
```

For the `safe_finish_time` objective, the sampler first draws a finish-rate
sample:

```math
p_a \sim \mathrm{Beta}(1 + f_a,\; 1 + e_a - f_a)
```

where `e_a` is episode count and `f_a` is finish count for the candidate. If any
candidate clears the configured safe finish-rate threshold, selection prefers
fast safe candidates. Otherwise it prefers candidates that are more likely to
finish.

The configured `uniform_exploration` mixes the model probabilities with a flat
distribution so every candidate keeps some chance of being sampled:

```math
P(a) = \lambda \frac{1}{|A|} + (1 - \lambda)P_\mathrm{model}(a)
```

## Greedy Selection

Training samples engine values. Evaluation should use the greedy projection from
the saved tuner state so the policy is evaluated with the best known engine
choice instead of continuing exploration.

The greedy rule depends on the objective:

- `finish_time`: prefer lower estimated finish time, then better best time.
- `finish_rate`: prefer higher mean score and finish statistics.
- `safe_finish_time`: prefer safe candidates first, then speed within the safe
  set; if nothing is safe, prefer finishing reliability.

## Experimental Backends

The config still contains Gaussian-process and MLP-ensemble backends. They are
experimental compared with the bandit backend. Keep docs and UI clear about
which backend is currently the default and which fields only apply to
experimental modes.

