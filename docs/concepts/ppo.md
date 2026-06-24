# PPO

This project trains policies with PPO-family trainers from SB3 and sb3x.

PPO is an on-policy actor-critic algorithm. Training alternates between:

1. collect rollout data with the current policy
2. compute discounted returns and advantages
3. update the policy for several minibatch epochs

The actor update uses the clipped PPO ratio:

```math
r_t(\theta) =
\exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_\mathrm{old}}(a_t \mid s_t))
```

```math
L_\mathrm{policy} =
-\mathbb{E}\left[
\min\left(
r_t(\theta) A_t,\;
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
```

`A_t` is the advantage estimate. Positive advantages push the sampled action up;
negative advantages push it down. `epsilon` is `train.clip_range`.

The optimizer loss used by sb3x is:

```text
policy_loss
+ train.ent_coef * entropy_regularizer_loss
+ train.vf_coef * value_loss
+ optional policy-side auxiliary losses
```

`target_kl` can stop minibatch updates early when the policy moves too far from
the rollout policy.

## Supported Trainers

The supported training algorithms are:

- `maskable_hybrid_action_ppo`
- `maskable_hybrid_recurrent_ppo`

Both require the `configured_hybrid` action adapter. The recurrent variant uses
an LSTM actor path and stores sequence masks plus LSTM states in the rollout
buffer.

## Hybrid Actions

The hybrid action space is:

```text
Dict(
  continuous = Box(-1, 1, shape=(n,)),
  discrete   = MultiDiscrete([...])
)
```

The continuous branch can carry axes such as `steer`, `drive`, `air_brake`, and
`pitch`. The discrete branch can carry actions such as `gas`, `boost`, `lean`,
`spin`, and bucketed `pitch`.

sb3x flattens the sampled action into one tensor for PPO storage:

```text
[continuous values..., discrete indices...]
```

The policy distribution is an independent pair:

- continuous branch: diagonal Gaussian
- discrete branch: multi-categorical distribution

The PPO log probability is the sum of both branches:

```text
log_prob(action) = log_prob(continuous) + log_prob(discrete)
```

Entropy is also available per action group. `train.entropy_group_weights` can
weight selected groups such as `boost`, `lean`, `steer`, or `pitch`.

## Action Masks

Masks apply to the discrete branch. They remove invalid or intentionally disabled
discrete choices before sampling and action evaluation.

Examples:

- disable boost while boost is already active
- disable air brake on ground
- hide spin while the spin cooldown is active
- apply episode-level action masks for ablation

Continuous axes stay continuous and are clipped to `[-1, 1]` before stepping the
environment.

## Auxiliary State Losses

`policy.auxiliary_state` adds supervised heads on the policy latent. The targets
come from RAM-derived state, carried through a hidden training-only observation
field.

Supported target kinds:

- scalar targets use Smooth L1 loss
- binary targets use BCE-with-logits loss
- course-id targets use cross entropy

Examples include speed, energy, airborne state, boost state, track edge ratio,
height above ground, surface flags, and built-in course id.

These losses train the representation used by the policy. Episode rewards remain
the environment reward signal.

## Actor Regularization

`train.actor_regularization` adds optional losses directly on the action
distribution:

- `grounded_pitch_neutral_loss_weight`: keep pitch mean near neutral while
  grounded
- `pitch_std_cap_loss_weight`: cap pitch exploration separately for grounded and
  airborne samples
- `steer_std_cap_loss_weight`: cap continuous steer exploration
- `steer_signed_balance_loss_weight`: penalize persistent left/right steer bias
- `lean_signed_balance_loss_weight`: penalize persistent left/right lean bias

Pitch losses that depend on grounded/airborne state require hidden auxiliary
targets. Continuous steer losses can run without auxiliary targets.

These terms are regularizers. Start with small weights and check TensorBoard
metrics under `train_aux/`, `train_entropy/`, and `train_std/`.

## Code Paths

- model construction:
  `src/rl_fzerox/core/training/session/model/builders.py`
- policy selection and kwargs:
  `src/rl_fzerox/core/training/session/model/policy.py`
- hybrid action adapter:
  `src/rl_fzerox/core/envs/actions/configured/hybrid.py`
- auxiliary state heads:
  `src/rl_fzerox/core/policy/auxiliary_state/heads.py`
- actor regularization:
  `src/rl_fzerox/core/policy/auxiliary_state/actor_regularization.py`
- sb3x hybrid distribution:
  `sb3x.common.hybrid_action.distributions`
- sb3x masked hybrid PPO:
  `sb3x.ppo_mask_hybrid_action.ppo_mask_hybrid_action`
- sb3x masked recurrent hybrid PPO:
  `sb3x.ppo_mask_hybrid_recurrent.ppo_mask_hybrid_recurrent`

## References

- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Schulman et al., High-Dimensional Continuous Control Using Generalized
  Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Huang and Ontanon, A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms](https://arxiv.org/abs/2006.14171)
- [Stable-Baselines3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [SB3-Contrib Maskable PPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [SB3-Contrib Recurrent PPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
