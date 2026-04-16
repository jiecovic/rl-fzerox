#!/usr/bin/env bash
# scripts/bench/train_angrylion_10_env_noop_reset.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv/bin/python -m rl_fzerox.apps.train \
  --config conf/local/archive/train.local.ppo_hybrid_full_gas_v3_stack4.yaml \
  -- \
  emulator.renderer=angrylion \
  train.num_envs=10 \
  train.vec_env=subproc \
  env.benchmark_noop_reset=true \
  train.total_timesteps=200000
