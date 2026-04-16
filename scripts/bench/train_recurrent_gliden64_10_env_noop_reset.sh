#!/usr/bin/env bash
# scripts/bench/train_recurrent_gliden64_10_env_noop_reset.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv/bin/python -m rl_fzerox.apps.train \
  --config conf/local/active/train.local.ppo_hybrid_full_gas_recurrent_v3.yaml \
  -- \
  emulator.renderer=gliden64 \
  train.num_envs=10 \
  train.vec_env=subproc \
  env.benchmark_noop_reset=true \
  train.total_timesteps=200000
