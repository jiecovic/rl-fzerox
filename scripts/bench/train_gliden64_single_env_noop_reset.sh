#!/usr/bin/env bash
# scripts/bench/train_gliden64_single_env_noop_reset.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv/bin/python -m rl_fzerox.apps.train \
  --config conf/local/archive/train.local.ppo_hybrid_full_gas_v3_stack4.yaml \
  -- \
  emulator.renderer=gliden64 \
  train.num_envs=1 \
  train.vec_env=dummy \
  env.benchmark_noop_reset=true \
  train.total_timesteps=200000
