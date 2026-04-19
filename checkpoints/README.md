# Checkpoints

Small curated checkpoint drops live here. Large model and policy zips are tracked
with Git LFS.

The ROM and emulator core are still local files. The vendored config points at
the usual local paths and includes its own baseline state:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/F-Zero X (USA).n64
checkpoints/mute-city-3steer-recurrent-ppo-v1/baseline.state
```

Legacy concrete reset-state sources:

```text
checkpoints/source_states/legacy_go_window/mute-city-novice-blue-falcon-balanced-go-window.state
checkpoints/source_states/legacy_go_window/jack/*.state
```

These are already-initialized race-start states for specific course/vehicle/
engine combinations. Training and watch runs materialize run-local reset states
under `local/cache/baseline_materializer/` and copy them into the run folder.
Do not treat these checked-in states as neutral sources for arbitrary vehicles.

Current checkpoint:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml --run-dir checkpoints/mute-city-3steer-recurrent-ppo-v1 --artifact best
```
