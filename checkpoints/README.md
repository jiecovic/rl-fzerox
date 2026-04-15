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

Reusable reset baselines:

```text
checkpoints/baselines/mute-city-novice-blue-falcon-balanced-go-window.state
```

Current checkpoint:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml --run-dir checkpoints/mute-city-3steer-recurrent-ppo-v1 --artifact best
```
