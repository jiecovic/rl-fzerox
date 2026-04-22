# Checkpoints

No model checkpoint is currently tracked here.

Runtime assets stay local:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/F-Zero X (USA).n64
local/runs/
local/cache/baseline_materializer/
```

Training materializes run-local reset states and stores policy artifacts under
the specific run directory. Do not commit ROMs, generated save states, or model
artifacts unless there is an explicit reason to publish a curated checkpoint.
