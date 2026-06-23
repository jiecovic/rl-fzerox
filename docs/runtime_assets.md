# Runtime Assets

`local/` is the machine-local runtime workspace. Keep ROMs, emulator cores,
save states, SRAM files, generated baselines, checkpoints, TensorBoard logs, and
other run artifacts there or in another ignored local cache path.

The repository includes one small class of binary runtime data:

```text
rust/core/minimap/masks/angrylion/*.bin
rust/core/minimap/masks/gliden64/*.bin
```

Those files are renderer-specific minimap observation masks embedded by
`rust/core/minimap/catalog.rs`. They are fixed-size byte masks for the minimap
ROI and contain only `0` and `255` values. They describe which pixels belong to
the minimap overlay.

Game ROMs, libretro cores, save states, SRAM files, screenshots, texture dumps,
generated baselines, checkpoints, TensorBoard logs, run exports, and SQLite
databases stay in ignored local/cache paths.
