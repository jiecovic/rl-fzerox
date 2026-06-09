# Runtime Assets

`local/` is the machine-local runtime workspace. Keep ROMs, emulator cores,
save states, SRAM files, generated baselines, checkpoints, TensorBoard logs, and
other run artifacts there or in another ignored local cache path.

The only intentionally tracked binary runtime assets are:

```text
rust/core/minimap/masks/angrylion/*.bin
rust/core/minimap/masks/gliden64/*.bin
```

Those files are renderer-specific minimap observation masks embedded by
`rust/core/minimap/catalog.rs`. They are fixed-size byte masks for the minimap
ROI and contain only `0` and `255` values; they are not ROM bytes, save-state
memory, screenshots, texture dumps, or model checkpoints.

Before adding any new tracked binary asset, verify it with `git ls-files`, keep
copyrighted game/runtime material out of git, and document why the asset is safe
to distribute.
