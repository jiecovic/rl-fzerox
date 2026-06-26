# Local Files

The run manager uses this default libretro core path and ROM directory:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/
```

The ROM filename can be arbitrary. Runtime launches scan `local/roms/` for
`.n64`, `.z64`, and `.v64` files and select the first compatible ROM.

Use the US F-Zero X ROM. RAM offsets and telemetry are maintained for that
version.
