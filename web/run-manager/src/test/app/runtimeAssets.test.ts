// web/run-manager/src/test/app/runtimeAssets.test.ts
import { describe, expect, it } from "vitest";

import { runtimeAssetNotice } from "@/app/runtimeAssets";
import { configMetadataFixture } from "@/test/fixtures";

describe("runtime asset notice", () => {
  it("omits the notice when runtime assets exist", () => {
    expect(
      runtimeAssetNotice({
        ...configMetadataFixture,
        runtime_assets: configMetadataFixture.runtime_assets.map((asset) => ({
          ...asset,
          exists: true,
        })),
      }),
    ).toBeNull();
  });

  it("lists missing runtime assets with their local paths", () => {
    const notice = runtimeAssetNotice({
      ...configMetadataFixture,
      runtime_assets: configMetadataFixture.runtime_assets.map((asset) => ({
        ...asset,
        exists: false,
      })),
    });

    expect(notice?.key).toBe(
      "libretro_core:local/libretro/mupen64plus_next_libretro.so|fzerox_rom:" +
        "local/roms/fzerox_usa.n64",
    );
    expect(notice?.message).toContain("Missing runtime assets");
    expect(notice?.message).toContain("local/libretro/mupen64plus_next_libretro.so");
    expect(notice?.message).toContain("local/roms/fzerox_usa.n64");
    expect(notice?.message).toContain("refresh this page");
  });
});
