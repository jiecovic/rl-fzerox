// web/run-manager/src/test/shared/domain/engineBuckets.test.ts
import { describe, expect, it } from "vitest";

import { centeredEngineBuckets } from "@/shared/domain/engineBuckets";

describe("centeredEngineBuckets", () => {
  it("mirrors slider steps around neutral engine 50", () => {
    expect(centeredEngineBuckets({ minimum: 0, maximum: 128, sliderSpacing: 13 })).toEqual([
      0, 12, 25, 38, 51, 64, 77, 90, 103, 116, 128,
    ]);
  });

  it("clips centered steps and keeps selected range edges", () => {
    expect(centeredEngineBuckets({ minimum: 35, maximum: 65, sliderSpacing: 10 })).toEqual([
      35, 44, 54, 64, 65,
    ]);
  });

  it("keeps narrow range edges even when the centered grid misses the range", () => {
    expect(centeredEngineBuckets({ minimum: 65, maximum: 70, sliderSpacing: 10 })).toEqual([
      65, 70,
    ]);
  });
});
