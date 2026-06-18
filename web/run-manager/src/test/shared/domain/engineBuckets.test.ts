// web/run-manager/src/test/shared/domain/engineBuckets.test.ts
import { describe, expect, it } from "vitest";

import { centeredEngineBuckets } from "@/shared/domain/engineBuckets";

describe("centeredEngineBuckets", () => {
  it("mirrors slider steps around neutral engine 50", () => {
    expect(centeredEngineBuckets({ minimum: 0, maximum: 128, sideCount: 5 })).toEqual([
      0, 13, 26, 38, 51, 64, 77, 90, 102, 115, 128,
    ]);
  });

  it("keeps the configured range edges as bucket endpoints", () => {
    expect(centeredEngineBuckets({ minimum: 44, maximum: 84, sideCount: 2 })).toEqual([
      44, 54, 64, 74, 84,
    ]);
  });

  it("rejects centered helper ranges that cannot include neutral", () => {
    expect(centeredEngineBuckets({ minimum: 65, maximum: 70, sideCount: 2 })).toEqual([]);
  });
});
