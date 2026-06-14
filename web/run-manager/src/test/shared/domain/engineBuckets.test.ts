// web/run-manager/src/test/shared/domain/engineBuckets.test.ts
import { describe, expect, it } from "vitest";

import { centeredEngineBuckets } from "@/shared/domain/engineBuckets";

describe("centeredEngineBuckets", () => {
  it("mirrors buckets around neutral engine 50", () => {
    expect(centeredEngineBuckets({ minimum: 0, maximum: 100, bucketSize: 20 })).toEqual([
      10, 30, 50, 70, 90,
    ]);
  });

  it("clips buckets to the selected range without forcing range edges", () => {
    expect(centeredEngineBuckets({ minimum: 35, maximum: 65, bucketSize: 10 })).toEqual([
      40, 50, 60,
    ]);
  });

  it("returns no buckets when the centered grid misses the range", () => {
    expect(centeredEngineBuckets({ minimum: 51, maximum: 55, bucketSize: 10 })).toEqual([]);
  });
});
