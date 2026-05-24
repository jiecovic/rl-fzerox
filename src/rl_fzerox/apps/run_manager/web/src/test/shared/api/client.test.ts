// src/rl_fzerox/apps/run_manager/web/src/test/shared/api/client.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";

import { fetchPolicyPreview } from "@/shared/api/client";
import { managedRunConfigFixture, policyPreviewFixture } from "@/test/fixtures";

describe("shared api client", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("surfaces policy preview failures without a legacy retry", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: "preview failed" }), {
        headers: { "Content-Type": "application/json" },
        status: 500,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchPolicyPreview(managedRunConfigFixture)).rejects.toThrow("preview failed");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("surfaces non-json failed responses by status text", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("Bad Gateway", {
        status: 502,
        statusText: "Bad Gateway",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchPolicyPreview(managedRunConfigFixture)).rejects.toThrow("Bad Gateway");
  });

  it("parses successful policy preview responses", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(policyPreviewFixture), {
        headers: { "Content-Type": "application/json" },
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchPolicyPreview(managedRunConfigFixture)).resolves.toEqual(
      policyPreviewFixture,
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
