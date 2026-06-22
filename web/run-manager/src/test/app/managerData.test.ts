// web/run-manager/src/test/app/managerData.test.ts

import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadManagerData } from "@/app/managerData";
import { configMetadataFixture, draftFixture, managedRunConfigFixture } from "@/test/fixtures";

const fetchConfigMetadataMock = vi.fn();
const fetchDraftsMock = vi.fn();
const fetchEvaluationsMock = vi.fn();
const fetchRunsMock = vi.fn();
const fetchSaveGamesMock = vi.fn();
const fetchTemplatesMock = vi.fn();

vi.mock("@/shared/api/client", () => ({
  fetchConfigMetadata: () => fetchConfigMetadataMock(),
  fetchDrafts: () => fetchDraftsMock(),
  fetchEvaluations: () => fetchEvaluationsMock(),
  fetchRuns: () => fetchRunsMock(),
  fetchSaveGames: () => fetchSaveGamesMock(),
  fetchTemplates: () => fetchTemplatesMock(),
}));

describe("loadManagerData", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchConfigMetadataMock.mockResolvedValue(configMetadataFixture);
    fetchDraftsMock.mockResolvedValue([draftFixture()]);
    fetchEvaluationsMock.mockResolvedValue([]);
    fetchRunsMock.mockResolvedValue([]);
    fetchSaveGamesMock.mockResolvedValue([]);
    fetchTemplatesMock.mockResolvedValue([
      { config: managedRunConfigFixture, id: "template-001", name: "default" },
    ]);
  });

  it("keeps core manager data available when evaluations fail to load", async () => {
    fetchEvaluationsMock.mockRejectedValueOnce(new Error("backend is missing eval endpoint"));

    const data = await loadManagerData();

    expect(data.drafts).toHaveLength(1);
    expect(data.metadata).toBe(configMetadataFixture);
    expect(data.templates).toHaveLength(1);
    expect(data.evaluations).toEqual([]);
    expect(data.evaluationError).toBe("backend is missing eval endpoint");
  });
});
