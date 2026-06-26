// web/run-manager/src/test/app/managerData.test.ts

import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadManagerData } from "@/app/managerData";
import {
  checkpointCatalogFixture,
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
} from "@/test/fixtures";

const fetchCheckpointCatalogMock = vi.fn();
const fetchConfigMetadataMock = vi.fn();
const fetchDraftsMock = vi.fn();
const fetchEvaluationDataMock = vi.fn();
const fetchRunsMock = vi.fn();
const fetchSaveGamesMock = vi.fn();
const fetchTemplatesMock = vi.fn();

vi.mock("@/shared/api/client", () => ({
  fetchCheckpointCatalog: () => fetchCheckpointCatalogMock(),
  fetchConfigMetadata: () => fetchConfigMetadataMock(),
  fetchDrafts: () => fetchDraftsMock(),
  fetchEvaluationData: () => fetchEvaluationDataMock(),
  fetchRuns: () => fetchRunsMock(),
  fetchSaveGames: () => fetchSaveGamesMock(),
  fetchTemplates: () => fetchTemplatesMock(),
}));

describe("loadManagerData", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchCheckpointCatalogMock.mockResolvedValue(checkpointCatalogFixture());
    fetchConfigMetadataMock.mockResolvedValue(configMetadataFixture);
    fetchDraftsMock.mockResolvedValue([draftFixture()]);
    fetchEvaluationDataMock.mockResolvedValue({
      baseline_suites: [],
      evaluations: [],
      presets: [],
    });
    fetchRunsMock.mockResolvedValue([]);
    fetchSaveGamesMock.mockResolvedValue([]);
    fetchTemplatesMock.mockResolvedValue([
      { config: managedRunConfigFixture, id: "template-001", name: "default" },
    ]);
  });

  it("keeps core manager data available when evaluations fail to load", async () => {
    fetchEvaluationDataMock.mockRejectedValueOnce(new Error("backend is missing eval endpoint"));

    const data = await loadManagerData();

    expect(data.drafts).toHaveLength(1);
    expect(data.metadata).toBe(configMetadataFixture);
    expect(data.templates).toHaveLength(1);
    expect(data.evaluations).toEqual([]);
    expect(data.evaluationError).toBe("backend is missing eval endpoint");
  });

  it("keeps core manager data available when checkpoint catalog fails to load", async () => {
    fetchCheckpointCatalogMock.mockRejectedValueOnce(new Error("catalog unavailable"));

    const data = await loadManagerData();

    expect(data.drafts).toHaveLength(1);
    expect(data.metadata).toBe(configMetadataFixture);
    expect(data.checkpointCatalog).toBeNull();
    expect(data.checkpointCatalogError).toBe("catalog unavailable");
  });
});
