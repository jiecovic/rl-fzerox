import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "@/app/App";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
} from "@/test/fixtures";

const loadManagerDataMock = vi.fn();
const createDraftMock = vi.fn();
const updateDraftMock = vi.fn();
const deleteDraftMock = vi.fn();
const fetchPolicyPreviewMock = vi.fn();

vi.mock("@/app/managerData", () => ({
  loadManagerData: () => loadManagerDataMock(),
}));

vi.mock("@/shared/api/client", () => ({
  createDraft: (name: string, config: typeof managedRunConfigFixture) =>
    createDraftMock(name, config),
  deleteDraft: (id: string) => deleteDraftMock(id),
  fetchPolicyPreview: (config: typeof managedRunConfigFixture) => fetchPolicyPreviewMock(config),
  updateDraft: (id: string, name: string, config: typeof managedRunConfigFixture) =>
    updateDraftMock(id, name, config),
}));

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    loadManagerDataMock.mockResolvedValue({
      drafts: [draftFixture()],
      metadata: configMetadataFixture,
      runs: [],
      templates: [{ config: managedRunConfigFixture, id: "template-001", name: "default" }],
    });
    createDraftMock.mockImplementation(async (name: string) => draftFixture({ id: name, name }));
    updateDraftMock.mockImplementation(async (id: string, name: string) =>
      draftFixture({ id, name }),
    );
    deleteDraftMock.mockResolvedValue(undefined);
    fetchPolicyPreviewMock.mockResolvedValue(policyPreviewFixture);
  });

  it("keeps multiple draft editors open as closable workspace tabs", async () => {
    const user = userEvent.setup();

    render(<App />);

    await screen.findByRole("button", { name: "Create draft" });

    await user.click(screen.getByRole("button", { name: "Create draft" }));
    expect(await screen.findByRole("textbox", { name: "Run name" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ppo_allcups_recurrent 2" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Drafts" }));
    await user.click(screen.getByRole("button", { name: /50,000,000 steps/i }));

    expect(await screen.findByRole("textbox", { name: "Run name" })).toHaveValue(
      "ppo_allcups_recurrent",
    );
    expect(screen.getByRole("button", { name: "ppo_allcups_recurrent" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ppo_allcups_recurrent 2" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Drafts" }));
    expect(screen.getByRole("button", { name: "Create draft" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ppo_allcups_recurrent" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ppo_allcups_recurrent 2" })).toBeInTheDocument();
  });
});
