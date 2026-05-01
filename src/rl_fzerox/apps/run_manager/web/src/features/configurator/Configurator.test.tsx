import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Configurator } from "@/features/configurator/Configurator";
import {
  configMetadataFixture,
  draftFixture,
  managedRunConfigFixture,
  policyPreviewFixture,
} from "@/test/fixtures";

const fetchPolicyPreviewMock = vi.fn();

vi.mock("@/shared/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/client")>("@/shared/api/client");
  return {
    ...actual,
    fetchPolicyPreview: (...args: Parameters<typeof actual.fetchPolicyPreview>) =>
      fetchPolicyPreviewMock(...args),
  };
});

describe("Configurator", () => {
  beforeEach(() => {
    fetchPolicyPreviewMock.mockReset();
    fetchPolicyPreviewMock.mockResolvedValue(policyPreviewFixture);
  });

  it("loads an opened draft into the configurator and updates it in place", async () => {
    const user = userEvent.setup();
    const onSaveDraft = vi.fn();
    const onUpdateDraft = vi.fn().mockResolvedValue(draftFixture());
    const loadedDraft = draftFixture({ name: "ppo_allcups_recurrent" });

    render(
      <Configurator
        baseConfig={managedRunConfigFixture}
        loadedDraft={loadedDraft}
        metadata={configMetadataFixture}
        onSaveDraft={onSaveDraft}
        onUpdateDraft={onUpdateDraft}
      />,
    );

    const runName = screen.getByLabelText("Run name");
    await waitFor(() => expect(runName).toHaveValue("ppo_allcups_recurrent"));

    await user.clear(runName);
    await user.type(runName, "ppo_allcups_recurrent_v2");
    await user.click(screen.getByRole("button", { name: "Update draft" }));

    await waitFor(() =>
      expect(onUpdateDraft).toHaveBeenCalledWith(
        "draft-001",
        "ppo_allcups_recurrent_v2",
        managedRunConfigFixture,
      ),
    );
    expect(onSaveDraft).not.toHaveBeenCalled();
  });
});
