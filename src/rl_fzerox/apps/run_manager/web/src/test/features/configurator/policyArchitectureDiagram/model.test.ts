// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/policyArchitectureDiagram/model.test.ts
import { describe, expect, it } from "vitest";

import { buildArchitectureGraph } from "@/features/configurator/sections/policyArchitectureDiagram/model";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";
import { policyPreviewFixture } from "@/test/fixtures";

describe("policy architecture graph model", () => {
  it("splits MLP activation details onto their own line", () => {
    const graph = buildArchitectureGraph(policyPreviewFixture);

    expect(graph.visuals.get("cnn")?.detailLines.at(-1)).toBe("nature → 3136");
    expect(graph.visuals.get("state_mlp")?.detailLines.at(-1)).toBe("relu");
    expect(graph.visuals.get("fusion")?.detailLines.at(-1)).toBe("relu");
  });

  it("bypasses the hidden fusion node when fusion is disabled", () => {
    const graph = buildArchitectureGraph(
      previewWithMutedNodes({
        fusion: { detail: "identity 1600", params: 0 },
      }),
    );
    const edgeIds = graph.elkGraph.edges?.map((edge) => edge.id) ?? [];

    expect(graph.visuals.has("fusion")).toBe(false);
    expect(edgeIds).toContain("concat:out-layer_norm");
    expect(edgeIds).toContain("layer_norm-lstm");
    expect(edgeIds).not.toContain("concat:out-fusion");
  });

  it("connects concat directly to heads when all post-concat stages are hidden", () => {
    const graph = buildArchitectureGraph(
      previewWithMutedNodes({
        fusion: { detail: "identity 1600", params: 0 },
        layer_norm: { detail: "off", params: 0 },
        lstm: { detail: "off", params: 0 },
      }),
    );
    const edgeIds = graph.elkGraph.edges?.map((edge) => edge.id) ?? [];

    expect(edgeIds).toContain("concat:out-policy_head");
    expect(edgeIds).toContain("concat:out-value_head");
  });
});

function previewWithMutedNodes(
  muted: Record<string, { detail: string; params: number }>,
): PolicyArchitecturePreview {
  return {
    ...policyPreviewFixture,
    architecture_lanes: policyPreviewFixture.architecture_lanes.map((lane) => ({
      ...lane,
      nodes: lane.nodes.map((node) =>
        node.id in muted
          ? {
              ...node,
              detail: muted[node.id].detail,
              params: muted[node.id].params,
              tone: "muted",
            }
          : node,
      ),
    })),
  };
}
