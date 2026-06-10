// web/run-manager/src/test/widgets/configurator/policyArchitectureDiagram/model.test.ts
import { describe, expect, it } from "vitest";
import { buildArchitectureGraph } from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/model";
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

    expect(edgeIds).toContain("concat:out-heads");
    expect(edgeIds).toContain("heads-policy_head");
    expect(edgeIds).toContain("heads-value_head");
    expect(edgeIds).not.toContain("concat:out-policy_head");
    expect(edgeIds).not.toContain("concat:out-value_head");
  });

  it("connects policy, auxiliary, and value heads through one fork", () => {
    const graph = buildArchitectureGraph(previewWithAuxiliaryHead());
    const edgeIds = graph.elkGraph.edges?.map((edge) => edge.id) ?? [];

    expect(edgeIds).toContain("lstm-heads");
    expect(edgeIds).toContain("heads-policy_head");
    expect(edgeIds).toContain("heads-aux_head");
    expect(edgeIds).toContain("heads-value_head");
    expect(edgeIds).not.toContain("lstm-policy_head");
    expect(edgeIds).not.toContain("lstm-aux_head");
    expect(edgeIds).not.toContain("lstm-value_head");
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

function previewWithAuxiliaryHead(): PolicyArchitecturePreview {
  return {
    ...policyPreviewFixture,
    architecture_lanes: policyPreviewFixture.architecture_lanes.map((lane) => ({
      ...lane,
      nodes: lane.nodes.flatMap((node) =>
        node.id === "policy_head"
          ? [
              node,
              {
                detail: "256 → [128], relu",
                id: "aux_head",
                label: "Aux head",
                params: 37900,
                tone: "normal",
              },
            ]
          : [node],
      ),
    })),
  };
}
