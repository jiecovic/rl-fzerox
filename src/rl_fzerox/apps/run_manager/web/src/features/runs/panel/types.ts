import type { ManagedRun } from "@/shared/api/contract";

export type RunSource =
  | { kind: "root" }
  | {
      kind: "fork";
      artifactLabel: string | null;
      parentName: string;
      stepLabel: string | null;
    };

export type RunLineageRun = {
  childCount: number;
  dependentDraftCount: number;
  depth: number;
  isRoot: boolean;
  source: RunSource;
  run: ManagedRun;
  stageLabel: string;
};

export type RunLineageGroup = {
  canDeleteLineage: boolean;
  createdAt: string;
  id: string;
  label: string;
  latestUpdatedAt: string;
  runs: RunLineageRun[];
};

export type PendingDelete =
  | { kind: "lineage"; lineage: RunLineageGroup }
  | { kind: "run"; run: ManagedRun };
