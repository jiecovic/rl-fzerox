// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/auxiliary_branches/types.ts
import type { ActionUpdateContext } from "@/features/configurator/sections/action/types";
import type { ManagedRunConfig } from "@/shared/api/contract";

export type AuxiliaryBranchesDisclosureProps = Omit<ActionUpdateContext, "setConfig"> & {
  checkpointLocked?: boolean;
  open: boolean;
  setOpen: (open: boolean) => void;
};

export type AuxiliaryActionConfig = ManagedRunConfig["action"];

export type UpdateAction = ActionUpdateContext["updateAction"];
export type UpdatePolicy = ActionUpdateContext["updatePolicy"];
export type UpdateTrain = ActionUpdateContext["updateTrain"];
