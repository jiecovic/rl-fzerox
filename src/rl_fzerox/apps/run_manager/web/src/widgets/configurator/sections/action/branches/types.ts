// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/action/branches/types.ts

import type { ManagedRunConfig } from "@/shared/api/contract";
import type { ActionUpdateContext } from "@/widgets/configurator/sections/action/types";

export type AuxiliaryBranchesDisclosureProps = Omit<ActionUpdateContext, "setConfig"> & {
  checkpointLocked?: boolean;
  open: boolean;
  setOpen: (open: boolean) => void;
};

export type AuxiliaryActionConfig = ManagedRunConfig["action"];

export type UpdateAction = ActionUpdateContext["updateAction"];
export type UpdatePolicy = ActionUpdateContext["updatePolicy"];
export type UpdateTrain = ActionUpdateContext["updateTrain"];
