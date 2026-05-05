import type { ActionUpdateContext } from "@/features/configurator/sections/action/types";
import type { ManagedRunConfig } from "@/shared/api/contract";

export type AuxiliaryBranchesDisclosureProps = Omit<
  ActionUpdateContext,
  "updatePolicy" | "setConfig"
> & {
  checkpointLocked?: boolean;
  open: boolean;
  setOpen: (open: boolean) => void;
};

export type AuxiliaryActionConfig = ManagedRunConfig["action"];

export type UpdateAction = ActionUpdateContext["updateAction"];
