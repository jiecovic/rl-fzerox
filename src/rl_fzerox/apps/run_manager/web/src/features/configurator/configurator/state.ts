// src/rl_fzerox/apps/run_manager/web/src/features/configurator/configurator/state.ts
import type { Dispatch, SetStateAction } from "react";
import type { ManagedRunConfig } from "@/shared/api/contract";

export type ConfigSetter = Dispatch<SetStateAction<ManagedRunConfig>>;
type ConfigObjectSectionKey = {
  [K in keyof ManagedRunConfig]: ManagedRunConfig[K] extends object ? K : never;
}[keyof ManagedRunConfig];
export type ConfigSectionPatch<K extends ConfigObjectSectionKey> =
  | Partial<ManagedRunConfig[K]>
  | ((config: ManagedRunConfig) => Partial<ManagedRunConfig[K]>);

export function patchConfigSection<K extends ConfigObjectSectionKey>(
  setConfig: ConfigSetter,
  section: K,
  patch: ConfigSectionPatch<K>,
) {
  setConfig((currentConfig) => {
    const sectionPatch = typeof patch === "function" ? patch(currentConfig) : patch;
    return {
      ...currentConfig,
      [section]: {
        ...currentConfig[section],
        ...sectionPatch,
      },
    };
  });
}
