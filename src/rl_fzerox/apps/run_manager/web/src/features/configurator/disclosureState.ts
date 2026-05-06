// src/rl_fzerox/apps/run_manager/web/src/features/configurator/disclosureState.ts
import { useEffect, useState } from "react";

export function usePersistentDisclosureMap<T extends Record<string, boolean>>(
  storageKey: string,
  defaults: T,
) {
  const [state, setState] = useState<T>(() => readDisclosureMap(storageKey, defaults));

  useEffect(() => {
    setState((current) => {
      const next = mergeDisclosureMap(defaults, current);
      return disclosureMapsEqual(current, next) ? current : next;
    });
  }, [defaults]);

  useEffect(() => {
    window.localStorage.setItem(storageKey, JSON.stringify(state));
  }, [state, storageKey]);

  return [state, setState] as const;
}

export function usePersistentCollapsedIds(
  storageKey: string,
  defaultCollapsedIds: readonly string[],
) {
  const [state, setState] = useState<readonly string[]>(() =>
    readCollapsedIds(storageKey, defaultCollapsedIds),
  );

  useEffect(() => {
    setState((current) => {
      const currentSet = new Set(current);
      const validIds = new Set(defaultCollapsedIds);
      const next = defaultCollapsedIds.filter((id) => currentSet.has(id) && validIds.has(id));
      return stringArraysEqual(current, next) ? current : next;
    });
  }, [defaultCollapsedIds]);

  useEffect(() => {
    window.localStorage.setItem(storageKey, JSON.stringify(state));
  }, [state, storageKey]);

  return [state, setState] as const;
}

function readDisclosureMap<T extends Record<string, boolean>>(storageKey: string, defaults: T): T {
  if (typeof window === "undefined") {
    return defaults;
  }
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw === null) {
      return defaults;
    }
    const parsed = JSON.parse(raw);
    return mergeDisclosureMap(defaults, parsed);
  } catch {
    return defaults;
  }
}

function mergeDisclosureMap<T extends Record<string, boolean>>(defaults: T, raw: unknown): T {
  if (typeof raw !== "object" || raw === null) {
    return defaults;
  }
  const candidate = raw as Partial<Record<keyof T, unknown>>;
  const merged = { ...defaults };
  for (const key of Object.keys(defaults) as Array<keyof T>) {
    if (typeof candidate[key] === "boolean") {
      merged[key] = candidate[key] as T[keyof T];
    }
  }
  return merged;
}

function readCollapsedIds(storageKey: string, defaultCollapsedIds: readonly string[]) {
  if (typeof window === "undefined") {
    return defaultCollapsedIds;
  }
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw === null) {
      return defaultCollapsedIds;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return defaultCollapsedIds;
    }
    return parsed.filter((value): value is string => typeof value === "string");
  } catch {
    return defaultCollapsedIds;
  }
}

function disclosureMapsEqual<T extends Record<string, boolean>>(left: T, right: T) {
  return (Object.keys(left) as Array<keyof T>).every((key) => left[key] === right[key]);
}

function stringArraysEqual(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
