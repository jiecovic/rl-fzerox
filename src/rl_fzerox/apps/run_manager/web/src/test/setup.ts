// src/rl_fzerox/apps/run_manager/web/src/test/setup.ts
import "@testing-library/jest-dom/vitest";
import { beforeEach } from "vitest";

const testLocalStorage = createMemoryStorage();

Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: testLocalStorage,
});
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: testLocalStorage,
});

beforeEach(() => {
  testLocalStorage.clear();
});

function createMemoryStorage(): Storage {
  const entries = new Map<string, string>();
  return {
    get length() {
      return entries.size;
    },
    clear() {
      entries.clear();
    },
    getItem(key: string) {
      return entries.get(key) ?? null;
    },
    key(index: number) {
      return Array.from(entries.keys())[index] ?? null;
    },
    removeItem(key: string) {
      entries.delete(key);
    },
    setItem(key: string, value: string) {
      entries.set(key, value);
    },
  };
}
