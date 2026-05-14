// src/rl_fzerox/apps/run_manager/web/src/shared/browser/useDocumentVisible.ts
import { useEffect, useState } from "react";

export function useDocumentVisible(): boolean {
  const [visible, setVisible] = useState(() => document.visibilityState !== "hidden");

  useEffect(() => {
    function syncVisibility() {
      setVisible(document.visibilityState !== "hidden");
    }

    document.addEventListener("visibilitychange", syncVisibility);
    return () => {
      document.removeEventListener("visibilitychange", syncVisibility);
    };
  }, []);

  return visible;
}
