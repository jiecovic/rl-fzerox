// web/run-manager/vite.config.ts
import { fileURLToPath, URL } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { createLogger } from "vite";
import { defineConfig } from "vitest/config";

const logger = createLogger();
const defaultError = logger.error;
logger.error = (message, options) => {
  if (isExpectedWebSocketProxyReset(message)) {
    return;
  }
  defaultError(message, options);
};

export default defineConfig({
  customLogger: logger,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8765",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    environmentOptions: {
      jsdom: {
        url: "http://localhost/",
      },
    },
    setupFiles: "./src/test/setup.ts",
  },
});

function isExpectedWebSocketProxyReset(message: string) {
  return (
    (message.includes("ws proxy error:") || message.includes("ws proxy socket error:")) &&
    (message.includes("ECONNRESET") || message.includes("EPIPE"))
  );
}
