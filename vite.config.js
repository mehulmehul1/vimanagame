import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import mkcert from "vite-plugin-mkcert";

export default defineConfig({
  plugins: [wasm(), topLevelAwait(), mkcert()],
  server: {
    https: true,
    host: true,
  },
  preview: {
    https: true,
    host: true,
  },
  build: {
    rollupOptions: {
      output: {
        format: "es",
        manualChunks: (id) => {
          // Force VFX modules into main bundle
          if (id.includes("/vfx/")) {
            return "index";
          }
        },
      },
    },
  },
});
