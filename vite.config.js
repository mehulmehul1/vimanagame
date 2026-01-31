import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import mkcert from "vite-plugin-mkcert";

// Plugin to load .wgsl files as raw strings
function wgsl() {
  return {
    name: 'wgsl',
    transform(code, id) {
      if (id.endsWith('.wgsl')) {
        return {
          code: `export default ${JSON.stringify(code)};`,
          map: null,
        };
      }
    },
  };
}

export default defineConfig({
  base: "/",
  plugins: [wasm(), topLevelAwait(), mkcert(), wgsl()],
  server: {
    https: true,
    host: true,
  },
  preview: {
    https: true,
    host: true,
  },
  optimizeDeps: {
    exclude: ["@sparkjsdev/spark"],
  },
  assetsInclude: ["**/*.wasm"],
  worker: {
    format: "es",
  },
  build: {
    minify: false,
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
  resolve: {
    dedupe: ["three"],
  },
});
