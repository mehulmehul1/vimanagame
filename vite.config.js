import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import mkcert from "vite-plugin-mkcert";

export default defineConfig({
  plugins: [wasm(), topLevelAwait(), mkcert()],
  resolve: {
    dedupe: ["three"],
    alias: {
      "three/examples/jsm/": "https://unpkg.com/three@0.178.0/examples/jsm/",
      three: "https://unpkg.com/three@0.178.0/build/three.module.js",
      "@sparkjsdev/spark":
        "https://cdn.jsdelivr.net/gh/sparkjsdev/spark@7c6e7452fd635f955003e3a885718d47b9f7f2cf/dist/spark.module.js",
    },
  },
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
      external: ["three", "@sparkjsdev/spark"],
    },
  },
});
