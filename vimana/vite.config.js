import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import path from 'path';

export default defineConfig({
  base: '/',
  plugins: [wasm(), topLevelAwait()],
  server: {
    port: 5173,
    open: true,
    https: false,
    host: true,
  },
  preview: {
    port: 4173,
  },
  resolve: {
    alias: {
      '@engine': path.resolve(__dirname, '../src'),
    },
    dedupe: ['three'],
  },
  assetsInclude: ['**/*.wasm'],
  worker: {
    format: 'es',
  },
  build: {
    minify: false,
    rollupOptions: {
      output: {
        format: 'es',
      },
    },
  },
});
