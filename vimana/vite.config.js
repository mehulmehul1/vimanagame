import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import path from 'path';

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
  base: '/',
  plugins: [wasm(), topLevelAwait(), wgsl()],
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
