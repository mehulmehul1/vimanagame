import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import topLevelAwait from 'vite-plugin-top-level-await';
import wasm from 'vite-plugin-wasm';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    wasm(),
    topLevelAwait({
      // The Promise name that will be used in the code
      promiseExportName: 'tla',
      // The maximum number of dynamic imports that can be awaited
      promiseImportName: 'tlaImport'
    })
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './'),
      '@/core': resolve(__dirname, './core'),
      '@/panels': resolve(__dirname, './panels'),
      '@/components': resolve(__dirname, './components'),
      '@/data': resolve(__dirname, './data'),
      '@/utils': resolve(__dirname, './utils'),
      '@/styles': resolve(__dirname, './styles'),
      // Alias to parent engine directory
      '@engine': resolve(__dirname, '../packages/engine/src'),
      // Alias to game assets in vimana/public
      '/game-assets': resolve(__dirname, '../vimana/public/assets')
    },
    dedupe: ['three']
  },
  server: {
    port: 3001,
    strictPort: false,
    host: true,
    open: true,
    headers: {
      'Access-Control-Allow-Origin': '*'
    },
    // Allow serving files from parent directory
    fs: {
      strict: false,
      allow: [
        // Allow accessing parent directory files for game assets and engine node_modules
        resolve(__dirname, '..'),
        // Explicitly allow vimana assets
        resolve(__dirname, '../vimana/public')
      ]
    }
  },
  publicDir: resolve(__dirname, '../public'),
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'pcui-vendor': ['@playcanvas/pcui', '@playcanvas/observer'],
          'three-vendor': ['three']
        }
      }
    }
  },
  assetsInclude: ['**/*.wasm'],
  optimizeDeps: {
    include: ['react', 'react-dom', '@playcanvas/pcui', '@playcanvas/observer', 'three'],
    exclude: ['@sparkjsdev/spark']
  }
});
