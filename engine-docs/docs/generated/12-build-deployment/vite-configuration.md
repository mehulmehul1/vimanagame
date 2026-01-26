# Vite Configuration - First Principles Guide

## Overview

**Vite** is the build tool and development server for this project. It provides instant hot module replacement (HMR) during development and creates optimized production bundles. Vite uses modern JavaScript modules (ESM) natively in the browser, making development faster and more intuitive than older bundlers like Webpack.

Think of Vite as the **"orchestra conductor"**—like a conductor coordinates all the musicians to play together at the right time, Vite coordinates all your source files, dependencies, and assets to work together in both development and production.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Make the development experience invisible to the creative process. Developers should focus on building the game, not fighting with build tools. Fast iteration means more time for polish and experimentation.

**Why Vite for Game Development?**
- **Instant Start**: No waiting for bundling
- **Live Reload**: See changes instantly while testing gameplay
- **Optimized Builds**: Small, fast downloads for players
- **Asset Handling**: Easy integration of 3D models, textures, audio
- **Modern Standards**: Uses native browser modules

**Developer Experience Flow**:
```
Start Development
    ↓
Vite Server Ready (~1 second)
    ↓
Open Browser → Game loads instantly
    ↓
Make Code Changes
    ↓
Browser Updates Live (~100ms)
    ↓
Test Gameplay Immediately
    ↓
Rapid Iteration → Better Game
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding Vite configuration, you should know:
- **Module systems** - ES modules (import/export) vs CommonJS
- **Build tools** - What bundlers do (combine, optimize, minify)
- **Development vs Production** - Different needs for each
- **Hot Module Replacement** - Updating code without page refresh
- **Plugins** - Extending Vite's functionality

### Core Architecture

```
VITE CONFIGURATION ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                      VITE CONFIG                        │
│  - Entry points                                        │
│  - Build options                                       │
│  - Dev server settings                                 │
│  - Plugins                                             │
│  - Asset handling                                      │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   DEV MODE    │  │  BUILD MODE  │  │   PLUGINS    │
│  - HMR        │  │  - Minify    │  │  - Three.js  │
│  - Source map │  │  - Tree-shake│  │  - WASM      │
│  - Fast refresh│  │  - Code-split│  │  - Assets    │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   OUTPUT     │
                    │  - Bundle    │
                    │  - Assets    │
                    │  - Index.html│
                    └───────────────┘
```

### vite.config.js

```javascript
import { defineConfig } from 'vite';
import path from 'path';

/**
 * Vite configuration for Shadow Engine
 * Optimized for WebGPU + Gaussian Splatting + 3D web games
 */
export default defineConfig(({ mode }) => {
  const isDevelopment = mode === 'development';
  const isProduction = mode === 'production';

  return {
    // Project root
    root: './',

    // Public directory (static assets)
    publicDir: 'public',

    // Entry point
    // If not specified, defaults to index.html in root
    // build.rollupOptions.input would be used for multi-page apps

    // Development server
    server: {
      port: 3000,
      host: true,  // Listen on all addresses
      strictPort: false,  // Try next port if 3000 is taken
      open: false,  // Don't auto-open browser
      cors: true,  // Enable CORS

      // Proxy API requests if needed
      proxy: {
        // Example: Proxy API calls to backend
        // '/api': {
        //   target: 'http://localhost:8080',
        //   changeOrigin: true,
        //   rewrite: (path) => path.replace(/^\/api/, '')
        // }
      },

      // Headers for security and feature detection
      headers: {
        // Required for SharedArrayBuffer (multi-threading)
        'Cross-Origin-Embedder-Policy': isDevelopment ? 'credentialless' : 'require-corp',
        'Cross-Origin-Opener-Policy': isDevelopment ? 'same-origin' : 'same-origin'
      }
    },

    // Build options
    build: {
      // Output directory
      outDir: 'dist',
      emptyOutDir: true,  // Clean outDir before build

      // Generate source maps
      sourcemap: isDevelopment ? 'inline' : false,

      // Minification
      minify: isProduction ? 'terser' : false,
      terserOptions: {
        compress: {
          drop_console: isProduction,  // Remove console.log in production
          drop_debugger: true
        },
        format: {
          comments: false  // Remove comments
        }
      },

      // Target browsers
      target: 'es2020',  // Modern browsers for WebGPU support

      // Module pre-loading (for faster initial load)
      modulePreload: {
        polyfill: false  // Don't polyfill, modern browsers only
      },

      // Rollup options (Vite uses Rollup for bundling)
      rollupOptions: {
        // Manual chunks for better caching
        output: {
          // Manual chunk splitting
          manualChunks: {
            // Vendor chunk: Third-party libraries
            'vendor': [
              'three',
              '@dimforge/rapier3d-compat',
              'howler',
              '@tensorflow/tfjs',
              '@tensorflow/tfjs-backend-webgl'
            ],

            // Engine chunk: Core game engine
            'engine': [
              './src/managers/*.js',
              './src/core/*.js'
            ],

            // Content chunk: Game data and scenes
            'content': [
              './src/content/*.js',
              './src/data/*.js'
            ]
          },

          // Naming pattern for chunks
          chunkFileNames: isDevelopment
            ? 'assets/js/[name]-[hash].js'
            : 'assets/js/[name]-[contenthash:8].js',

          // Naming for entry points
          entryFileNames: isDevelopment
            ? 'assets/js/[name]-[hash].js'
            : 'assets/js/[name]-[contenthash:8].js',

          // Naming for asset outputs
          assetFileNames: isDevelopment
            ? 'assets/[name]-[hash][extname]'
            : 'assets/[name]-[contenthash:8][extname]'
        },

        // External dependencies (don't bundle)
        external: isDevelopment ? [] : [
          // Don't externalize in production, we want everything bundled
        ],

        // Preserve symlinks (for monorepo setups)
        preserveSymlinks: false,

        // Preserve entry module signatures (for better tree-shaking)
        preserveEntrySignatures: 'strict'
      },

      // Chunk size warning limit (KB)
      chunkSizeWarningLimit: 1000,

      // CSS code splitting
      cssCodeSplit: true,

      // CSS minification
      cssMinify: isProduction,

      // Enable asset optimization
      assetsInlineLimit: 4096,  // Inline assets smaller than 4KB

      // Set maximum concurrent requests for build
      maxParallelFileOps: 8,

      // Report compressed size
      reportCompressedSize: true,

      // Write build manifest
      manifest: isProduction,

      // Build analysis (for debugging bundle size)
      // analyze: true  // Uncomment to analyze bundle size
    },

    // Dependencies to optimize
    optimizeDeps: {
      include: [
        'three',
        '@dimforge/rapier3d-compat',
        'howler',
        '@tensorflow/tfjs',
        '@tensorflow/tfjs-backend-webgl'
      ],

      // Exclude certain dependencies from optimization
      exclude: [],

      // Use esbuild for faster pre-bundling
      esbuildOptions: {
        target: 'es2020',
        // Preserve class names for better debugging
        keepNames: isDevelopment
      }
    },

    // CSS configuration
    css: {
      modules: {
        // CSS Modules configuration (if using)
        localsConvention: 'camelCase'
      },

      // Preprocessor options (if using SCSS/Less)
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`
        }
      },

      // Dev sourcemaps for CSS
      devSourcemap: isDevelopment
    },

    // JSON configuration
    json: {
      // Named exports for JSON imports
      namedExports: true,

      // Stringify imported JSON
      stringify: isDevelopment
    },

    // ESBuild options (used for minification and transpilation)
    esbuild: {
      // Target environment
      target: 'es2020',

      // Preserve JSX comments if needed
      jsxPreserve: false,

      // Define global constants
      define: {
        __DEV__: isDevelopment,
        __PROD__: isProduction
      },

      // Drop console in production (terser handles this, but esbuild can too)
      drop: isProduction ? ['console', 'debugger'] : [],

      // Tree-shaking
      treeShaking: true,

      // Keep names for better debugging in development
      keepNames: isDevelopment
    },

    // Preview server (for testing production build)
    preview: {
      port: 4173,
      host: true,
      strictPort: false,
      open: false,

      // Headers for preview
      headers: {
        'Cross-Origin-Embedder-Policy': 'credentialless',
        'Cross-Origin-Opener-Policy': 'same-origin'
      }
    },

    // Path aliases (for imports)
    resolve: {
      alias: {
        // Map @ to src directory
        '@': path.resolve(__dirname, './src'),

        // Map common paths
        '@core': path.resolve(__dirname, './src/core'),
        '@managers': path.resolve(__dirname, './src/managers'),
        '@content': path.resolve(__dirname, './src/content'),
        '@util': path.resolve(__dirname, './src/util'),
        '@data': path.resolve(__dirname, './src/data')
      },

      // File extensions to try when resolving
      extensions: ['.js', '.json', '.wasm', '.glb', '.gltf', '.splat']
    },

    // Assets handling
    assetsInclude: [
      '**/*.glb',
      '**/*.gltf',
      '**/*.splat',
      '**/*.bin',
      '**/*.hdr',
      '**/*.exr',
      '**/*.ktx2',
      '**/*.webp',
      '**/*.woff2',
      '**/*.wasm'
    ],

    // Worker handling (for multi-threading)
    worker: {
      format: 'es',
      plugins: []
    },

    // Experimental features
    experimental: {
      // Enable build-time rendering (if using SSG)
      renderBuiltUrl(filename, { hostType }) {
        return { relative: true };
      }
    },

    // Plugins
    plugins: [
      // Custom plugins for game-specific handling
      // Example: asset optimization plugin
      // Example: WASM loading plugin
    ]
  };
});
```

### Package.json Scripts

```json
{
  "name": "shadow-engine",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "dev:debug": "VITE_DEBUG=true vite --debug",
    "build": "vite build",
    "build:analyze": "vite build --mode analyze",
    "preview": "vite preview",
    "serve": "vite preview --port 4173",
    "clean": "rimraf dist node_modules/.vite",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "three": "^0.160.0",
    "@dimforge/rapier3d-compat": "^0.12.0",
    "howler": "^2.2.4",
    "@tensorflow/tfjs": "^4.15.0",
    "@tensorflow/tfjs-backend-webgl": "^4.15.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "terser": "^5.26.0",
    "rimraf": "^5.0.5",
    "@types/three": "^0.160.0",
    "typescript": "^5.3.3"
  }
}
```

### Environment Variables

Create `.env` files for environment-specific configuration:

```bash
# .env (shared)
VITE_APP_TITLE=Shadow Engine
VITE_APP_VERSION=1.0.0
VITE_API_BASE_URL=/api

# .env.development
VITE_DEBUG=true
VITE_LOG_LEVEL=debug
VITE_ENABLE_PROFILING=true
VITE_PERFORMANCE_MONITORING=true

# .env.production
VITE_DEBUG=false
VITE_LOG_LEVEL=warn
VITE_ENABLE_PROFILING=false
VITE_PERFORMANCE_MONITORING=false

# .env.local (not committed, for local overrides)
VITE_DEV_TOKEN=your-dev-token-here
```

---

## 📝 Configuration Options Explained

### Server Options

| Option | Purpose | Recommended Value |
|--------|---------|-------------------|
| port | Development server port | 3000 |
| host | Network accessibility | true (LAN testing) |
| strictPort | Fail if port in use | false (find open port) |
| cors | Cross-origin requests | true |
| open | Auto-open browser | false (manual control) |

### Build Options

| Option | Purpose | Recommended Value |
|--------|---------|-------------------|
| outDir | Build output | 'dist' |
| sourcemap | Debug maps | dev: inline, prod: false |
| minify | Code minification | terser |
| target | Browser targets | es2020 (WebGPU) |
| chunkSizeWarningLimit | Warning threshold | 1000 KB |
| cssCodeSplit | Separate CSS chunks | true |

### Rollup Output Options

| Option | Purpose | Recommended Value |
|--------|---------|-------------------|
| manualChunks | Code splitting | { vendor: [...], engine: [...] } |
| chunkFileNames | Chunk naming | [name]-[contenthash:8].js |
| entryFileNames | Entry naming | [name]-[contenthash:8].js |
| assetFileNames | Asset naming | [name]-[contenthash:8][extname] |

---

## Common Mistakes Beginners Make

### 1. Not Using Path Aliases

```javascript
// ❌ WRONG: Relative import hell
import GameManager from '../../../../managers/GameManager.js';
// Hard to maintain, breaks on move

// ✅ CORRECT: Use aliases
import GameManager from '@managers/GameManager.js';
// Clean, portable
```

### 2. Bundling Everything Together

```javascript
// ❌ WRONG: Single giant bundle
manualChunks: {}
// Slow initial load, no caching benefits

// ✅ CORRECT: Split by usage frequency
manualChunks: {
  vendor: ['three', 'howler'],  // Changes rarely
  engine: ['./src/managers/*.js']  // Changes often
}
// Better caching, faster updates
```

### 3. Source Maps in Production

```javascript
// ❌ WRONG: Source maps in production
sourcemap: true
// Exposes source code, larger downloads

// ✅ CORRECT: Only in development
sourcemap: mode === 'development' ? 'inline' : false
// Production is clean and fast
```

---

## Performance Optimization

### Code Splitting Strategy

```
SPLIT STRATEGY:

Vendor Chunk (3MB)
├── Three.js (rarely changes)
├── Rapier physics (rarely changes)
├── Howler audio (rarely changes)
└── TensorFlow.js (rarely changes)
    ↓ Cached for months

Engine Chunk (500KB)
├── Managers (occasional updates)
├── Core systems (occasional updates)
└── Utilities (occasional updates)
    ↓ Re-downloaded on updates

Content Chunk (varies)
├── Scene data (frequent updates)
├── Dialog data (frequent updates)
└── Game assets (frequent updates)
    ↓ Re-downloaded frequently
```

### Build Analysis

To analyze bundle size:

```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Analyze dependencies
          console.log('Module:', id);
        }
      }
    }
  }
});
```

Then run:
```bash
npm run build
# Check output for module sizes
```

---

## Related Systems

- [Development Setup](./development-setup.md) - Local environment
- [Build Process](./build-process.md) - Creating production builds
- [WASM Handling](./wasm-handling.md) - WebAssembly integration

---

## Source File Reference

**Primary Files**:
- `vite.config.js` - Main Vite configuration (project root)
- `package.json` - Dependencies and scripts
- `.env*` - Environment variables

**Key Configuration Areas**:
- Server settings (development)
- Build settings (production)
- Plugin configuration
- Path resolution

---

## References

- [Vite Documentation](https://vitejs.dev/) - Official docs
- [Vite Config Reference](https://vitejs.dev/config/) - All options
- [Rollup Options](https://vitejs.dev/config/build-options#build-rollupoptions) - Bundler settings

*Documentation last updated: January 12, 2026*
