# Build Process - First Principles Guide

## Overview

The **Build Process** transforms your development source code into an optimized, deployable production bundle. This includes bundling JavaScript, minifying code, optimizing assets, and creating a distributable package that players can download and run. A good build process balances small file sizes (fast downloads) with runtime performance (smooth gameplay).

Think of the build process like the **"packaging department"**—like how products are packaged for shipping with protective materials and efficient organization, the build process prepares your game for distribution to players.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Ensure players can start playing quickly without long loading screens. The build process creates lean, efficient downloads that get players into the game faster while maintaining all the visual quality and gameplay features you created.

**Why Build Optimization Matters?**
- **First Impressions**: Fast load = happy players
- **Mobile Players**: Limited data plans need efficiency
- **Retention**: Long loads = player drop-off
- **SEO**: Faster sites rank better
- **Server Costs**: Smaller files = less bandwidth

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the build process, you should know:
- **Bundling** - Combining modules into fewer files
- **Minification** - Removing whitespace and renaming variables
- **Tree-shaking** - Eliminating unused code
- **Code splitting** - Dividing code into chunks
- **Asset optimization** - Compressing images, audio, models

### Build Stages

```
BUILD PROCESS STAGES:

┌─────────────────────────────────────────────────────────┐
│  1. RESOLUTION         │
│  - Import paths → actual files                        │
│  - Alias resolution                                   │
│  - Environment variables                               │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│  2. TRANSFORM    │               │  3. OPTIMIZE    │
│  - TypeScript     │               │  - Minify       │
│  → JavaScript     │               │  - Tree-shake   │
│  - JSX/TSX        │               │  - Dead code    │
│  → JS calls       │               │    elimination  │
│  - CSS preprocess │               │  - Compress     │
└───────────────────┘               └───────────────────┘
                                            │
        ┌───────────────────────────────┴───────────┐
        ↓                                           ↓
┌───────────────────┐               ┌───────────────────┐
│  4. BUNDLE       │               │  5. OUTPUT      │
│  - Module chunks  │               │  - File writing │
│  - Dependencies   │               │  - Hash naming  │
│  - Entry points  │               │  - Manifest     │
└───────────────────┘               └───────────────────┘
```

### Build Commands

```bash
# Development build (fast, unoptimized)
npm run dev

# Production build (optimized, minified)
npm run build

# Preview production build locally
npm run preview

# Clean build artifacts
npm run clean
```

### Build Output Structure

After running `npm run build`, the `dist/` directory contains:

```
dist/
├── index.html                  # Entry HTML (processed)
├── assets/
│   ├── js/
│   │   ├── vendor-[hash].js         # Third-party libraries
│   │   ├── engine-[hash].js         # Core engine
│   │   ├── content-[hash].js        # Game content
│   │   └── main-[hash].js           # Entry point
│   ├── models/
│   │   ├── character-[hash].glb     # 3D models
│   │   └── environment-[hash].glb
│   ├── textures/
│   │   ├── diffuse-[hash].png        # Textures
│   │   └── normal-[hash].png
│   ├── audio/
│   │   ├── music-[hash].mp3          # Audio files
│   │   └── sfx-[hash].mp3
│   └── splats/
│       └── scene-[hash].splat        # Gaussian splats
└── manifest.json               # Build manifest (optional)
```

### Build Configuration

The production build is configured in `vite.config.js`:

```javascript
build: {
  // Output directory
  outDir: 'dist',

  // Clean before build
  emptyOutDir: true,

  // Source maps (disabled for production)
  sourcemap: false,

  // Minification
  minify: 'terser',

  // Target modern browsers
  target: 'es2020',

  // Code splitting
  rollupOptions: {
    output: {
      manualChunks: {
        vendor: ['three', 'howler', '@dimforge/rapier3d-compat'],
        engine: ['./src/managers/*.js'],
        content: ['./src/content/*.js']
      }
    }
  },

  // Asset optimization
  assetsInlineLimit: 4096,

  // Chunk size warnings
  chunkSizeWarningLimit: 1000,

  // Report compressed size
  reportCompressedSize: true
}
```

---

## Build Optimization Techniques

### 1. Code Splitting

Divide code into chunks that can be loaded independently:

```javascript
// Manual chunks by category
manualChunks: {
  // Vendor: Rarely changes, cached longest
  vendor: ['three', 'howler', '@tensorflow/tfjs'],

  // Engine: Changes occasionally, cached medium
  engine: ['./src/managers/GameManager.js',
           './src/managers/SceneManager.js'],

  // Content: Changes frequently, cached shortest
  content: ['./src/content/SceneData.js',
            './src/content/DialogData.js']
}
```

### 2. Tree Shaking

Remove unused code:

```javascript
// Only import what you need
import { Vector3 } from 'three';  // ✅ Good
import * as THREE from 'three';    // ❌ May include unused code

// Use ES modules for better tree-shaking
export function myFunction() { /* ... */ }  // ✅ Can be shaken if unused
```

### 3. Asset Optimization

Optimize assets during build:

```javascript
// Inline small assets
assetsInlineLimit: 4096,  // 4KB or less = inline

// Hash-based filenames for cache busting
assetFileNames: 'assets/[name]-[contenthash:8][extname]'
```

### 4. Compression

Enable text compression:

```javascript
// In vite.config.js or server config
build: {
  // Enable gzip compression in server
  // Files are pre-compressed for delivery
}
```

---

## Build Analysis

### Analyze Bundle Size

```bash
# Install rollup-plugin-visualizer
npm install --save-dev rollup-plugin-visualizer

# Add to vite.config.js
import { visualizer } from 'rollup-plugin-visualizer';

export default {
  plugins: [
    visualizer({ open: true, gzipSize: true })
  ]
}

# Build and visualize
npm run build
```

### Bundle Size Guidelines

```
BUNDLE SIZE TARGETS:

Initial Load (Critical)
├── HTML: < 10KB
├── CSS: < 50KB
├── Main JS: < 200KB
├── Vendor JS: < 500KB (cached)
└── Total: < 750KB

Lazy Loaded (Per zone)
├── Zone data: < 100KB
├── Models: < 500KB
├── Textures: < 1MB
└── Per zone: < 2MB

Total Game Size
├── All code: < 5MB
├── All assets: < 50MB
└── Complete download: < 55MB
```

---

## Production Deployment

### Build for Production

```bash
# 1. Clean previous build
npm run clean

# 2. Run production build
npm run build

# 3. Test production build locally
npm run preview

# 4. Verify in browser
# Open http://localhost:4173
```

### Deployment Options

#### Static Hosting (Netlify, Vercel, GitHub Pages)

```bash
# Build
npm run build

# Deploy dist/ directory
# Most hosts have CLI tools or Git integration
```

#### CDN Deployment

```bash
# Upload to CDN
# Configure cache headers
# - HTML: no-cache
# - JS/CSS: long cache with hash
# - Assets: long cache with hash
```

#### Self-Hosted

```bash
# Copy dist/ to web server
scp -r dist/* user@server:/var/www/html/

# Or use rsync
rsync -avz dist/ user@server:/var/www/html/
```

---

## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/build.yml`:

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.npm
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-node-

      - name: Install dependencies
        run: npm ci

      - name: Run type check
        run: npm run type-check

      - name: Build
        run: npm run build

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/

      - name: Deploy to Netlify
        if: github.ref == 'refs/heads/main'
        uses: nwtgck/actions-netlify@v2.0
        with:
          publish-dir: './dist'
          production-branch: main
          github-token: ${{ secrets.GITHUB_TOKEN }}
          deploy-key: ${{ secrets.NETLIFY_DEPLOY_KEY }}
```

---

## Build Troubleshooting

### Common Build Errors

#### Module Resolution Failed

```
Error: Could not resolve import
```

**Solution**: Check import paths and aliases in `vite.config.js`

```
import GameManager from '@/managers/GameManager.js';  // Use alias
```

#### Build Exceeds Memory Limit

```
FATAL ERROR: Reached heap limit
```

**Solution**: Increase Node.js memory

```bash
NODE_OPTIONS="--max-old-space-size=8192" npm run build
```

#### WASM Import Error

```
Error: WebAssembly compilation failed
```

**Solution**: Ensure `.wasm` files are included in assets

```javascript
// vite.config.js
assetsInclude: ['**/*.wasm']
```

---

## Build Best Practices

### 1. Always Test Production Build

```bash
# Before releasing, always test
npm run build
npm run preview
# Test thoroughly in preview mode
```

### 2. Monitor Build Size

```bash
# Check build output for warnings
npm run build
# Look for "(!) Some chunks are larger than X"
```

### 3. Use Environment Variables

```javascript
// Check environment in code
if (import.meta.env.PROD) {
  // Production-only code
} else {
  // Development-only code
}
```

### 4. Clean Before Build

```bash
# Ensure clean build
npm run clean
npm run build
```

---

## Related Systems

- [Vite Configuration](./vite-configuration.md) - Build tool setup
- [Development Setup](./development-setup.md) - Local environment
- [WASM Handling](./wasm-handling.md) - WebAssembly in builds

---

## Source File Reference

**Build Files**:
- `vite.config.js` - Main build configuration
- `package.json` - Build scripts
- `dist/` - Build output directory

**Build Scripts**:
- `npm run build` - Production build
- `npm run preview` - Test production build
- `npm run clean` - Clean artifacts

---

## References

- [Vite Build Mode](https://vitejs.dev/guide/build.html) - Production builds
- [Rollup Options](https://vitejs.dev/config/build-options) - Build configuration
- [Tree Shaking](https://rollupjs.org/introduction/#tree-shaking) - Dead code elimination

*Documentation last updated: January 12, 2026*
