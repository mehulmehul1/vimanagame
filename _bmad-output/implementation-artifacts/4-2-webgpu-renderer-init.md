# Story 4.2: WebGPU Renderer Initialization

Status: ready-for-dev

**Note:** Updated for Visionary single-canvas architecture (2026-01-27 course correction)

## Story

As a rendering engineer,
I want to create WebGPU renderer initialization with feature detection and async compilation,
so that the game can use WebGPU's advanced features (TSL, compute shaders, Visionary) while gracefully falling back to WebGL2 for unsupported browsers.

## Acceptance Criteria

1. `createWebGPURenderer()` function created in `src/rendering/` directory
2. `WebGPU.isAvailable()` check before initialization (using Three.js utility)
3. Graceful fallback to WebGL2 if WebGPU unavailable (Visionary splats disabled in fallback)
4. Async shader compilation via `renderer.compileAsync()`
5. Color management: `outputColorSpace = SRGBColorSpace`
6. Mobile pixel ratio limits (1.5 for mobile, 2 for desktop)
7. `setAnimationLoop()` pattern used (not `requestAnimationFrame`)
8. Tone mapping configured (ACESFilmicToneMapping, exposure 0.5)
9. Visionary compatibility check (Ubuntu not supported, macOS limited)

## Tasks / Subtasks

- [ ] Create createWebGPURenderer module (AC: 1)
  - [ ] Create `src/rendering/createWebGPURenderer.ts`
  - [ ] Export async function with proper TypeScript types
  - [ ] Define WebGPURendererParameters interface
- [ ] Implement WebGPU feature detection (AC: 2, 3)
  - [ ] Check `navigator.gpu` availability
  - [ ] Request adapter with fallback handling
  - [ ] Return WebGL2 renderer if WebGPU unavailable
  - [ ] Log renderer type for debugging
- [ ] Implement platform detection for Visionary (AC: 9)
  - [ ] Detect Ubuntu (Visionary NOT supported)
  - [ ] Detect macOS (limited performance)
  - [ ] Log platform warnings
  - [ ] Disable Visionary on unsupported platforms
- [ ] Configure renderer properties (AC: 5, 8)
  - [ ] Set outputColorSpace to SRGBColorSpace
  - [ ] Configure tone mapping (ACESFilmicToneMapping)
  - [ ] Set tone mapping exposure (0.5)
  - [ ] Enable alpha for transparency support
- [ ] Implement async shader compilation (AC: 4)
  - [ ] Call `renderer.compileAsync()` after scene setup
  - [ ] Handle compilation errors gracefully
  - [ ] Show loading indicator during compilation
  - [ ] Hide loading when complete
- [ ] Implement pixel ratio limits (AC: 6)
  - [ ] Detect mobile platform
  - [ ] Cap pixel ratio at 1.5 for mobile
  - [ ] Cap pixel ratio at 2.0 for desktop
  - [ ] Use window.devicePixelRatio as base
- [ ] Set up animation loop (AC: 7)
  - [ ] Use `renderer.setAnimationLoop()` instead of requestAnimationFrame
  - [ ] Pass callback function for frame updates
  - [ ] Ensure consistent time delta handling

## Dev Notes

### Visionary Architecture Context (Updated 2026-01-27)

**Course Correction:**
- ❌ **REJECTED:** Dual-canvas architecture (WebGPU + WebGL2 overlay)
- ✅ **ADOPTED:** Visionary single-canvas architecture with proper depth occlusion

**Key Changes from Original:**
- No dual-canvas overlay needed
- Visionary uses same WebGPU context as main renderer
- Visionary's `GaussianThreeJSRenderer` handles splats internally
- Proper depth occlusion: splats behind walls are hidden

### Existing Architecture Context

**Current Renderer Detection (`src/core/renderer.js`):**
```javascript
// Current implementation has feature detection already
export async function isWebGPUSupported() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}
```

**Current Create Pattern:**
```javascript
// src/core/renderer.js lines 57-110
export async function createOptimalRenderer(options, canvas, constraints) {
  const { requiresWebGL = false } = constraints;
  const supportsWebGPU = !requiresWebGL && await isWebGPUSupported();
  // ... creates WebGPU or WebGL2 renderer
}
```

**Integration in main.js:**
```javascript
// OLD (before Visionary):
// src/main.js lines 100-104
this.renderer = await createOptimalRenderer(
  { alpha: true, antialias: false },
  null,
  { requiresWebGL: true } // Forces WebGL2 for SparkRenderer
);

// NEW (with Visionary):
this.renderer = await createWebGPURenderer(canvas, {
  alpha: true,
  antialias: false
});

// Initialize Visionary after renderer
this.splatRenderer = new VisionarySplatRenderer();
await this.splatRenderer.initialize(this.renderer, this.scene);
```

### New File Structure

**Create new file:**
```
src/rendering/createWebGPURenderer.ts
```

**This differs from existing `src/core/renderer.js`:**
- Existing file handles WebGL2 fallback for single-canvas setup
- New file focuses on WebGPU-first approach for Visionary
- Both can coexist during transition period

### WebGPURenderer Import Pattern

```typescript
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';

// WebGPURenderer is in the 'three/webgpu' import path
// This is different from the main 'three' import
```

### Async Initialization Pattern

WebGPU requires async initialization unlike WebGL2:

```typescript
// CORRECT - WebGPU requires init()
const renderer = new WebGPURenderer(options);
await renderer.init(); // REQUIRED - async initialization

// INCORRECT - will fail silently
const renderer = new WebGPURenderer(options);
renderer.init(); // Missing await - won't work
```

### Shader Compilation

```typescript
// Compile shaders asynchronously to prevent frame drops
await renderer.compileAsync(scene);

// Call this after scene is set up but before render loop
// This compiles all WGSL shaders ahead of time
```

### Platform Detection for Visionary

**IMPORTANT - Visionary Platform Limitations:**

| Platform | Visionary Support | Notes |
|----------|-------------------|-------|
| Windows 10/11 | ✅ YES | Discrete GPU recommended |
| Ubuntu/Linux | ❌ NO | fp16 WebGPU bug |
| macOS | ⚠️ LIMITED | M4 Max+ recommended |
| Mobile | ⚠️ LIMITED | Performance varies |

```typescript
// Platform detection in createWebGPURenderer
const isUbuntu = /Ubuntu/i.test(navigator.userAgent);
const isMacOS = /Mac|iPod|iPhone|iPad/i.test(navigator.platform);
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

if (isUbuntu) {
  console.warn('Visionary is NOT supported on Ubuntu');
  console.warn('Gaussian splats will be disabled');
  // Continue with WebGPU, but no splats
}

if (isMacOS) {
  console.warn('Visionary performance on macOS may be limited');
  console.warn('M4 Max+ chip recommended for optimal performance');
}
```

### Mobile Detection

Use existing pattern from codebase:

```typescript
// Similar to existing platform detection
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const pixelRatioCap = isMobile ? 1.5 : 2.0;
const pixelRatio = Math.min(window.devicePixelRatio, pixelRatioCap);
renderer.setPixelRatio(pixelRatio);
```

### setAnimationLoop Pattern

```typescript
// Three.js setAnimationLoop manages RAF automatically
// Handles XR sessions, visibility changes, frame timing

renderer.setAnimationLoop((time) => {
  const delta = clock.getDelta();
  updateGameLogic(delta);

  // Visionary handles splat rendering internally
  this.splatRenderer.render(camera);

  renderer.render(scene, camera);
});

// Do NOT use window.requestAnimationFrame directly
```

### Fallback Strategy

When WebGPU is unavailable:
1. Log warning message explaining fallback
2. Create WebGL2 renderer instead
3. Visionary splats disabled (show message to user)
4. Game continues with reduced feature set (no compute shaders, no TSL, no splats)

```typescript
interface RendererResult {
  renderer: THREE.WebGLRenderer | THREE.WebGPURenderer;
  type: 'WebGPU' | 'WebGL2';
  visionarySupported: boolean;
}
```

### Color Space Configuration

```typescript
// Required for correct color reproduction
renderer.outputColorSpace = THREE.SRGBColorSpace;

// ACES filmic tone mapping for cinematic look
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.5;
```

### Visionary Integration

```typescript
// After renderer creation, initialize Visionary:
import { VisionarySplatRenderer } from './rendering/VisionarySplatRenderer';

// Check platform compatibility
const platform = getPlatformInfo();
if (platform.visionarySupported) {
  this.splatRenderer = new VisionarySplatRenderer();
  await this.splatRenderer.initialize(this.renderer, this.scene);
} else {
  console.warn('Visionary not supported on this platform');
  // Show UI message: Gaussian splats unavailable
  this.splatRenderer = null;
}

// In render loop:
if (this.splatRenderer) {
  this.splatRenderer.render(this.camera);
}
```

### Loading Indicator

```typescript
// Show loading during async shader compilation
function showLoadingIndicator(message: string): void {
  const existing = document.getElementById('loading-indicator');
  if (existing) {
    existing.textContent = message;
    return;
  }

  const indicator = document.createElement('div');
  indicator.id = 'loading-indicator';
  indicator.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 20px 40px;
    border-radius: 8px;
    font-family: sans-serif;
    z-index: 10000;
  `;
  indicator.textContent = message;
  document.body.appendChild(indicator);
}

function hideLoadingIndicator(): void {
  const indicator = document.getElementById('loading-indicator');
  if (indicator) {
    indicator.remove();
  }
}
```

### Testing Requirements

1. Test on Chrome/Edge (WebGPU supported, Visionary works)
2. Test on Firefox (WebGPU may require flag)
3. Test on Safari (WebGPU fallback, no splats)
4. Test on Ubuntu (Visionary NOT supported - verify graceful degradation)
5. Test on macOS (performance limited - verify warning shown)
6. Verify no memory leaks during fallback
7. Confirm async compilation doesn't block UI
8. Check pixel ratio limits on mobile devices
9. Test Visionary depth occlusion (splats behind walls)

### Platform-Specific Testing Matrix

| Platform | WebGPU | Visionary | Splats | Compute | Expected |
|----------|--------|-----------|-------|---------|----------|
| Win + Chrome | ✅ | ✅ | ✅ | ✅ | Full features |
| Win + Firefox | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Flag needed |
| Ubuntu + Chrome | ✅ | ❌ | ❌ | ✅ | No splats |
| macOS + Safari | ❌ | ❌ | ❌ | ❌ | WebGL2 fallback |
| iOS Safari | ❌ | ❌ | ❌ | ❌ | WebGL2 fallback |

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 123-163
- Visionary: https://github.com/Visionary-Laboratory/visionary
- Visionary Platform Notes: https://ai4sports.opengvlab.com/help/index.html
- Existing renderer code: `src/core/renderer.js`
- Three.js WebGPU docs: https://threejs.org/docs/#api/en/renderers/webgpu/WebGPURenderer
- WebGPU Spec: https://www.w3.org/TR/webgpu/

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

- Updated 2026-01-27 for Visionary single-canvas architecture
- Removed dual-canvas references
- Added Visionary platform detection

### File List
