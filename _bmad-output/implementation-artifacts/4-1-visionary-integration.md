# Story 4.1: Visionary Gaussian Splat Integration

Status: ready-for-dev

**Note:** Replaces obsolete Story 4.1 (Dual Canvas Setup) as part of course correction on 2026-01-27.

## Story

As a rendering engineer,
I want to integrate Visionary's WebGPU Gaussian splat renderer with the existing Three.js scene,
so that Gaussian splats render with proper depth occlusion against scene meshes.

## Acceptance Criteria

1. Install `visionary-core` package from npm
2. Initialize `GaussianThreeJSRenderer` with WebGPU renderer
3. Load Gaussian splat files (PLY, SPLAT format support)
4. **CRITICAL:** Verify depth occlusion works (splats behind walls are hidden)
5. Integrate with existing render loop via `onBeforeRender` callbacks
6. Support 6 jelly creature splats with individual transforms
7. Proper disposal of Visionary resources

## Tasks / Subtasks

- [ ] Install Visionary package (AC: 1)
  - [ ] Run `npm install visionary-core`
  - [ ] Verify package installation in package.json
  - [ ] Check Three.js version compatibility (requires r171.0+)
- [ ] Create VisionarySplatRenderer class (AC: 2, 5)
  - [ ] Create `src/rendering/VisionarySplatRenderer.ts`
  - [ ] Import `GaussianThreeJSRenderer` and `GaussianModel` from visionary
  - [ ] Implement `initialize()` method with async init
  - [ ] Implement `render()` method calling renderThreeScene + drawSplats
  - [ ] Implement `dispose()` method for cleanup
- [ ] Implement splat file loading (AC: 3)
  - [ ] Add `loadSplat(url)` method
  - [ ] Support PLY format loading
  - [ ] Support SPLAT format loading
  - [ ] Handle loading errors gracefully
- [ ] Integrate with main render loop (AC: 5, 6)
  - [ ] Update main.js to initialize VisionarySplatRenderer
  - [ ] Call initialize after WebGPU renderer creation
  - [ ] Add render calls in animation loop
  - [ ] Support multiple splat instances (6 jellies)
- [ ] Verify depth occlusion (AC: 4)
  - [ ] Test splat behind wall - should be hidden
  - [ ] Test splat in front of wall - should be visible
  - [ ] Test splat partially occluded - should clip correctly
  - [ ] Add debug view mode for depth buffer visualization
- [ ] Implement resource cleanup (AC: 7)
  - [ ] Dispose Visionary renderer on scene cleanup
  - [ ] Remove GaussianThreeJSRenderer from scene graph
  - [ ] Clear all GaussianModel instances

## Dev Notes

### Visionary Integration Context

**Package:** `visionary-core`
**Repository:** https://github.com/Visionary-Laboratory/visionary
**Documentation:** https://ai4sports.opengvlab.com/help/modules/12-three-integration/index.html
**License:** Apache 2.0
**Three.js Requirement:** r171.0+

### Why Visionary Instead of Dual-Canvas

**The Problem We Solved:**
Original dual-canvas approach (WebGPU + WebGL2 overlay) had a **critical flaw**:
- Separate GPU contexts = separate depth buffers
- No depth testing between canvases
- Splats would always float on top, breaking occlusion

**Visionary's Solution:**
```typescript
// Single WebGPU context, shared depth buffer
const renderer = new GaussianThreeJSRenderer(
    webgpuRenderer,  // THREE.WebGPURenderer
    scene,
    models
);
await renderer.init();
scene.add(renderer); // Gets onBeforeRender callbacks

// In render loop:
renderer.renderThreeScene(camera);  // Captures COLOR + DEPTH
renderer.drawSplats(...);            // Uses captured depth
```

**Result:** Proper occlusion - splats behind walls are hidden!

### Basic Class Structure

```typescript
// src/rendering/VisionarySplatRenderer.ts
import * as THREE from 'three';
import { GaussianThreeJSRenderer, GaussianModel } from 'visionary';

export class VisionarySplatRenderer {
    private renderer: GaussianThreeJSRenderer | null = null;
    private models: GaussianModel[] = [];

    async initialize(
        webgpuRenderer: THREE.WebGPURenderer,
        scene: THREE.Scene
    ): Promise<void> {
        this.renderer = new GaussianThreeJSRenderer(
            webgpuRenderer,
            scene,
            this.models
        );
        await this.renderer.init();
        scene.add(this.renderer);
    }

    async loadSplat(url: string): Promise<void> {
        const model = await GaussianModel.load(url);
        this.models.push(model);
    }

    render(camera: THREE.Camera): void {
        if (!this.renderer) return;
        // Visionary handles the two-pass rendering:
        // Pass 1: Three.js scene → RenderTarget (captures depth)
        // Pass 2: Gaussian splats (uses captured depth)
        this.renderer.renderThreeScene(camera);
        this.renderer.drawSplats(
            webgpuRenderer,
            scene,
            camera
        );
    }

    dispose(): void {
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
        this.models = [];
    }
}
```

### Integration in main.js

```typescript
// After WebGPU renderer creation:
this.renderer = await createWebGPURenderer(canvas, options);

// Initialize Visionary
this.splatRenderer = new VisionarySplatRenderer();
await this.splatRenderer.initialize(this.renderer, this.scene);

// Load splat files
await this.splatRenderer.loadSplat('/assets/jellies.splat');

// In render loop:
function animate() {
    this.splatRenderer.render(this.camera);
    this.renderer.render(this.scene, this.camera);
}
```

### Jelly Creature Support

The HarpRoom has 6 jelly creatures that need splat rendering:

```typescript
// Load each jelly splat
const jellyPositions = [
    new THREE.Vector3(-10, 0, 0),
    new THREE.Vector3(-6, 0, 0),
    new THREE.Vector3(-2, 0, 0),
    new THREE.Vector3(2, 0, 0),
    new THREE.Vector3(6, 0, 0),
    new THREE.Vector3(10, 0, 0)
];

for (let i = 0; i < 6; i++) {
    const model = await GaussianModel.load(`/assets/jelly-${i}.splat`);
    model.position.copy(jellyPositions[i]);
    this.models.push(model);
}
```

### Platform Limitations

**IMPORTANT - Visionary Platform Support:**

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | ✅ RECOMMENDED | Discrete GPU (NVIDIA/AMD) |
| Ubuntu/Linux | ❌ NOT SUPPORTED | fp16 WebGPU bug |
| macOS | ⚠️ LIMITED | M4 Max+ recommended |
| Browser | Chrome/Edge | WebGPU required |

**Fallback Strategy:**
If WebGPU unavailable:
- Graceful degradation to WebGL2 for main rendering
- Gaussian splats disabled (show error message in UI)
- Game continues with TSL shaders converted to GLSL

### Depth Occlusion Testing

**Test Cases:**
1. **Wall Test:** Place splat behind wall - verify it's hidden
2. **Transparency Test:** Splat through transparent water - should be visible
3. **Partial Occlusion:** Splat halfway through geometry - should clip
4. **Multiple Splats:** Overlapping splats - should depth-sort correctly

**Debug Visualization:**
```typescript
// Enable depth visualization for testing
this.renderer.setDepthDebugMode(true);
```

### Performance Considerations

- Visionary uses GPU-based sorting for millions of Gaussian particles
- Auto-depth mode adds one render pass (Three.js scene → RenderTarget)
- Memory: Each GaussianModel stores positions, covariance, SH coefficients
- Target: 60 FPS on desktop, 30 FPS on mobile

### File Structure

**New file:**
```
src/rendering/VisionarySplatRenderer.ts    # Main wrapper class
```

**Modified:**
```
src/main.js                                 # Initialize VisionarySplatRenderer
package.json                                # Add visionary-core dependency
```

### Troubleshooting

**Issue:** "Cannot read property of undefined"
- **Cause:** Three.js version too old
- **Fix:** Upgrade to r171.0+

**Issue:** Splats render but occlusion doesn't work
- **Cause:** Auto-depth mode disabled
- **Fix:** Ensure `renderer.init()` is called before adding to scene

**Issue:** Performance degradation
- **Cause:** Too many Gaussian particles
- **Fix:** Reduce splat count or use LOD

### Testing Requirements

1. Verify PLY file loading
2. Verify SPLAT file loading
3. Test depth occlusion with wall geometry
4. Test 6 jelly splats at different positions
5. Test splat disposal (memory cleanup)
6. Performance test on target hardware
7. Test WebGL2 fallback behavior

### References

- Visionary GitHub: https://github.com/Visionary-Laboratory/visionary
- Three.js Integration: https://ai4sports.opengvlab.com/help/modules/12-three-integration/index.html
- Visionary API Reference: https://ai4sports.opengvlab.com/help/api/index.html
- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 58-120

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
