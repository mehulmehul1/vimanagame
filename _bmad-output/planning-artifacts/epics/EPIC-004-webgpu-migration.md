---
title: 'WebGPU Migration Epic - Visionary Single-Canvas Architecture'
project: 'VIMANA'
date: '2026-01-27'
author: 'Mehul'
version: '2.0'
status: 'backlog'
epic_number: 4
points: 40
note: 'Revised from v1.0 - dual-canvas replaced with Visionary single-canvas architecture for proper depth occlusion'
---

# Epic 4: WebGPU Migration - Visionary Single-Canvas Architecture

## Overview

**Epic Goal:** Migrate VIMANA's rendering architecture from WebGL2 to WebGPU with Gaussian Splat support through Visionary's hybrid rendering system.

**Why This Epic:**
- WaterBall fluid simulation requires WebGPU compute shaders (unavailable in WebGL2)
- Original dual-canvas approach had critical depth buffer occlusion flaw
- **Solution: Visionary platform** - Single WebGPU canvas with proper depth compositing between meshes and splats

**Approach:** Visionary WebGPU-First Hybrid
- **Single WebGPU Context** for all rendering (meshes + splats)
- Visionary's `GaussianThreeJSRenderer` orchestrates hybrid pipeline:
  - Pass A: Three.js scene → RenderTarget (captures Color + Depth)
  - Pass B: Gaussian splats rendered using captured depth for occlusion
- Result: Proper depth testing between meshes and splats

**Duration Target:** ~3-4 weeks

**Course Correction (2026-01-27):**
- ❌ **REJECTED:** Dual-canvas architecture (WebGPU + WebGL2 overlay)
  - Reason: Separate GPU contexts cannot share depth buffers
  - Impact: Splats would always float on top, breaking occlusion
- ✅ **ADOPTED:** Visionary single-canvas architecture
  - Native WebGPU with depth compositing
  - Apache 2.0 license, active development
  - Three.js r171.0+ integration

---

## Success Definition

**Done When:**
- WebGPURenderer is the primary renderer
- All scene shaders (Vortex, Water, Shells, Jellies) converted to TSL (outputs WGSL)
- WaterBall fluid simulation active at 60 FPS
- Visionary Gaussian splats render with proper depth occlusion
- Splats are correctly hidden behind walls and geometry
- Fallback to WebGL2 works for non-WebGPU browsers (no splats in fallback)

---

## Stories

### Story 4.1: Visionary Gaussian Splat Integration

**Status:** backlog
**Estimated:** 2 days
**Dependencies:** None

**Description:**
Integrate Visionary's WebGPU Gaussian splat renderer with the existing Three.js scene, enabling proper depth occlusion between splats and scene meshes.

**Acceptance Criteria:**
- [ ] Install `visionary-core` package from npm
- [ ] Initialize `GaussianThreeJSRenderer` with WebGPU renderer
- [ ] Load Gaussian splat files (PLY, SPLAT format support)
- [ ] **CRITICAL:** Verify depth occlusion works (splats behind walls are hidden)
- [ ] Integrate with existing render loop via `onBeforeRender` callbacks
- [ ] Support 6 jelly creature splats with individual transforms
- [ ] Proper disposal of Visionary resources

**Technical Implementation:**
```typescript
// src/rendering/VisionarySplatRenderer.ts
import { GaussianThreeJSRenderer } from 'visionary';
import { GaussianModel } from 'visionary';

export class VisionarySplatRenderer {
  private renderer: GaussianThreeJSRenderer;
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
    scene.add(this.renderer); // Gets onBeforeRender callbacks
  }

  async loadSplat(url: string): Promise<void> {
    const model = await GaussianModel.load(url);
    this.models.push(model);
  }

  render(camera: THREE.Camera): void {
    // Visionary handles depth capture automatically
    this.renderer.renderThreeScene(camera);
    this.renderer.drawSplats(webgpuRenderer, scene, camera);
  }

  dispose(): void {
    this.renderer.dispose();
  }
}
```

**Dev Notes:**
- Visionary requires Three.js r171.0+
- Auto-depth mode is enabled by default - captures scene depth for occlusion
- Platform limitations: Ubuntu NOT supported, macOS performance limited

---

### Story 4.2: WebGPU Renderer Initialization

**Status:** backlog
**Estimated:** 1 day
**Dependencies:** None (changed from Story 4.1)

**Description:**
Create WebGPU renderer initialization with feature detection and async compilation.

**Acceptance Criteria:**
- [ ] `createWebGPURenderer()` function with feature detection
- [ ] `WebGPU.isAvailable()` check before initialization
- [ ] Graceful fallback to WebGL2 if WebGPU unavailable
- [ ] Async shader compilation via `renderer.compileAsync()`
- [ ] Color management: `outputColorSpace = SRGBColorSpace`
- [ ] Mobile pixel ratio limits (1.5 for mobile, 2 for desktop)
- [ ] setAnimationLoop pattern (not requestAnimationFrame)

**Technical Implementation:**
```typescript
// src/rendering/createWebGPURenderer.ts
export async function createWebGPURenderer(
  canvas: HTMLCanvasElement,
  options?: THREE.WebGPURendererParameters
): Promise<THREE.WebGPURenderer>

// Integration in main.js:
this.renderer = await createWebGPURenderer(
  canvas,
  { alpha: true, antialias: false }
);

// Initialize Visionary after renderer
this.splatRenderer = new VisionarySplatRenderer();
await this.splatRenderer.initialize(this.renderer, this.scene);
```

**Dev Notes:**
- Changed from original: `requiresWebGL: false` to use WebGPU for Visionary
- Visionary requires WebGPU - splats won't render in WebGL2 fallback

---

### Story 4.3: Vortex Shader → TSL Migration

**Status:** backlog
**Estimated:** 2 days
**Dependencies:** Story 4.2

**Description:**
Convert the vortex sphere constraint shader from GLSL to TSL (Three.js Shading Language).

**Acceptance Criteria:**
- [ ] Vortex shader converted to TSL using `Fn()` pattern
- [ ] Sphere constraint logic implemented in TSL nodes
- [ ] `positionLocal.sub(center)` for vertex transformation
- [ ] `mix()`, `distance()`, `max()` TSL math functions
- [ ] `If()` conditional for tunnel radius constraint
- [ ] Activation-based animation preserved
- [ ] Visual parity with GLSL version

**Technical Implementation:**
```typescript
// src/shaders/tsl/VortexShader.ts
import { Fn, positionLocal, uniform, mix, distance, vec3, float, max, If } from 'three/tsl';

const sphereConstraint = Fn(({ position, center, activation }) => {
  // Sphere constraint logic
});

export const vortexMaterial = new MeshStandardNodeMaterial();
vortexMaterial.positionNode = sphereConstraint({...});
```

---

### Story 4.4: Water Material → TSL Migration

**Status:** backlog
**Estimated:** 3 days
**Dependencies:** Story 4.3

**Description:**
Convert the water surface shader to TSL with Fresnel effects and bioluminescence.

**Acceptance Criteria:**
- [ ] Water shader converted to TSL
- [ ] Fresnel effect using `dot(normalView, positionView.normalize())`
- [ ] `pow(float(1).sub(dot(...)), float(3))` for edge intensity
- [ ] Bioluminescent color mixing with `mix()`
- [ ] 6-string frequency response preserved
- [ ] `timerLocal()` for animation
- [ ] Physical material: transmission, roughness, IOR

**Technical Implementation:**
```typescript
// src/shaders/tsl/WaterShader.ts
import { MeshPhysicalNodeMaterial, normalView, positionView, cameraPosition,
         dot, pow, float, mix, color, texture, timerLocal, sin } from 'three/tsl';

const fresnel = pow(
  float(1).sub(abs(dot(normalView, positionView.normalize()))),
  float(3)
);

material.transmissionNode = float(0.8);
material.iorNode = float(1.33); // Water IOR
```

---

### Story 4.5: Shell SDF → TSL Migration

**Status:** backlog
**Estimated:** 2 days
**Dependencies:** Story 4.4

**Description:**
Convert the procedural SDF shell shader to TSL.

**Acceptance Criteria:**
- [ ] Nautilus spiral SDF converted to TSL
- [ ] Iridescence effect using view angle
- [ ] Simplex noise dissolve effect
- [ ] 3-second appear animation
- [ ] Easing functions in TSL (`smoothstep()`)
- [ ] Bobbing idle animation

**Technical Implementation:**
```typescript
// src/shaders/tsl/ShellShader.ts
import { Fn, positionLocal, cameraPosition, distance, sin, cos,
         noise, smoothstep, float } from 'three/tsl';

const nautilusSDF = Fn(({ position, time }) => {
  // Nautilus spiral SDF formula
});
```

---

### Story 4.6: Jelly Shader → TSL Migration

**Status:** backlog
**Estimated:** 2 days
**Dependencies:** Story 4.4

**Description:**
Convert the bioluminescent jelly creature shader to TSL.

**Acceptance Criteria:**
- [ ] Jelly shader converted to TSL
- [ ] Organic pulsing animation using `sin(timerLocal())`
- [ ] Bioluminescent glow with teaching state
- [ ] Per-jelly frequency variations
- [ ] Emissive material with `emissiveNode`

**Technical Implementation:**
```typescript
// src/shaders/tsl/JellyShader.ts
const pulse = sin(timerLocal().mul(jellyFrequency));
material.emissiveNode = baseJellyColor.mul(pulse);
```

---

### Story 4.7: Fluid System Activation

**Status:** backlog
**Estimated:** 3 days
**Dependencies:** Story 4.2, Story 4.4

**Description:**
Enable WaterBall fluid simulation with WebGPU context and compute shader execution.

**Acceptance Criteria:**
- [ ] WebGPURenderer context passed to fluid system
- [ ] Compute shaders execute via `renderer.compute(computeNode)`
- [ ] Compute executed BEFORE render in loop
- [ ] TSL compute shaders for particle physics
- [ ] 10,000 particles at 60 FPS
- [ ] Storage buffers: position, velocity, force
- [ ] Boundary collision with `If()` conditions

**Technical Implementation:**
```typescript
// In render loop:
if (this.computeNode) {
  this.webgpuRenderer.compute(this.computeNode);
}
this.webgpuRenderer.render(this.scene, this.camera);

// src/systems/fluid/MLSMPMSimulator.ts (TSL version)
const p2gCompute = Fn(() => {
  const position = positionBuffer.element(instanceIndex);
  const velocity = velocityBuffer.element(instanceIndex);
  position.addAssign(velocity.mul(deltaTime));
  // Boundary collision...
});
```

---

### Story 4.8: Performance Validation

**Status:** backlog
**Estimated:** 2 days
**Dependencies:** Story 4.7

**Description:**
Profile and optimize the Visionary + WebGPU system for target frame rates.

**Acceptance Criteria:**
- [ ] Frame time profiling (WebGPU: 8ms, Fluid: 4ms, Splats: 3ms, Post: 1.5ms)
- [ ] Desktop 60 FPS achieved
- [ ] Mobile 30 FPS achieved
- [ ] Memory profiling (<500MB on iOS Safari)
- [ ] Device profile system (high/medium/low)
- [ ] Pixel ratio limits enforced
- [ ] Async shader compilation prevents frame drops
- [ ] Visionary depth occlusion verified (splats hidden behind walls)

**Technical Implementation:**
```typescript
// src/utils/DeviceCapabilities.ts
interface DeviceProfile {
  pixelRatio: number;
  textureSize: number;
  shadows: boolean;
  postprocessing: boolean;
  antialias: boolean;
  visionarySupported: boolean; // Ubuntu not supported
}

const getDeviceProfile = (): DeviceProfile => {
  // Mobile detection and profile selection
  // Note: Ubuntu returns visionarySupported: false
};
```

---

## Epic Retrospective

**Status:** optional
**To be completed after:** All stories reach "done" status

**Retrospective Questions:**
- Was Visionary the right choice over dual-canvas?
- Did TSL conversion go smoothly?
- Were there unexpected WebGPU compatibility issues?
- How did Visionary's depth compositing perform?
- Were performance targets met?
- What would we do differently?

---

## Technical References

### Skills & Documentation
- `.claude/skills/3d-graphics/references/16-webgpu.md` - WebGPU fundamentals
- `.claude/skills/3d-graphics/references/13-node-materials.md` - Node materials
- `.claude/skills/three-best-practices/rules/tsl-complete-reference.md` - Full TSL API
- `.claude/skills/three-best-practices/rules/tsl-compute-shaders.md` - Compute shaders
- `.claude/skills/three-best-practices/rules/tsl-post-processing.md` - Post-processing
- `.claude/skills/three-best-practices/rules/mobile-optimization.md` - Mobile optimization

### External Resources
- Three.js TSL Wiki: https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language
- React Three Fiber v9 Migration: https://r3f.docs.pmnd.rs/tutorials/v9-migration-guide
- WebGPU Spec: https://www.w3.org/TR/webgpu/
- **Visionary: https://github.com/Visionary-Laboratory/visionary**
- **Visionary Docs: https://ai4sports.opengvlab.com/help/index.html**
- **Visionary Three.js Integration: https://ai4sports.opengvlab.com/help/modules/12-three-integration/index.html**
- WaterBall Reference: https://github.com/matsuoka-601/WaterBall
- Maxime Heckel's TSL Field Guide: https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              VISIONARY HYBRID RENDERING ARCHITECTURE             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SINGLE WebGPU Canvas + Context                          │    │
│  │                                                          │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  PASS A: Three.js Scene → RenderTarget           │  │    │
│  │  │  • Vortex (TSL → WGSL)                           │  │    │
│  │  │  • Water Material (TSL → WGSL)                   │  │    │
│  │  │  • Shells/Jellies (TSL → WGSL)                   │  │    │
│  │  │  → Captures COLOR + DEPTH to RenderTarget        │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  │                          │                               │    │
│  │                          ▼                               │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  PASS B: Gaussian Splats (uses captured depth)    │  │    │
│  │  │  • Visionary: GaussianThreeJSRenderer            │  │    │
│  │  │  • Reads depth from Pass A                        │  │    │
│  │  │  • Proper occlusion: splats behind walls hidden  │  │    │
│  │  │  • Jelly creature splats (6 instances)            │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  │                                                          │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  PASS C: WaterBall Fluid (WGSL compute)           │  │    │
│  │  │  • 10,000 particles @ 60 FPS                      │  │    │
│  │  │  • TSL compute shaders                            │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  UI Overlay (pointer-events: auto)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

KEY BENEFIT: Single WebGPU context = shared depth buffer = proper occlusion
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| TSL learning curve | Medium | Use skill references; prototype in TSL playground |
| Visionary platform stability | Low | Apache 2.0 license; active development; research-backed |
| **Ubuntu NOT supported** | Medium | Document platform limitation; target Windows/macOS |
| **macOS performance limited** | Medium | M4 Max+ recommended; test on target hardware |
| Three.js version compatibility | Low | Requires r171.0+; verify current version |
| WebGPU browser support | Low | Feature detection; fallback to WebGL2 (no splats) |
| Performance regression | Medium | Profile each phase; use device profiles |
| Memory on iOS Safari | High | Aggressive texture limits; lazy load splats |
| Compute shader bugs | Medium | Test particle physics thoroughly |

---

## Timeline Notes

**Dependencies:**
- Story 4.1 (Visionary) and Story 4.2 (WebGPU init) can proceed in parallel
- Story 4.2 required for any WebGPU work
- Stories 4.3, 4.4, 4.5, 4.6 can proceed in parallel after 4.2
- Story 4.7 depends on 4.2 and 4.4
- Story 4.8 is final validation

**Estimated effort:** 3-4 weeks with one developer

**Platform Limitations:**
- Ubuntu: NOT supported (Visionary fp16 WebGPU bug)
- macOS: Performance limited (M4 Max+ recommended)
- Windows 10/11: RECOMMENDED (discrete GPU)

---

**Total Points: 40** (reduced from 45 due to simplified architecture)
**Estimated Duration:** 3-4 weeks
