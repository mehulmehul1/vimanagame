---
title: 'WebGPU Migration Architecture'
project: 'VIMANA'
date: '2026-01-27'
author: 'Mehul'
version: '2.0'
status: 'in-progress'
epic: 'EPIC-004'

# Source Documents
gdd: '../gdd.md'
epic_waterball: 'epics/EPIC-002-waterball-fluid-system.md'
skills:
  - '.claude/skills/3d-graphics/references/16-webgpu.md'
  - '.claude/skills/3d-graphics/references/13-node-materials.md'
  - '.claude/skills/three-best-practices/rules/tsl-complete-reference.md'
  - '.claude/skills/three-best-practices/rules/tsl-compute-shaders.md'
  - '.claude/skills/three-best-practices/rules/tsl-post-processing.md'
  - '.claude/skills/three-best-practices/rules/mobile-optimization.md'
---

# WebGPU Migration Architecture v2.0

## Document Status

This architecture document defines the migration path from WebGL2 to WebGPU for the VIMANA project, enabling the WaterBall fluid simulation system while maintaining Gaussian Splat rendering compatibility.

**Approach:** WebGPU-First Hybrid (WebGPU main renderer + WebGL2 overlay for splats)

**Changes in v2.0:**
- Fixed code errors from v1.0
- Updated to modern Three.js r170+ import paths
- Added setAnimationLoop pattern
- Added async compilation
- Added compute shader execution pattern
- Added comprehensive TSL node reference
- Added mobile optimization considerations
- Added post-processing setup
- Added proper color management

---

## Executive Summary

### The Problem

VIMANA's rendering architecture faces a fundamental incompatibility:

| Component | Current Tech | Target Tech | Compatible? |
|-----------|--------------|-------------|-------------|
| WaterBall Fluid | (disabled) | WebGPU Compute | ❌ Requires WebGPU |
| Gaussian Splats | @sparkjsdev/spark | WebGL2 only | ❌ WebGL2 only |
| Scene Shaders | GLSL | WGSL needed | ❌ Different language |

**Root Cause:** WebGPURenderer and WebGLRenderer cannot share a single canvas context.

### The Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-RENDERER ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 1 (Bottom): WebGPU Canvas - Primary Renderer     │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  • Vortex (TSL → WGSL)                            │  │    │
│  │  │  • Water Material (TSL → WGSL)                    │  │    │
│  │  │  • Shells/Jellies (TSL → WGSL)                    │  │    │
│  │  │  • WaterBall Fluid (native WGSL compute)          │  │    │
│  │  │  • Post-processing (TSL pass-based)               │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 2 (Top): WebGL2 Canvas - Overlay (Transparent)   │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  • Spark.js Gaussian Splats (WebGL2 only)         │  │    │
│  │  │  • pointer-events: none (clicks pass through)     │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 3: UI Overlay (pointer-events: auto)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Decisions

### Decision 1: Renderer Strategy

**Chosen Approach:** WebGPU-First Hybrid

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **WebGPU-First Hybrid** | Full compute power, unified post-processing, future-proof | Splat layer excludes from post-processing | ✅ **SELECTED** |
| **WebGL2-First + Hidden WebGPU** | Existing content unchanged, lower risk | Complex dual-context management, fluid as second-class | ❌ |
| **Full WebGL2 Fluid Rewrite** | Single renderer, simple | Significant quality loss, defeats purpose | ❌ |
| **Wait for Spark.js WebGPU** | Native solution | Unknown timeline (6+ months), blocks development | ❌ |

**Rationale:**
- WebGPU is the GDD's stated primary target
- WaterBall fluid requires native compute shaders
- Performance and visual quality are project goals
- Overlay pattern is proven (used by many hybrid engines)

### Decision 2: Shader Migration Path

**Technology:** Three.js TSL (Three.js Shading Language)

**Why TSL:**
- ✅ Outputs BOTH WGSL (WebGPU) and GLSL (WebGL2) via different builders
- ✅ Type-safe with TypeScript
- ✅ Official Three.js solution (future-proof)
- ✅ Automatic optimization
- ✅ Supports WebGPU compute pipelines
- ✅ Function composition with `Fn()`

**Import Paths (Three.js r170+):**

```typescript
// Modern import paths for WebGPU
import * as THREE from 'three/webgpu';
import {
  // Core nodes
  uniform, timerLocal, time,
  positionLocal, positionWorld, positionView,
  normalLocal, normalView, normalWorld,
  cameraPosition, cameraNear, cameraFar,
  uv, screenUV, viewportUV,

  // Math
  sin, cos, tan, abs, min, max, clamp, mix,
  add, sub, mul, div, pow, sqrt, length, normalize,
  vec2, vec3, vec4, float, int, bool, color,

  // Flow control
  Fn, If, Else, select, Loop, Break, Continue, Discard,

  // Textures
  texture, textureLoad, cubeTexture,

  // Compute
  storage, storageBuffer, instancedArray, instanceIndex,
  workgroupId, localId, globalId,

  // Post-processing
  pass, bloom, gaussianBlur, grayscale, saturation,

  // Material types
  MeshStandardNodeMaterial,
  MeshPhysicalNodeMaterial,
  MeshBasicNodeMaterial,
} from 'three/tsl';
```

### Decision 3: Canvas Composition

**Technique:** CSS Absolute Layering

```html
<div id="render-container" style="position: relative; width: 100vw; height: 100vh;">
  <!-- Layer 1: WebGPU (bottom) -->
  <canvas id="webgpu-canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></canvas>

  <!-- Layer 2: WebGL2 (top, transparent) -->
  <canvas id="webgl-canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>

  <!-- Layer 3: UI (above all) -->
  <div id="ui-layer" style="position: absolute; top: 0; left: 0; pointer-events: auto;">
    <!-- UI elements here -->
  </div>
</div>
```

**Key Considerations:**
- WebGL2 canvas has `pointer-events: none` — clicks pass through to WebGPU layer
- UI layer restores `pointer-events: auto` for interactivity
- Both renderers share camera transforms via state synchronization

---

## Migration Roadmap

### Phase 1: Foundation (2 stories)
**Goal:** Set up dual-renderer infrastructure

| Story | Description | Output |
|-------|-------------|--------|
| **1.1** Dual Canvas Setup | Create canvas composition system | `RenderComposer.ts` |
| **1.2** WebGPU Renderer Init | Initialize WebGPURenderer with async init | `createWebGPURenderer.ts` |

### Phase 2: Shader Migration to TSL (4 stories)
**Goal:** Port all GLSL shaders to TSL

| Story | Shader File | Complexity |
|-------|-------------|------------|
| **2.1** Vortex Sphere Constraint | `src/shaders/index.ts` | Medium |
| **2.2** Water Material | `src/entities/WaterMaterial.ts` | High |
| **2.3** Shell SDF | `src/entities/ShellCollectible.ts` | High |
| **2.4** Jelly Shader | `src/entities/JellyCreature.ts` | Medium |

### Phase 3: Spark.js Integration (1 story)
**Goal:** Adapt Spark.js for overlay canvas

| Story | Description | Output |
|-------|-------------|--------|
| **3.1** Splat Overlay Adapter | Wrap Spark.js for transparent overlay | `SplatOverlayRenderer.ts` |

### Phase 4: Integration & Testing (2 stories)
**Goal:** Complete migration and validate

| Story | Description | Output |
|-------|-------------|--------|
| **4.1** Fluid System Activation | Enable WaterBall with WebGPU context | Active fluid |
| **4.2** Performance Validation | Profile and optimize | 60fps target |

---

## Technical Specifications

### WebGPU Initialization

```typescript
// src/rendering/createWebGPURenderer.ts
import * as THREE from 'three/webgpu';
import WebGPU from 'three/addons/capabilities/WebGPU.js';

export async function createWebGPURenderer(
  canvas: HTMLCanvasElement,
  options?: THREE.WebGPURendererParameters
): Promise<THREE.WebGPURenderer> {
  // Check WebGPU support first
  if (!WebGPU.isAvailable()) {
    const error = WebGPU.getErrorMessage();
    throw new Error('WebGPU not supported: ' + error);
  }

  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: false, // WebGPU handles differently
    alpha: true,
    ...options
  });

  await renderer.init();

  // Configure renderer (FIXED from v1.0 - removed typo)
  renderer.configure({
    antialias: false,
    alpha: true
  });

  // Color management (r151+ API)
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // Mobile optimization
  const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const pixelRatio = Math.min(window.devicePixelRatio, isMobile ? 1.5 : 2);
  renderer.setPixelRatio(pixelRatio);

  return renderer;
}
```

### Render Loop with setAnimationLoop

```typescript
// src/rendering/RenderComposer.ts
import * as THREE from 'three/webgpu';
import { timerLocal } from 'three/tsl';

export class RenderComposer {
  private webgpuRenderer: THREE.WebGPURenderer;
  private webglRenderer: SplatOverlayRenderer;
  private camera: THREE.PerspectiveCamera;
  private scene: THREE.Scene;
  private computeNode?: THREE.ComputeNode;

  async initialize(): Promise<void> {
    // Initialize WebGPU
    this.webgpuRenderer = await createWebGPURenderer(
      document.getElementById('webgpu-canvas') as HTMLCanvasElement
    );

    // Initialize WebGL2 overlay
    this.webglRenderer = new SplatOverlayRenderer(
      document.getElementById('webgl-canvas') as HTMLCanvasElement
    );
    await this.webglRenderer.initialize('/assets/splats/harp-room.sog');

    // Shared camera
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

    // Create scene
    this.scene = new THREE.Scene();

    // IMPORTANT: Compile materials asynchronously to avoid frame drops
    await this.webgpuRenderer.compileAsync(this.scene, this.camera);

    // Use setAnimationLoop (not requestAnimationFrame)
    this.webgpuRenderer.setAnimationLoop(this.render.bind(this));
  }

  render(): void {
    // Execute compute shaders first (for fluid simulation)
    if (this.computeNode) {
      this.webgpuRenderer.compute(this.computeNode);
    }

    // Render WebGPU layer (TSL materials, fluid)
    this.webgpuRenderer.render(this.scene, this.camera);

    // Sync and render WebGL2 overlay (splats)
    this.webglRenderer.syncCamera(this.camera);
    this.webglRenderer.render();
  }

  dispose(): void {
    // Cleanup both renderers
    this.webgpuRenderer.dispose();
    this.webglRenderer.dispose();
  }
}
```

### TSL Shader Conversion Examples

#### Example 1: Simple Animated Material (GLSL → TSL)

```glsl
// BEFORE: GLSL ShaderMaterial
uniform float uTime;
uniform float uActivation;
varying vec2 vUv;

void main() {
  vec2 uv = vUv;
  float pulse = sin(uTime * 2.0) * 0.5 + 0.5;
  vec3 color = mix(vec3(0.0, 0.5, 1.0), vec3(1.0, 0.5, 0.0), pulse * uActivation);
  gl_FragColor = vec4(color, 1.0);
}
```

```typescript
// AFTER: TSL NodeMaterial
import { MeshStandardNodeMaterial, timerLocal, sin, mix, vec3, color, uv } from 'three/tsl';

const uActivation = uniform(0);
const material = new MeshStandardNodeMaterial();

// Animated pulse using timerLocal
const pulse = sin(timerLocal().mul(2)).mul(0.5).add(0.5);

// Mix colors based on activation
const baseColor = color(0x0088ff);
const activeColor = color(0xff8800);
material.colorNode = mix(baseColor, activeColor, pulse.mul(uActivation));

// emissive glow
material.emissiveNode = mix(baseColor, activeColor, pulse.mul(uActivation));
```

#### Example 2: Vortex Sphere Constraint (GLSL → TSL)

```glsl
// BEFORE: GLSL
uniform float uTime;
uniform float uActivation;
uniform vec3 uCenter;

varying vec3 vPosition;

void main() {
  vec3 centered = position - uCenter;
  float dist = length(centered.xz);
  float tunnelRadius = mix(2.0, 1.0, uActivation);
  float constraint = max(tunnelRadius, dist);
  vec3 constrainedPosition = centered * (constraint / max(dist, 0.001)) + uCenter;

  if (dist < tunnelRadius) {
    constrainedPosition = centered + uCenter;
  }

  gl_Position = projectionMatrix * modelViewMatrix * vec4(constrainedPosition, 1.0);
}
```

```typescript
// AFTER: TSL with Fn() for reusability
import {
  Fn,
  positionLocal,
  uniform,
  mix,
  distance,
  vec3,
  float,
  max,
  modelWorldMatrix,
  cameraViewMatrix,
  cameraProjectionMatrix,
  If
} from 'three/tsl';

const uActivation = uniform(0);
const uCenter = vec3(0, 0.5, 2);

// Create reusable function with Fn()
const sphereConstraint = Fn(({ position, center, activation }) => {
  const centered = position.sub(center);
  const distXZ = distance(
    vec3(centered.x, float(0), centered.z),
    vec3(0, 0, 0)
  );
  const tunnelRadius = mix(float(2), float(1), activation);
  const constraint = max(tunnelRadius, distXZ);

  const result = centered.mul(constraint.div(max(distXZ, float(0.001)))).add(center);

  If(distXZ.lessThan(tunnelRadius), () => {
    result.assign(centered.add(center));
  });

  return result;
});

// Use in vertex node
const constrainedPosition = sphereConstraint({
  position: positionLocal,
  center: uCenter,
  activation: uActivation
});

material.positionNode = constrainedPosition;
```

#### Example 3: Fresnel Effect (Water Surface)

```typescript
import {
  MeshPhysicalNodeMaterial,
  normalView,
  positionView,
  cameraPosition,
  dot,
  pow,
  float,
  mix,
  color,
  texture
} from 'three/tsl';

const material = new MeshPhysicalNodeMaterial();

// Fresnel effect for edge glow
const viewDir = positionView.sub(cameraPosition).normalize();
const fresnel = pow(
  float(1).sub(abs(dot(normalView, viewDir))),
  float(3) // Edge intensity
);

// Mix water color with edge glow
const waterColor = texture(waterTexture).rgb;
const edgeColor = color(0x00ffff); // Cyan bioluminescence
material.colorNode = mix(waterColor, edgeColor, fresnel.mul(0.5));

// Transmission for water-like appearance
material.transmissionNode = float(0.8);
material.roughnessNode = float(0.1);
material.iorNode = float(1.33); // Water IOR
```

### Compute Shaders for WaterBall Fluid

```typescript
// src/systems/fluid/MLSMPMSimulator.ts (TSL version)
import {
  Fn,
  instancedArray,
  instanceIndex,
  deltaTime,
  storage,
  float,
  vec3,
  If,
  workgroupBarrier
} from 'three/tsl';

const PARTICLE_COUNT = 10000;

// Storage buffers for particle data
const positionBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const velocityBuffer = instancedArray(PARTICLE_COUNT, 'vec3');
const forceBuffer = instancedArray(PARTICLE_COUNT, 'vec3');

// Compute shader: particle to grid transfer
const p2gCompute = Fn(() => {
  const position = positionBuffer.element(instanceIndex);
  const velocity = velocityBuffer.element(instanceIndex);

  // Apply velocity to position
  position.addAssign(velocity.mul(deltaTime));

  // Boundary collision
  const bounds = float(5);
  If(position.x.abs().greaterThan(bounds), () => {
    velocity.x.assign(velocity.x.negate().mul(0.8)); // Bounce with damping
    position.x.assign(bounds.mul(position.x.sign()));
  });

  If(position.y.lessThan(0), () => {
    velocity.y.assign(velocity.y.negate().mul(0.8));
    position.y.assign(0);
  });

  If(position.z.abs().greaterThan(bounds), () => {
    velocity.z.assign(velocity.z.negate().mul(0.8));
    position.z.assign(bounds.mul(position.z.sign()));
  });
});

// Create compute node (workgroup size of 64 for optimal GPU utilization)
const computeNode = p2gCompute().compute(PARTICLE_COUNT);

// In render loop:
// renderer.compute(computeNode);
```

### Post-Processing with TSL

```typescript
// src/rendering/PostProcessing.ts
import {
  pass,
  bloom,
  gaussianBlur,
  saturation,
  toneMapping
} from 'three/tsl';

class PostProcessing {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.scene = scene;
    this.camera = camera;
  }

  getOutputNode() {
    // Scene pass (renders the scene to a texture)
    const scenePass = pass(this.scene, this.camera);
    const beauty = scenePass.getTextureNode();

    // Apply effects chain
    const saturated = saturation(beauty, 1.2);
    const bloomed = bloom(saturated, 0.5, 0.4, 0.85);

    return bloomed;
  }
}

// In renderer setup:
// const postProcessing = new PostProcessing(scene, camera);
// renderer.postProcessing.outputNode = postProcessing.getOutputNode();
```

### Spark.js Overlay Adapter

```typescript
// src/rendering/SplatOverlayRenderer.ts
import { Spark } from '@sparkjsdev/spark';

export class SplatOverlayRenderer {
  private spark: Spark | null = null;
  private canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  async initialize(splatUrl: string): Promise<void> {
    this.spark = await Spark.init(this.canvas, {
      url: splatUrl,
      enableStatic: true,
      splatAlpha: 1.0
    });

    // Ensure transparent background
    this.canvas.style.background = 'transparent';
  }

  syncCamera(webgpuCamera: THREE.PerspectiveCamera): void {
    if (!this.spark) return;

    // Copy camera transform from WebGPU renderer
    this.spark.setCamera(webgpuCamera);
  }

  render(): void {
    this.spark?.render();
  }

  dispose(): void {
    this.spark?.dispose();
    this.spark = null;
  }
}
```

### Mobile Optimization

```typescript
// src/utils/DeviceCapabilities.ts
interface DeviceProfile {
  pixelRatio: number;
  textureSize: number;
  shadows: boolean;
  postprocessing: boolean;
  antialias: boolean;
  precision: 'mediump' | 'highp';
}

const getDeviceProfile = (): DeviceProfile => {
  const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const isLowEnd = navigator.hardwareConcurrency <= 4;
  const hasLimitedMemory = navigator.deviceMemory && navigator.deviceMemory < 4;

  if (!isMobile) {
    // Desktop
    return {
      pixelRatio: Math.min(window.devicePixelRatio, 2),
      textureSize: 2048,
      shadows: true,
      postprocessing: true,
      antialias: true,
      precision: 'highp'
    };
  }

  if (isLowEnd || hasLimitedMemory) {
    // Low-end mobile
    return {
      pixelRatio: 1,
      textureSize: 512,
      shadows: false,
      postprocessing: false,
      antialias: false,
      precision: 'mediump'
    };
  }

  // High-end mobile
  return {
    pixelRatio: 1.5,
    textureSize: 1024,
    shadows: true,
    postprocessing: false, // Minimal on mobile
    antialias: false,
    precision: 'mediump'
  };
};
```

---

## File Structure

```
src/
├── rendering/
│   ├── RenderComposer.ts          # NEW - Orchestrates dual renderers
│   ├── createWebGPURenderer.ts    # NEW - WebGPU initialization
│   ├── SplatOverlayRenderer.ts    # NEW - Spark.js wrapper
│   └── PostProcessing.ts          # NEW - TSL post-processing setup
│
├── shaders/
│   └── tsl/                        # NEW - TSL shader definitions
│       ├── VortexShader.ts         # Sphere constraint in TSL
│       ├── WaterShader.ts          # Water surface with Fresnel
│       ├── ShellShader.ts          # SDF shell material
│       └── JellyShader.ts          # Bioluminescent jelly
│
├── entities/
│   ├── VortexMaterial.ts           # MODIFY - Convert to TSL
│   ├── WaterMaterial.ts            # MODIFY - Convert to TSL
│   ├── ShellCollectible.ts         # MODIFY - Convert to TSL
│   └── JellyCreature.ts            # MODIFY - Convert to TSL
│
├── systems/fluid/                  # EXISTING - Already WebGPU ready
│   ├── MLSMPMSimulator.ts          # MODIFY - Use TSL compute
│   ├── render/
│   │   ├── DepthThicknessRenderer.ts
│   │   └── FluidSurfaceRenderer.ts
│   └── interaction/
│
├── utils/
│   └── DeviceCapabilities.ts       # NEW - Mobile detection
│
└── scenes/
    └── HarpRoom.ts                 # MODIFY - Use RenderComposer
```

---

## Performance Budgets

### Frame Time Allocation (60 FPS target = 16.6ms)

| System | Budget | Notes |
|--------|--------|-------|
| WebGPU Scene Render | 8ms | Vortex, water, shells, jellies |
| Fluid Compute | 4ms | 10,000 particles, compute shaders |
| Splat Overlay | 3ms | Gaussian splats (WebGL2) |
| Post-Processing | 1.5ms | TSL effects (bloom, etc.) |
| Overhead | 0.1ms | Canvas sync, camera copy |

### Memory Budget

| Component | Budget | Implementation |
|-----------|--------|----------------|
| WebGPU Context | 200MB | Primary renderer |
| WebGL2 Context | 150MB | Splat renderer |
| Fluid System | 80MB | Particle buffers |
| Textures | 100MB | All materials |
| Total | ~530MB | Near iOS 500MB limit - optimize if needed |

### Mobile Performance Targets

| Metric | Desktop | Mobile | Low-End |
|--------|---------|--------|---------|
| Frame Rate | 60 FPS | 30 FPS | 30 FPS |
| Pixel Ratio | 2 | 1.5 | 1 |
| Texture Size | 2048 | 1024 | 512 |
| Post-Processing | Full | Minimal | None |
| Shadows | Yes | Yes | No |

---

## Common TSL Patterns

### Uniforms with Update Events

```typescript
import { uniform } from 'three/tsl';

// Per-frame update
const timeValue = uniform(0);
timeValue.onFrameUpdate(() => performance.now() / 1000);

// Per-render update
const cameraY = uniform(0);
cameraY.onRenderUpdate(() => camera.position.y);
```

### Custom Functions with Fn()

```typescript
import { Fn, vec3, sin, cos } from 'three/tsl';

const circularMotion = Fn(({ center, radius, angle }) => {
  const x = center.x.add(radius.mul(cos(angle)));
  const z = center.z.add(radius.mul(sin(angle)));
  return vec3(x, center.y, z);
});

// Usage
const position = circularMotion({
  center: vec3(0, 0, 0),
  radius: float(5),
  angle: timerLocal()
});
```

### Conditionals

```typescript
import { If, select } from 'three/tsl';

// Ternary (inline) - preferred for simple cases
const result = select(value.greaterThan(1), float(1), value);

// If-Else (block) - for complex logic
const process = Fn(({ input }) => {
  const result = float(input);

  If(result.greaterThan(10), () => {
    result.assign(10);
  }).ElseIf(result.lessThan(0), () => {
    result.assign(0);
  });

  return result;
});
```

### Loops

```typescript
import { Loop, int, Break } from 'three/tsl';

// Basic loop
Loop(10, ({ i }) => {
  // Process index i
});

// With condition
const counter = int(0);
Loop(counter.lessThan(100), () => {
  counter.addAssign(1);
  If(counter.greaterThan(50), () => {
    Break(); // Exit early
  });
});
```

### Texture Sampling

```typescript
import {
  texture,
  textureLoad,
  textureSize,
  screenUV,
  viewportDepthTexture
} from 'three/tsl';

// Standard texture lookup with filtering
const color = texture(diffuseTexture, uv());

// Direct lookup (no filtering)
const rawColor = textureLoad(dataTexture, uv(), 0);

// Get texture dimensions
const size = textureSize(diffuseTexture, 0);

// Screen-space effects
const depth = viewportDepthTexture();
const screenColor = texture(screenTexture, screenUV());
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **TSL learning curve** | Medium | Use skill references; prototype in TSL playground |
| **Dual-renderer sync** | Medium | Single camera object; minimal per-frame overhead |
| **Spark.js overlay issues** | Low | `pointer-events: none` is proven pattern |
| **WebGPU browser support** | Low | Feature detection; fallback to WebGL2 |
| **Performance regression** | Medium | Profile each phase; use device profiles |
| **Memory on iOS Safari** | High | Aggressive texture limits; lazy load splats |
| **Compute shader bugs** | Medium | Test particle physics thoroughly |

---

## Success Criteria

### Migration Complete When:

- [ ] WebGPURenderer is primary renderer
- [ ] All scene shaders use TSL (outputs WGSL)
- [ ] WaterBall fluid simulation active at target frame rate
- [ ] Gaussian splats render on transparent overlay
- [ ] Camera sync maintains visual consistency
- [ ] No memory leaks (dual context cleanup)
- [ ] Fallback to WebGL2 works for non-WebGPU browsers
- [ ] Mobile performance targets met (30fps minimum)
- [ ] Async compilation prevents frame drops

### Performance Targets:

| Metric | Desktop | Mobile |
|--------|---------|--------|
| Frame Rate | 60 FPS | 30 FPS |
| Initial Load Time | <5 seconds | <8 seconds |
| Memory (iOS Safari) | <500MB | <500MB |
| Shader Compile Time | <3 seconds | <5 seconds |

---

## Migration Checklist

### Pre-Migration

- [ ] Verify Three.js version >= r170
- [ ] Test WebGPU availability on target browsers
- [ ] Create backup branch
- [ ] Run existing test suite

### Phase 1: Foundation

- [ ] Create dual canvas HTML structure
- [ ] Implement RenderComposer
- [ ] Implement createWebGPURenderer with feature detection
- [ ] Implement SplatOverlayRenderer
- [ ] Test both renderers independently

### Phase 2: Shader Migration

- [ ] Convert Vortex shader to TSL
- [ ] Convert Water shader to TSL
- [ ] Convert Shell shader to TSL
- [ ] Convert Jelly shader to TSL
- [ ] Visual comparison test (GLSL vs TSL)

### Phase 3: Integration

- [ ] Enable WaterBall fluid system
- [ ] Connect compute shaders to renderer
- [ ] Verify splat overlay positioning
- [ ] Test camera synchronization

### Phase 4: Optimization

- [ ] Profile frame times
- [ ] Implement device profiles
- [ ] Optimize particle count if needed
- [ ] Add async compilation
- [ ] Test on mobile devices

### Post-Migration

- [ ] Run full test suite
- [ ] Test on Chrome/Edge (WebGPU)
- [ ] Test on Safari (WebGL2 fallback)
- [ ] Test on mobile devices
- [ ] Memory profiling
- [ ] Performance regression testing

---

## Quick TSL Reference

### Built-in Variables

```typescript
// Position
positionLocal      // Local transformed (post-skinning)
positionWorld      // World space position
positionView       // View space position

// Normal
normalLocal        // Local normal
normalView         // View space (normalized)
normalWorld        // World space (normalized)

// Camera
cameraPosition     // Camera world position
cameraNear         // Near plane distance
cameraFar          // Far plane distance

// Time
time               // Global time in seconds
timerLocal()       // Time since renderer start
deltaTime          // Time since last frame

// UVs
uv()               // Current UV
screenUV           // Normalized screen coordinate
viewportUV         // Viewport coordinate

// Compute
instanceIndex      // Current instance index
workgroupId        // Current workgroup ID
localId            // Local invocation ID
globalId           // Global invocation ID
```

### Common Node Categories

```typescript
// Input
uniform(0)                 // Uniform value
attribute('position')      // Geometry attribute
texture(tex)               // Texture sampler
storage(buffer, 'vec3')    // Storage buffer

// Math
sin(), cos(), tan()        // Trigonometry
abs(), min(), max()        // Basic math
pow(), sqrt(), exp()       // Advanced math
mix(a, b, t)               // Linear interpolation
clamp(v, min, max)         // Clamp value

// Vector
vec3(x, y, z)              // 3D vector
color(0xff0000)            // Color (vec3 wrapper)
float(1)                   // Scalar float

// Flow
If(condition, fn)          // Conditional
Loop(count, fn)            // Loop
select(cond, a, b)         // Ternary
Break(), Continue()        // Loop control
```

---

## Next Steps

1. **Create EPIC-004** in planning artifacts with 8 stories
2. **Run Sprint Planning** to add EPIC-004 to sprint-status.yaml
3. **Begin Story 1.1** - Dual Canvas Setup (RenderComposer.ts)
4. **Profile** after each phase to validate budgets
5. **Test** on target devices early and often

---

## References

### Skills Used
- `.claude/skills/3d-graphics/references/16-webgpu.md` - WebGPU fundamentals
- `.claude/skills/3d-graphics/references/13-node-materials.md` - Node materials
- `.claude/skills/three-best-practices/rules/tsl-complete-reference.md` - Full TSL API
- `.claude/skills/three-best-practices/rules/tsl-compute-shaders.md` - Compute shaders
- `.claude/skills/three-best-practices/rules/tsl-post-processing.md` - Post-processing
- `.claude/skills/three-best-practices/rules/mobile-optimization.md` - Mobile optimization
- `.claude/skills/three-best-practices/rules/migration-checklist.md` - Migration guide

### External Resources
- **Three.js TSL Wiki:** https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language
- **React Three Fiber v9:** https://r3f.docs.pmnd.rs/tutorials/v9-migration-guide
- **WebGPU Spec:** https://www.w3.org/TR/webgpu/
- **Spark.js Repository:** https://github.com/sparkjsdev/spark
- **WaterBall Reference:** https://github.com/matsuoka-601/WaterBall
- **Maxime Heckel's TSL Field Guide:** https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/
- **sbcode.net TSL Tutorials:** https://sbcode.net/tsl/getting-started/

---

**Architecture Status:** Ready for Implementation

*Cloud Dragonborn, Game Architect*
*Foundation reinforced with best practices. The path to WebGPU is clearer.*
