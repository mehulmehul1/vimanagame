# STORY-002-001: MLS-MPM Particle Physics System

**Epic**: `EPIC-002` - WaterBall Fluid Simulation System
**Story ID**: `STORY-002-001`
**Points**: `8`
**Status**: `Ready for Dev`
**Owner**: `TBD`

---

## User Story

As a **player**, I want **water that behaves like real fluid**, so that **my interactions and the music create realistic, satisfying water responses**.

---

## Overview

Implement the MLS-MPM (Moving Least Squares Material Point Method) particle physics simulation from [WaterBall](https://github.com/matsuoka-601/WaterBall) using WebGPU compute shaders. This is the core physics engine that drives all fluid behavior.

**Sources:**
- [WaterBall MLS-MPM Implementation](https://github.com/matsuoka-601/WaterBall/tree/main/mls-mpm)
- [WebGPU-Ocean MLS-MPM](https://github.com/matsuoka-601/WebGPU-Ocean/tree/main/mls-mpm)

---

## Technical Specification

### 1. Particle Data Structure

```typescript
// Particle struct (80 bytes per particle)
struct Particle {
    position: vec3f,      // 12 bytes - world position
    v: vec3f,            // 12 bytes - velocity
    C: mat3x3f,          // 36 bytes - affine momentum matrix
    density: f32,        // 4 bytes - particle density
    padding: f32,        // 4 bytes - alignment
}

// Max particles: 10,000
const PARTICLE_BUFFER_SIZE = 80 * 10_000; // 800KB
```

### 2. Grid Cell Structure

```typescript
// Cell for MLS-MPM grid (16 bytes per cell)
struct Cell {
    vx: i32,  // velocity X (fixed point)
    vy: i32,  // velocity Y (fixed point)
    vz: i32,  // velocity Z (fixed point)
    mass: i32, // cell mass
}

// Grid: 80×80×80 max
const GRID_SIZE = 80 * 80 * 80; // 512,000 cells = 8MB
```

### 3. Compute Pipeline Stages

Create 6 compute shaders in `src/systems/fluid/compute/`:

| Shader | Purpose | Workgroups |
|--------|---------|------------|
| `clearGrid.wgsl` | Reset grid cells each frame | ceil(gridCount/64) |
| `p2g_1.wgsl` | Particle-to-Grid: transfer mass/velocity | ceil(particles/64) |
| `p2g_2.wgsl` | Particle-to-Grid: update grid with MLS-MPM | ceil(particles/64) |
| `updateGrid.wgsl` | Grid-to-Grid: apply forces, constraints | ceil(gridCount/64) |
| `g2p.wgsl` | Grid-to-Particle: update particle velocities | ceil(particles/64) |
| `copyPosition.wgsl` | Copy positions to render buffer | ceil(particles/64) |

### 4. Physics Constants (from WaterBall)

```typescript
const FLUID_CONFIG = {
    // Simulation
    stiffness: 3.0,              // Material stiffness
    restDensity: 4.0,            // Rest density
    dynamicViscosity: 0.1,       // Viscosity
    dt: 0.016,                   // Time step (~60fps)
    fixedPointMultiplier: 1e7,    // For velocity encoding

    // Grid
    gridResolution: 80,
    cellSize: 1.0,

    // Particles
    maxParticles: 10000,
    spawnRate: 100,              // Particles per spawn
    particleSpacing: 0.55,

    // Sphere constraint (for tunnel effect)
    sphereRadius: 15.0,          // Will be dynamic
    sphereStiffness: 3.0,
    sphereAttraction: 0.1,
};
```

### 5. File Structure

```
src/systems/fluid/
├── MLSMPMSimulator.ts      # Main particle system class
├── compute/
│   ├── clearGrid.wgsl
│   ├── p2g_1.wgsl
│   ├── p2g_2.wgsl
│   ├── updateGrid.wgsl
│   ├── g2p.wgsl
│   └── copyPosition.wgsl
└── types.ts                     # Particle, Cell structs
```

---

## Implementation Tasks

1. **[COMPUTE]** Create WGSL compute shaders for MLS-MPM pipeline
2. **[TYPESCRIPT]** Implement `MLSMPMSimulator` class with WebGPU setup
3. **[BUFFERS]** Allocate and bind particle buffer, grid buffer, uniform buffers
4. **[PIPELINE]** Create compute pipelines for each shader stage
5. **[EXECUTION]** Implement `execute()` method running all stages per frame
6. **[SPAWNING]** Implement dam-break pattern particle spawning
7. **[CONSTRAINTS]** Implement sphere constraint for tunnel effect

---

## Sphere Constraint (Critical for Tunnel Effect)

From `g2p.wgsl` - this creates the sphere tunnel:

```wgsl
let center = vec3f(real_box_size.x / 2, real_box_size.y / 2, real_box_size.z / 2);
let dist = center - particles[id.x].position;
let dirToOrigin = normalize(dist);
let r: f32 = sphereRadius;

// Push particles toward sphere surface
if (dot(dist, dist) < r * r) {
    particles[id.x].v += -(r - sqrt(dot(dist, dist))) * dirToOrigin * 3.0;
}

// Gentle attraction to center
particles[id.x].v += dirToOrigin * 0.1;
```

**Animation**: As `uDuetProgress` goes 0→1, animate `realBoxSize.z` to shrink box, forcing sphere constraint to create hollow tunnel.

---

## WebGPU Bind Group Layout

```typescript
// BindGroup 0: Particle and Grid buffers
BindGroup(0) = [
    { binding: 0, resource: particleBuffer },      // storage, read_write
    { binding: 1, resource: cellBuffer },         // storage, read_write
    { binding: 2, resource: realBoxSizeBuffer },   // uniform
    { binding: 3, resource: initBoxSizeBuffer },   // uniform
    { binding: 4, resource: numParticlesBuffer },  // uniform
    { binding: 5, resource: sphereRadiusBuffer }, // uniform
]
```

---

## Acceptance Criteria

- [ ] `MLSMPMSimulator` class instantiated with WebGPU device
- [ ] All 6 compute shaders compile without errors
- [ ] 10,000 particles simulate at 60fps (measure with performance.now())
- [ ] Particles spawn in dam-break pattern on initialization
- [ ] Sphere constraint activates when `uDuetProgress > 0`
- [ ] Debug view: `window.debugVimana.fluid.getParticleCount()` returns live count
- [ ] Debug view: `window.debugVimana.fluid.toggleWireframe()` shows particle bounds

---

## Dependencies

- **Requires**: WebGPU device (already have fallback in `renderer.js`)
- **Blocked by**: None (can start immediately)
- **Blocks**: STORY-002 (needs particle positions for rendering)

---

## Notes from WaterBall Analysis

- **Fixed Point Encoding**: Velocities encoded as i32 with multiplier 1e7 for precision
- **Sub-stepping**: Physics runs 2x per frame for stability (see `execute()` loop)
- **Grid indexing**: `idx = x * (y*z) + y * z + z`
- **Bicubic weights**: `[0.5-d)², 0.75-d², (0.5+d)²]` for particle-to-grid transfer

---

**Sources:**
- [WaterBall MLS-MPM](https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts)
- [WebGPU Fluid Simulations Article](https://tympanis.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/)
