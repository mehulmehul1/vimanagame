# Story 4.7: Fluid System Activation

Status: ready-for-dev

## Story

As a rendering engineer,
I want to enable WaterBall fluid simulation with WebGPU compute shader execution,
so that 10,000 fluid particles can simulate at 60 FPS using TSL compute shaders.

## Acceptance Criteria

1. WebGPURenderer context passed to fluid system
2. Compute shaders execute via `renderer.compute(computeNode)`
3. Compute executed BEFORE render in loop
4. TSL compute shaders for particle physics (MLS-MPM)
5. 10,000 particles at 60 FPS on desktop
6. Storage buffers: position, velocity, force
7. Boundary collision with `If()` conditions
8. Sphere constraint animation integrated with compute

## Tasks / Subtasks

- [ ] Create TSL compute module (AC: 4, 6)
  - [ ] Create `src/systems/fluid/compute/FluidCompute.ts`
  - [ ] Import TSL compute nodes from 'three/tsl'
  - [ ] Define storage buffers for particle data
  - [ ] Implement particle-to-grid (P2G) compute Fn
  - [ ] Implement grid-to-particle (G2P) compute Fn
  - [ ] Implement boundary conditions with If()
- [ ] Create compute node wrapper (AC: 2, 3)
  - [ ] Create `FluidComputeNode` class
  - [ ] Integrate with WebGPURenderer
  - [ ] Schedule compute before render
  - [ ] Handle async initialization
- [ ] Implement particle physics in TSL (AC: 7)
  - [ ] Add boundary collision detection
  - [ ] Implement sphere constraint for duet progress
  - [ ] Apply gravity and buoyancy forces
  - [ ] Update particle positions and velocities
- [ ] Integrate with main render loop (AC: 1, 3)
  - [ ] Update `main.js` to create compute node
  - [ ] Call renderer.compute() before renderer.render()
  - [ ] Pass compute results to water material
  - [ ] Handle frame timing for 60 FPS target
- [ ] Performance validation (AC: 5)
  - [ ] Profile compute shader execution time
  - [ ] Optimize for 10,000 particles
  - [ ] Target 4ms compute time budget
  - [ ] Test on mobile for 30 FPS fallback

## Dev Notes

### Existing WaterBall Fluid Context

**Current System:** `src/systems/fluid/index.ts`
- MLS-MPM (Moving Least Squares Material Point Method) simulation
- Currently uses WebGL2 or planned for WebGPU
- 10,000 particles for fluid simulation
- Sphere constraint for duet progress animation

**Compute Requirements:**
- Particle-to-grid transfer
- Grid operations (pressure solve, advection)
- Grid-to-particle transfer
- Boundary conditions
- Sphere constraint animation

### TSL Compute Shader Pattern

```typescript
import { Fn, instanceIndex, storage, attribute, float, vec3, vec2,
         uniform, add, sub, mul, div, If, Else } from 'three/tsl';

// Storage buffer declaration
const positionBuffer = storage(new Float32Array(10000 * 3), 'vec3', 10000);
const velocityBuffer = storage(new Float32Array(10000 * 3), 'vec3', 10000);
const forceBuffer = storage(new Float32Array(10000 * 3), 'vec3', 10000);
```

### Particle Update Compute

```typescript
const deltaTime = uniform(0.016); // 60 FPS = 16.67ms
const gravity = uniform(9.8);
const sphereRadius = uniform(5.0);
const sphereCenter = uniform(new THREE.Vector3(0, 0, 0));

const updateParticles = Fn(() => {
  const idx = instanceIndex;

  // Read current position and velocity
  const position = positionBuffer.element(idx);
  const velocity = velocityBuffer.element(idx);

  // Apply gravity
  const newVelocity = velocity.add(vec3(0, -gravity.mul(deltaTime), 0));

  // Update position
  const newPosition = position.add(newVelocity.mul(deltaTime));

  // Boundary collision with If()
  const boundaryX = float(10);
  const boundaryZ = float(10);

  const checkedX = If(
    newPosition.x.abs().greaterThan(boundaryX),
    {
      // Reflect velocity
      newVelocity.x.assign(newVelocity.x.negate().mul(0.5));
      // Clamp position
      newPosition.x.assign(newPosition.x.sign().mul(boundaryX));
    },
    {}
  );

  const checkedZ = If(
    newPosition.z.abs().greaterThan(boundaryZ),
    {
      newVelocity.z.assign(newVelocity.z.negate().mul(0.5));
      newPosition.z.assign(newPosition.z.sign().mul(boundaryZ));
    },
    {}
  );

  // Sphere constraint for duet progress
  const distFromCenter = distance(newPosition.xz, sphereCenter.xz);
  const maxRadius = sphereRadius;

  const constrained = If(
    distFromCenter.greaterThan(maxRadius),
    {
      // Pull toward sphere surface
      const dir = newPosition.sub(sphereCenter).normalize();
      newPosition.assign(sphereCenter.add(dir.mul(maxRadius)));

      // Reflect velocity inward
      const inwardVel = dir.mul(-1);
      newVelocity.assign(inwardVel.mul(newVelocity.dot(inwardVel).mul(0.5)));
    },
    {}
  );

  // Write back
  positionBuffer.element(idx).assign(newPosition);
  velocityBuffer.element(idx).assign(newVelocity);
});
```

### Compute Node Integration

```typescript
import { ComputeNode } from 'three/tsl';

export class FluidComputeNode extends ComputeNode {
  private timeUniform = uniform(0);
  private deltaTimeUniform = uniform(0.016);
  private sphereRadiusUniform = uniform(5.0);
  private duetProgressUniform = uniform(0);

  constructor(particleCount: number = 10000) {
    super();

    // Create compute function
    this.computeFn = updateParticles();

    // Bind uniforms
    this.computeFn.setUniform('uDeltaTime', this.deltaTimeUniform);
    this.computeFn.setUniform('uSphereRadius', this.sphereRadiusUniform);
    this.computeFn.setUniform('uDuetProgress', this.duetProgressUniform);
  }

  setTime(t: number): void {
    this.timeUniform.value = t;
  }

  setSphereRadius(radius: number): void {
    this.sphereRadiusUniform.value = radius;
  }

  setDuetProgress(progress: number): void {
    this.duetProgressUniform.value = progress;
  }

  dispatch(renderer: THREE.WebGPURenderer): void {
    // Execute compute shader
    renderer.compute(this);
  }
}
```

### Render Loop Integration

```typescript
// In main.js or fluid system update:
import { WebGPURenderer } from 'three/webgpu';

class FluidSystem {
  private computeNode: FluidComputeNode;

  constructor(
    private renderer: WebGPURenderer,
    private scene: THREE.Scene
  ) {
    this.computeNode = new FluidComputeNode(10000);
  }

  update(deltaTime: number, time: number): void {
    // Step 1: Execute compute shader (BEFORE render)
    this.computeNode.setTime(time);
    this.computeNode.dispatch(this.renderer);

    // Step 2: Render scene with updated particle data
    // Water material reads from storage buffers updated by compute
  }

  dispose(): void {
    this.computeNode.dispose();
  }
}
```

### Storage Buffer Management

```typescript
// Particle data structure
interface ParticleData {
  positions: Float32Array;  // [x, y, z, x, y, z, ...] (10000 * 3)
  velocities: Float32Array; // [vx, vy, vz, ...] (10000 * 3)
  forces: Float32Array;     // [fx, fy, fz, ...] (10000 * 3)
}

class FluidStorage {
  private particleCount: number;
  private positionBuffer: THREE.StorageBufferAttribute;
  private velocityBuffer: THREE.StorageBufferAttribute;
  private forceBuffer: THREE.StorageBufferAttribute;

  constructor(particleCount: number) {
    this.particleCount = particleCount;

    // Initialize storage buffers
    this.positionBuffer = new THREE.StorageBufferAttribute(
      particleCount * 3,
      Float32Array.BYTES_PER_ELEMENT * 3,
      THREE.Float32Type
    );

    this.velocityBuffer = new THREE.StorageBufferAttribute(
      particleCount * 3,
      Float32Array.BYTES_PER_ELEMENT * 3,
      THREE.Float32Type
    );

    this.forceBuffer = new THREE.StorageBufferAttribute(
      particleCount * 3,
      Float32Array.BYTES_PER_ELEMENT * 3,
      THREE.Float32Type
    );

    // Initialize particles in grid formation
    this.initializeParticles();
  }

  private initializeParticles(): void {
    const positions = this.positionBuffer.array as Float32Array;
    const size = Math.cbrt(this.particleCount);
    const spacing = 0.5;

    let i = 0;
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        for (let z = 0; z < z++) {
          positions[i++] = (x - size/2) * spacing;
          positions[i++] = y * spacing;
          positions[i++] = (z - size/2) * spacing;
        }
      }
    }
  }

  getPositions(): THREE.StorageBufferAttribute {
    return this.positionBuffer;
  }

  getVelocities(): THREE.StorageBufferAttribute {
    return this.velocityBuffer;
  }

  getForces(): THREE.StorageBufferAttribute {
    return this.forceBuffer;
  }
}
```

### MLS-MPM in TSL

The MLS-MPM algorithm involves multiple stages:

1. **Particle-to-Grid (P2G):** Transfer particle masses/velocities to grid
2. **Grid Update:** Compute pressure forces, update grid velocities
3. **Grid-to-Particle (G2P):** Transfer updated grid velocities back to particles
4. **Particle Update:** Apply advection and external forces

```typescript
// Simplified MLS-MPM compute stages
const p2gStage = Fn(({ particleIdx, gridResolution }) => {
  const particlePos = positionBuffer.element(particleIdx);
  const particleVel = velocityBuffer.element(particleIdx);

  // Compute grid cell coordinates
  const gridX = floor(particlePos.x.mul(gridResolution));
  const gridY = floor(particlePos.y.mul(gridResolution));
  const gridZ = floor(particlePos.z.mul(gridResolution));

  // Transfer to grid (simplified - actual MLS-MPM uses splines)
  const gridIdx = gridX.add(gridY.mul(gridResolution)).add(gridZ.mul(gridResolution).mul(gridResolution));
  gridVelocityBuffer.element(gridIdx).addAssign(particleVel);
});

const g2pStage = Fn(({ particleIdx, gridResolution }) => {
  const particlePos = positionBuffer.element(particleIdx);
  const gridX = floor(particlePos.x.mul(gridResolution));
  const gridY = floor(particlePos.y.mul(gridResolution));
  const gridZ = floor(particlePos.z.mul(gridResolution));

  const gridIdx = gridX.add(gridY.mul(gridResolution)).add(gridZ.mul(gridResolution).mul(gridResolution));
  const gridVel = gridVelocityBuffer.element(gridIdx);

  // Update particle velocity from grid
  velocityBuffer.element(particleIdx).assign(gridVel);
});
```

### File Structure Changes

**New files:**
```
src/systems/fluid/compute/
  FluidCompute.ts         # Main compute node class
  ParticleBuffers.ts       # Storage buffer management
  MLSMPMStages.ts          # TSL compute Fn stages
```

**Modified:**
```
src/systems/fluid/index.ts        # Integrate compute node
src/main.js                        # Initialize compute system
src/entities/WaterMaterial.ts     # Read from storage buffers
```

### Performance Considerations

**Target Budgets:**
- Compute time: < 4ms at 60 FPS
- Total frame: < 16ms (60 FPS) or < 33ms (30 FPS mobile)
- Memory: ~200MB for 10,000 particles (pos + vel + force)

**Optimization Strategies:**
1. Use storage buffers instead of textures for particle data
2. Minimize GPU-CPU synchronization
3. Batch compute operations
4. Use reduced precision (16-bit float) where acceptable
5. LOD: Reduce particle count on mobile

### Platform Limitations

| Platform | Compute Support | Particles | Target FPS |
|----------|----------------|----------|------------|
| Windows (GPU) | ✅ Full | 10,000 | 60 |
| macOS (M4+) | ⚠️ Limited | 5,000 | 30 |
| Mobile | ⚠️ Limited | 3,000 | 30 |
| Ubuntu | ❌ No WebGPU | N/A | N/A |

### Testing Requirements

1. Verify 10,000 particles render correctly
2. Test boundary collision at all walls
3. Verify sphere constraint animation at different duet progress levels
4. Performance: measure compute shader execution time
5. Test frame time budget (target 4ms for compute)
6. Visual validation: particles should behave like water
7. Test memory usage (target < 200MB)
8. Mobile fallback: reduce particle count to 3000

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 289-322
- Existing fluid system: `src/systems/fluid/index.ts`
- WaterBall reference: https://github.com/matsuoka-601/WaterBall
- TSL compute: `.claude/skills/three-best-practices/rules/tsl-compute-shaders.md`

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
