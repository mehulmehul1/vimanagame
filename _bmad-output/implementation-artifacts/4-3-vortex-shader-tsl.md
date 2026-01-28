# Story 4.3: Vortex Shader → TSL Migration

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a rendering engineer,
I want to convert the vortex sphere constraint shader from GLSL to TSL (Three.js Shading Language),
so that it outputs WGSL for WebGPU rendering while maintaining the same visual behavior as the GLSL version.

## Acceptance Criteria

1. Vortex shader converted to TSL using `Fn()` pattern in `src/shaders/tsl/VortexShader.ts`
2. Sphere constraint logic implemented in TSL nodes
3. `positionLocal.sub(center)` for vertex transformation
4. `mix()`, `distance()`, `max()` TSL math functions
5. `If()` conditional for tunnel radius constraint
6. Activation-based animation preserved (uVortexProgress)
7. Visual parity with GLSL version (spin, breathe, glow)
8. Fresnel effect using `dot()` and `normalize()`

## Tasks / Subtasks

- [ ] Create TSL vortex shader module (AC: 1)
  - [ ] Create `src/shaders/tsl/VortexShader.ts`
  - [ ] Import TSL nodes from 'three/tsl'
  - [ ] Define `Fn()` based shader functions
- [ ] Implement vertex displacement in TSL (AC: 2, 3, 5)
  - [ ] Convert spin animation to TSL
  - [ ] Convert breathing displacement to TSL
  - [ ] Implement sphere constraint with If() conditional
  - [ ] Apply activation-based scaling
- [ ] Implement fragment shader in TSL (AC: 4, 8)
  - [ ] Convert fresnel calculation to TSL
  - [ ] Implement color mixing with mix()
  - [ ] Apply distance-based effects
  - [ ] Implement swirl noise in TSL
- [ ] Create material class wrapper (AC: 6, 7)
  - [ ] Extend MeshStandardNodeMaterial
  - [ ] Expose uniform setters for activation, duet progress
  - [ ] Maintain existing VortexMaterial API
  - [ ] Test visual parity with GLSL version
- [ ] Integrate with VortexSystem (AC: 7)
  - [ ] Update VortexSystem to use TSL material
  - [ ] Verify activation levels work correctly
  - [ ] Test visual output matches original

## Dev Notes

### Existing GLSL Shader Context

**Current Vortex Vertex Shader (`src/shaders/index.ts` lines 307-333):**
```glsl
uniform float uTime;
uniform float uVortexActivation;
uniform float uDuetProgress;

void main() {
  float spinSpeed = 1.0 + uVortexActivation * 3.0;
  float angle = uTime * spinSpeed;
  float breathe = sin(uTime * 2.0) * 0.05 * (0.5 + uVortexActivation * 0.5);
  float dist = length(position.xz);
  float swirlAngle = angle * (1.0 - dist * 0.3);
  mat2 rotation = mat2(cos(swirlAngle), -sin(swirlAngle), sin(swirlAngle), cos(swirlAngle));
  vec3 newPosition = position;
  newPosition.xz = rotation * position.xz;
  newPosition += normal * breathe;
  newPosition += normal * uVortexActivation * 0.1;
  // ...
}
```

**Current Vortex Fragment Shader (`src/shaders/index.ts` lines 336-426):**
```glsl
uniform float uTime;
uniform float uVortexActivation;
uniform vec3 uInnerColor;    // 0x00ffff (Cyan)
uniform vec3 uOuterColor;    // 0x8800ff (Purple)
uniform vec3 uCoreColor;     // 0xffffff (White)

void main() {
  vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
  float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0);
  // Color mixing, glow, core intensity...
}
```

### Existing Material Class

**`src/entities/VortexMaterial.ts`:**
- Extends `THREE.ShaderMaterial`
- Has uniforms: uTime, uVortexActivation, uDuetProgress
- Methods: `setTime()`, `setActivation()`, `setDuetProgress()`, `setCameraPosition()`
- Used by `VortexSystem` entity

### TSL Import Pattern

```typescript
import { Fn, positionLocal, normalLocal, uniform, uv,
         mix, distance, max, min, sin, cos, pow, float, vec3, vec2,
         normalize, dot, cameraPosition, localToWorld, normalWorld,
         MeshStandardNodeMaterial, If, cond, timerLocal } from 'three/tsl';
```

### TSL Shader Structure Pattern

```typescript
// src/shaders/tsl/VortexShader.ts

// Vertex displacement function
const vortexDisplacement = Fn(({ time, activation }) => {
  const spinSpeed = float(1).add(activation.mul(3));
  const angle = time.mul(spinSpeed);
  const breathe = sin(time.mul(2)).mul(0.05).mul(float(0.5).add(activation.mul(0.5)));
  const dist = distance(positionLocal.xz, vec2(0, 0));
  const swirlAngle = angle.mul(float(1).sub(dist.mul(0.3)));

  // Apply rotation to XZ
  const cosAngle = cos(swirlAngle);
  const sinAngle = sin(swirlAngle);
  const newX = positionLocal.x.mul(cosAngle).sub(positionLocal.z.mul(sinAngle));
  const newZ = positionLocal.x.mul(sinAngle).add(positionLocal.z.mul(cosAngle));

  // Add breathing and activation displacement
  const displacement = normalLocal.mul(breathe).add(normalLocal.mul(activation.mul(0.1)));

  return vec3(newX, positionLocal.y.add(displacement), newZ);
});

// Fresnel calculation
const vortexFresnel = Fn(() => {
  const viewDir = cameraPosition.sub(localToWorld(positionLocal)).normalize();
  const normal = normalWorld;
  const dotProd = dot(normal.normalize(), viewDir);
  return float(1).sub(dotProd.abs()).pow(2);
});
```

### Material Class Conversion

**Before (ShaderMaterial):**
```typescript
export class VortexMaterial extends THREE.ShaderMaterial {
  public uniforms: {
    uTime: { value: number };
    uVortexActivation: { value: number };
    // ...
  };
}
```

**After (NodeMaterial):**
```typescript
export class VortexMaterialTSL extends MeshStandardNodeMaterial {
  private timeUniform = uniform(0);
  private activationUniform = uniform(0);
  private duetProgressUniform = uniform(0);

  constructor() {
    super();

    // Set TSL material properties
    this.positionNode = vortexDisplacement({
      time: this.timeUniform,
      activation: this.activationUniform
    });

    this.emissiveNode = vortexEmissive({
      fresnel: vortexFresnel(),
      activation: this.activationUniform,
      time: this.timeUniform
    });

    this.transparent = true;
    this.side = THREE.DoubleSide;
  }

  setTime(time: number) { this.timeUniform.value = time; }
  setActivation(value: number) { this.activationUniform.value = value; }
  // ... maintain same API as original
}
```

### TSL Conditional Pattern

```typescript
// Sphere constraint with tunnel radius check
const sphereConstraint = Fn(({ position, center, radius }) => {
  const distFromCenter = distance(position.xz, center.xz);
  const hollowRadius = radius.mul(0.3);

  return If(distFromCenter.lessThan(hollowRadius),
    // Inside hollow center - dissolve
    position.y.sub(10),
    // Outside - keep normal
    position.y
  );
});
```

### Color Mixing Pattern

```typescript
const vortexColor = Fn(({ activation, innerColor, outerColor, coreColor }) => {
  return If(activation.lessThan(0.5),
    // Mix inner → outer
    mix(innerColor, outerColor, activation.mul(2)),
    // Mix outer → core
    mix(outerColor, coreColor, activation.sub(0.5).mul(2))
  );
});
```

### Uniform Declarations in TSL

```typescript
// Create uniform with initial value
const timeUniform = uniform(0);
const activationUniform = uniform(0);

// Use in TSL functions
const animated = positionLocal.add(normalLocal.mul(sin(timeUniform).mul(0.1)));

// Update from JS
timeUniform.value = clock.getElapsedTime();
```

### Backward Compatibility

Maintain existing `VortexMaterial` class interface:

```typescript
// src/entities/VortexMaterial.ts
export class VortexMaterial extends MeshStandardNodeMaterial {
  // Keep existing method signatures
  setTime(time: number): void { /* ... */ }
  setActivation(activation: number): void { /* ... */ }
  setDuetProgress(progress: number): void { /* ... */ }
  setCameraPosition(position: THREE.Vector3): void { /* ... */ }
}
```

### File Structure Changes

**New file:**
```
src/shaders/tsl/VortexShader.ts    # TSL implementation
```

**Modified:**
```
src/entities/VortexMaterial.ts     # Convert to use TSL
src/entities/VortexSystem.ts       # Update import if needed
src/shaders/index.ts               # Export TSL version
```

### Testing Requirements

1. Visual comparison: TSL output must match GLSL version
2. Test activation levels: 0.0, 0.5, 1.0
3. Verify spin animation speed matches
4. Check breathing displacement intensity
5. Confirm fresnel edge glow works
6. Test color transitions at different activation values
7. Performance: no regression vs GLSL version

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 109-140
- Existing vortex shader: `src/shaders/index.ts` lines 307-426
- Existing material: `src/entities/VortexMaterial.ts`
- Three.js TSL reference: https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language
- TSL complete reference: `.claude/skills/three-best-practices/rules/tsl-complete-reference.md`

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
