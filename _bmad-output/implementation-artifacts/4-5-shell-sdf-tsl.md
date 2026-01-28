# Story 4.5: Shell SDF → TSL Migration

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a rendering engineer,
I want to convert the procedural SDF shell shader from GLSL to TSL,
so that the nautilus spiral collectible shells work with WebGPU rendering.

## Acceptance Criteria

1. Nautilus spiral SDF converted to TSL in `src/shaders/tsl/ShellShader.ts`
2. Iridescence effect using view angle (`dot(viewDir, normal)`)
3. Simplex noise dissolve effect
4. 3-second appear animation with easing
5. Easing functions in TSL (`smoothstep()`)
6. Bobbing idle animation
7. Fresnel-based edge glow
8. Spiral pattern visualization on shell surface

## Tasks / Subtasks

- [ ] Create TSL shell shader module (AC: 1)
  - [ ] Create `src/shaders/tsl/ShellShader.ts`
  - [ ] Import TSL nodes from 'three/tsl'
  - [ ] Define Fn() for SDF calculations
- [ ] Implement nautilus spiral SDF (AC: 1, 8)
  - [ ] Convert spiral formula to TSL
  - [ ] Apply spiral pattern to surface
  - [ ] Generate UV coordinates from position
- [ ] Implement iridescence effect (AC: 2, 7)
  - [ ] Calculate view angle
  - [ ] Mix colors based on angle
  - [ ] Apply fresnel edge glow
- [ ] Implement dissolve effect (AC: 3, 4)
  - [ ] Convert simplex noise to TSL
  - [ ] Apply dissolve threshold
  - [ ] Animate dissolve over 3 seconds
- [ ] Implement appear animation (AC: 5)
  - [ ] Add smoothstep for easing
  - [ ] Scale from 0 to 1 over 3 seconds
  - [ ] Add bobbing motion
- [ ] Create material class wrapper (AC: 6)
  - [ ] Extend MeshStandardNodeMaterial
  - [ ] Maintain existing API
  - [ ] Test visual parity

## Dev Notes

### Existing GLSL Shader Context

**Current Shell Vertex Shader (`src/shaders/index.ts` lines 541-558):**
```glsl
uniform float uTime;
uniform float uAppearProgress;
uniform float uDissolveAmount;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vLocalPosition;
```

**Current Shell Fragment Shader (`src/shaders/index.ts` lines 561-646):**
```glsl
uniform float uTime;
uniform float uAppearProgress;
uniform float uDissolveAmount;

// Iridescence with 5 colors
vec3 iridescence(vec3 viewDir, vec3 normal, float time) {
  vec3 colors[5];
  colors[0] = vec3(1.0, 0.8, 0.6);  // Peach
  colors[1] = vec3(0.8, 0.6, 1.0);  // Lavender
  colors[2] = vec3(0.6, 0.9, 1.0);  // Sky blue
  colors[3] = vec3(1.0, 0.7, 0.5);  // Orange
  colors[4] = vec3(0.7, 1.0, 0.8);  // Mint
  float viewAngle = dot(viewDir, normal);
  float t = sin(viewAngle * 5.0 + time) * 0.5 + 0.5;
  // ... mix between colors based on t
}

// Spiral pattern
float angle = atan(vLocalPosition.z, vLocalPosition.x);
float radius = length(vLocalPosition.xz);
float spiral = sin(angle * 3.0 + radius * 5.0) * 0.5 + 0.5;

// Dissolve with noise
float dissolve = snoise(vLocalPosition * 3.0 + uTime) * 0.5 + 0.5;
float alpha = smoothstep(uDissolveAmount - 0.1, uDissolveAmount + 0.1, dissolve);
alpha *= uAppearProgress;
```

### Shell Collectible Context

**Existing Entity (`src/entities/ShellCollectible.ts`):**
- Shell collectibles appear after jelly teaches a note
- 3-second appear animation
- Dissolve effect based on distance from player
- Iridescent material with spiral pattern

**ShellManager Integration:**
- Spawns shells at specific positions
- Triggers appear animation when unlocked
- Handles collection on player approach

### TSL Import Pattern

```typescript
import { Fn, positionLocal, normalLocal, uniform, uv,
         mix, distance, max, min, sin, cos, pow, float, vec3, vec2,
         normalize, dot, cameraPosition, timerLocal,
         MeshStandardNodeMaterial, smoothstep, atan, length, abs,
         add, sub, mul, div, fract, floor, mod } from 'three/tsl';
```

### Simplex Noise in TSL

```typescript
// Simplex noise function for TSL
const snoise3 = Fn(({ p }) => {
  // Simplified 3D simplex noise for TSL
  // Based on Perlin's improved noise
  const C = vec2(1.0/6.0, 1.0/3.0);
  const D = vec3(0.0, 0.5, 1.0);

  const i = floor(p.add(p.dot(C.yyy)));
  const x0 = p.sub(i).add(i.dot(C.xxx));

  const g = step(x0.yzx(), x0.xyz());
  const l = float(1).sub(g);
  const i1 = min(g.xyz(), l.zxy());
  const i2 = max(g.xyz(), l.zxy());

  // ... rest of noise calculation

  return noiseValue;
});
```

### Iridescence in TSL

```typescript
const shellIridescence = Fn(({ viewDir, normal, time }) => {
  // Define 5 iridescent colors
  const color0 = vec3(1.0, 0.8, 0.6);  // Peach
  const color1 = vec3(0.8, 0.6, 1.0);  // Lavender
  const color2 = vec3(0.6, 0.9, 1.0);  // Sky blue
  const color3 = vec3(1.0, 0.7, 0.5);  // Orange
  const color4 = vec3(0.7, 1.0, 0.8);  // Mint

  // Calculate view angle
  const viewAngle = dot(viewDir, normal);
  const t = sin(viewAngle.mul(5).add(time)).mul(0.5).add(0.5);

  // Mix through colors based on t
  const t4 = t.mul(4);
  const index = floor(t4);
  const localT = fract(t4);

  // Manual color interpolation (5 colors)
  const c01 = mix(color0, color1, localT);
  const c12 = mix(color1, color2, localT);
  const c23 = mix(color2, color3, localT);
  const c34 = mix(color3, color4, localT);

  // Select based on index
  const result1 = If(index.lessThan(1), c01, c12);
  const result2 = If(index.lessThan(2), result1, c23);
  return If(index.lessThan(3), result2, c34);
});
```

### Spiral Pattern

```typescript
const shellSpiralPattern = Fn(({ position }) => {
  // Calculate angle and radius from center
  const angle = atan(position.z, position.x);
  const radius = length(position.xz);

  // Create spiral pattern
  const spiral = sin(angle.mul(3).add(radius.mul(5))).mul(0.5).add(0.5);

  return spiral;
});
```

### Dissolve Effect

```typescript
const shellDissolve = Fn(({ position, time, dissolveAmount }) => {
  // Generate noise for dissolve
  const noise = snoise3({ p: position.mul(3).add(time) });
  const noiseValue = noise.mul(0.5).add(0.5);

  // Apply dissolve threshold
  const alpha = smoothstep(
    dissolveAmount.sub(0.1),
    dissolveAmount.add(0.1),
    noiseValue
  );

  return alpha;
});
```

### Appear Animation with Easing

```typescript
const shellAppear = Fn(({ appearProgress, time }) => {
  // 3-second appear animation with smoothstep easing
  // appearProgress goes from 0 to 1 over 3 seconds

  // Apply smoothstep for ease-in-ease-out
  const easedProgress = smoothstep(0, 1, appearProgress);

  // Bobbing motion: sine wave on Y
  const bobOffset = sin(time.mul(2)).mul(0.05);

  return vec3(
    easedProgress,  // Scale X
    easedProgress.add(bobOffset),  // Scale Y with bob
    easedProgress   // Scale Z
  );
});
```

### Fresnel Edge Glow

```typescript
const shellFresnel = Fn(({ normal, viewDir }) => {
  const cosTheta = dot(normal, viewDir);
  const fresnel = pow(float(1).sub(cosTheta.abs()), 3);
  return fresnel;
});
```

### Material Class

```typescript
export class ShellMaterialTSL extends MeshStandardNodeMaterial {
  private timeUniform = uniform(0);
  private appearProgressUniform = uniform(0);
  private dissolveAmountUniform = uniform(0);
  private cameraPositionUniform = uniform(new THREE.Vector3());

  constructor() {
    super();

    const time = this.timeUniform;
    const appearProgress = this.appearProgressUniform;
    const dissolveAmount = this.dissolveAmountUniform;
    const cameraPos = this.cameraPositionUniform;

    // Get view direction
    const positionWorld = localToWorld(positionLocal);
    const viewDir = cameraPos.sub(positionWorld).normalize();
    const normalWorld = normalLocal.transformDirection(modelWorldMatrix);

    // Calculate effects
    const iridescence = shellIridescence({ viewDir, normal: normalWorld, time });
    const spiral = shellSpiralPattern({ position: positionLocal });
    const fresnel = shellFresnel({ normal: normalWorld, viewDir });
    const dissolve = shellDissolve({
      position: positionLocal,
      time,
      dissolveAmount
    });

    // Base color with spiral pattern
    const baseColor = vec3(0.95, 0.9, 0.85); // Cream base
    const spiralColor = mix(baseColor, iridescence, spiral.mul(0.6).add(fresnel.mul(0.4)));

    // Final color with fresnel glow
    this.colorNode = spiralColor.add(iridescence.mul(fresnel.mul(0.3)));

    // Alpha with dissolve and appear progress
    this.transparentNode = dissolve.mul(appearProgress);
    this.transparent = true;
    this.side = THREE.DoubleSide;
  }

  setTime(t: number): void {
    this.timeUniform.value = t;
  }

  setAppearProgress(progress: number): void {
    this.appearProgressUniform.value = Math.max(0, Math.min(1, progress));
  }

  setDissolveAmount(amount: number): void {
    this.dissolveAmountUniform.value = amount;
  }

  setCameraPosition(position: THREE.Vector3): void {
    this.cameraPositionUniform.value.copy(position);
  }
}
```

### Animation Timing

```typescript
// In ShellCollectible entity
private appearDuration = 3.0; // seconds
private appearTime = 0;

update(deltaTime: number, time: number): void {
  if (this.state === 'appearing') {
    this.appearTime += deltaTime;
    const progress = Math.min(this.appearTime / this.appearDuration, 1.0);

    // Apply smoothstep easing
    const easedProgress = progress * progress * (3 - 2 * progress);

    this.material.setAppearProgress(easedProgress);
    this.material.setTime(time);

    if (progress >= 1.0) {
      this.state = 'idle';
    }
  } else if (this.state === 'idle') {
    // Bobbing animation
    this.material.setTime(time);
  }
}
```

### File Structure Changes

**New file:**
```
src/shaders/tsl/ShellShader.ts    # TSL implementation
```

**Modified:**
```
src/entities/ShellCollectible.ts  # Update to use TSL material
src/shaders/index.ts               # Export TSL version
```

### Testing Requirements

1. Visual comparison: TSL output must match GLSL version
2. Test 3-second appear animation timing
3. Verify dissolve threshold at different amounts
4. Check iridescence color shifts at different view angles
5. Confirm spiral pattern is visible on shell surface
6. Test bobbing animation during idle state
7. Performance: no regression vs GLSL

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 178-204
- Existing shell shader: `src/shaders/index.ts` lines 541-646
- Existing entity: `src/entities/ShellCollectible.ts`
- TSL reference: `.claude/skills/three-best-practices/rules/tsl-complete-reference.md`

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
