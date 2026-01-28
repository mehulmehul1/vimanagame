# Story 4.4: Water Material → TSL Migration

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a rendering engineer,
I want to convert the water surface shader from GLSL to TSL with Fresnel effects and bioluminescence,
so that it works with WebGPU rendering while preserving all 6-string frequency response and WaterBall-inspired effects.

## Acceptance Criteria

1. Water shader converted to TSL in `src/shaders/tsl/WaterShader.ts`
2. Fresnel effect using `dot(normalView, positionView.normalize())`
3. `pow(float(1).sub(dot(...)), float(3))` for edge intensity
4. Bioluminescent color mixing with `mix()`
5. 6-string frequency response preserved with array uniforms
6. `timerLocal()` for animation time
7. Physical material properties: transmission, roughness, IOR
8. Jelly creature ripples preserved (6 jellies × 3 positions)
9. Sphere constraint animation for duet progress

## Tasks / Subtasks

- [ ] Create TSL water shader module (AC: 1)
  - [ ] Create `src/shaders/tsl/WaterShader.ts`
  - [ ] Import TSL nodes from 'three/tsl'
  - [ ] Define Fn() functions for wave displacement
- [ ] Implement wave animation with harp response (AC: 5)
  - [ ] Convert base wave FBM to TSL
  - [ ] Implement 6-string frequency response
  - [ ] Add string wave propagation with distance decay
  - [ ] Store frequency and velocity arrays as uniforms
- [ ] Implement jelly creature ripples (AC: 8)
  - [ ] Convert jelly ripple loop to TSL
  - [ ] Add position-based ripple influence
  - [ ] Implement animated ripple ring effect
- [ ] Implement fresnel and transmission (AC: 2, 3, 7)
  - [ ] Calculate view direction
  - [ ] Implement Schlick fresnel approximation
  - [ ] Set transmission node for water transparency
  - [ ] Configure IOR (1.33 for water)
  - [ ] Set roughness for water surface
- [ ] Implement bioluminescent color mixing (AC: 4)
  - [ ] Mix deep/shallow colors based on wave height
  - [ ] Add bioluminescent glow
  - [ ] Implement harmonic resonance boost
- [ ] Implement sphere constraint for duet progress (AC: 9)
  - [ ] Convert flat-to-sphere transformation
  - [ ] Add hollow center tunnel effect
  - [ ] Blend based on duet progress uniform
- [ ] Create material class wrapper
  - [ ] Extend MeshPhysicalNodeMaterial
  - [ ] Maintain existing WaterMaterial API
  - [ ] Test visual parity with GLSL version

## Dev Notes

### Existing GLSL Shader Context

**Current Water Vertex Shader (`src/shaders/index.ts` lines 6-159):**
- Base FBM wave animation
- 6-string frequency response (uHarpFrequencies[6], uHarpVelocities[6])
- Jelly creature ripples (uJellyPositions[6], uJellyVelocities[6], uJellyActive[6])
- Sphere constraint for duet progress animation

**Current Water Fragment Shader (`src/shaders/index.ts` lines 162-304):**
- Fresnel-based transparency
- Depth-based color absorption
- Caustics animation
- Bioluminescent glow
- Environment reflections (optional)

**Current WaterMaterial (`src/entities/WaterMaterial.ts`):**
- Extends `THREE.ShaderMaterial`
- 16 uniforms including arrays for 6 strings and 6 jellies
- Methods for updating frequencies, velocities, jelly positions
- WaterBall-inspired properties (thickness, density, F0)

### TSL Import Pattern for Water

```typescript
import { Fn, positionLocal, normalLocal, uniform, uv,
         mix, distance, max, min, sin, cos, pow, float, vec3, vec2,
         normalize, dot, cameraPosition, timerLocal,
         MeshPhysicalNodeMaterial, texture, exp, smoothstep,
         abs, add, sub, mul, div, mod, floor, fract } from 'three/tsl';
```

### Wave Animation Conversion

**GLSL:**
```glsl
float baseWave = fbm(waveCoord + uTime * 0.1, 3) * 0.15;
```

**TSL:**
```typescript
const fbm = Fn(({ coord, octaves, time }) => {
  let value = float(0);
  let amplitude = float(0.5);
  let frequency = float(1);

  // Manual loop unrolling for TSL (4 octaves)
  const octave1 = perlin(coord.mul(frequency).add(time.mul(0.1))).mul(amplitude);
  const octave2 = perlin(coord.mul(frequency.mul(2)).add(time.mul(0.1))).mul(amplitude.mul(0.5));
  const octave3 = perlin(coord.mul(frequency.mul(4)).add(time.mul(0.1))).mul(amplitude.mul(0.25));
  const octave4 = perlin(coord.mul(frequency.mul(8)).add(time.mul(0.1))).mul(amplitude.mul(0.125));

  return octave1.add(octave2).add(octave3).add(octave4);
});
```

### String Frequency Response

**GLSL:**
```glsl
for (int i = 0; i < 6; i++) {
  vec2 stringOrigin = vec2(float(i) * 2.0 - 5.0, 0.0);
  float distTo = distance(pos.xz, stringOrigin);
  float stringWave = sin(distTo * 3.0 - uTime * 2.0 + uHarpFrequencies[i]) * 0.05;
  stringWave *= smoothstep(3.0, 0.0, distTo);
  stringResonance += stringWave * uHarpVelocities[i] * 0.5;
}
```

**TSL:**
```typescript
// Uniform arrays for 6 strings
const stringFrequencies = uniform(new Float32Array(6));
const stringVelocities = uniform(new Float32Array(6));
const bioluminescentColor = uniform(new THREE.Color(0x00ff88));

const stringResonance = Fn(({ position, time }) => {
  const string0 = calculateStringWave({
    position, time,
    origin: vec2(-5, 0),
    freq: stringFrequencies.element(0),
    velocity: stringVelocities.element(0)
  });

  const string1 = calculateStringWave({
    position, time,
    origin: vec2(-3, 0),
    freq: stringFrequencies.element(1),
    velocity: stringVelocities.element(1)
  });

  // ... repeat for strings 2-5

  return string0.add(string1).add(string2).add(string3).add(string4).add(string5);
});
```

### Fresnel in TSL

```typescript
const waterFresnel = Fn(({ normal, viewDir, f0 }) => {
  const cosTheta = dot(normal, viewDir.negate()).max(0);
  const x = float(1).sub(cosTheta);
  // Schlick's approximation: F0 + (1-F0) * (1-cosTheta)^5
  return f0.add(float(1).sub(f0).mul(x.pow(5)));
});

// Usage
const normalWorld = normalLocal.transformDirection(modelWorldMatrix);
const viewDir = cameraPosition.sub(positionLocal).normalize();
const fresnel = waterFresnel({
  normal: normalWorld,
  viewDir: viewDir,
  f0: float(0.02) // Water F0
});
```

### Physical Material Properties

```typescript
export class WaterMaterialTSL extends MeshPhysicalNodeMaterial {
  constructor() {
    super();

    // Physical water properties
    this.transmissionNode = float(0.8);      // Transparent
    this.iorNode = float(1.33);               // Water IOR
    this.roughnessNode = float(0.1);          // Smooth surface
    this.thicknessNode = float(1.0);          // Water depth
    this.attenuationColor = new THREE.Color(0x004488);
    this.attenuationDistance = 10;

    // Custom displacement
    this.positionNode = waterDisplacement({ time: timerLocal() });

    // Custom color
    this.colorNode = waterColor({ fresnel, time: timerLocal() });

    this.transparent = true;
    this.side = THREE.DoubleSide;
  }
}
```

### Jelly Creature Ripples

**GLSL:**
```glsl
for (int i = 0; i < 6; i++) {
  if (uJellyActive[i] > 0.5) {
    vec2 jellyXZ = uJellyPositions[i].xz;
    float dist = distance(pos.xz, jellyXZ);
    float rippleInfluence = exp(-dist * 0.8);
    float ripple = sin(dist * 8.0 - uTime * 5.0) * uJellyVelocities[i];
    jellyRipple += ripple * rippleInfluence;
  }
}
```

**TSL:**
```typescript
const jellyPositions = uniform(new Float32Array(18)); // 6 jellies × 3 (xyz)
const jellyVelocities = uniform(new Float32Array(6));
const jellyActive = uniform(new Float32Array(6));

const jellyRippleEffect = Fn(({ position, time }) => {
  // Process jelly 0
  const jelly0Active = jellyActive.element(0);
  const jelly0Pos = vec3(
    jellyPositions.element(0),
    jellyPositions.element(1),
    jellyPositions.element(2)
  );
  const ripple0 = If(jelly0Active.greaterThan(0.5),
    calculateJellyRipple({
      position,
      jellyPos: jelly0Pos,
      velocity: jellyVelocities.element(0),
      time
    }),
    float(0)
  );

  // ... repeat for jellies 1-5
  // Return sum of all ripples
});
```

### Sphere Constraint for Duet Progress

```typescript
const sphereConstraint = Fn(({ position, duetProgress, radius, center }) => {
  const centeredXZ = position.xz.sub(center.xz);
  const distFromCenter = distance(centeredXZ, vec2(0, 0));

  const pullStrength = smoothstep(0.3, 1.0, duetProgress);

  // Calculate sphere position
  const dir = centeredXZ.normalize();
  const spherePos = vec3(
    center.x.add(dir.x.mul(radius)),
    center.y,
    center.z.add(dir.y.mul(radius))
  );

  // Hollow center tunnel
  const hollowRadius = radius.mul(float(0.2).add(float(0.5).mul(float(1).sub(duetProgress))));
  const hollowFade = smoothstep(hollowRadius.mul(0.5), hollowRadius, distFromCenter);

  const dissolvedY = position.y.sub(float(10).mul(float(1).sub(hollowFade)).mul(duetProgress));

  return vec3(
    position.x,
    If(duetProgress.greaterThan(0.5).and(distFromCenter.lessThan(hollowRadius)),
      dissolvedY,
      position.y
    ),
    position.z
  );
});
```

### Uniform Update Methods

```typescript
export class WaterMaterialTSL extends MeshPhysicalNodeMaterial {
  // Uniform declarations
  private timeUniform = uniform(0);
  private stringFreqUniform = uniform(new Float32Array(6));
  private stringVelUniform = uniform(new Float32Array(6));
  private duetProgressUniform = uniform(0);
  private jellyPositionsUniform = uniform(new Float32Array(18));
  private jellyVelocitiesUniform = uniform(new Float32Array(6));
  private jellyActiveUniform = uniform(new Float32Array(6));

  // Update methods matching existing API
  setStringFrequency(index: number, frequency: number): void {
    if (index >= 0 && index < 6) {
      const arr = this.stringFreqUniform.value as Float32Array;
      arr[index] = frequency;
    }
  }

  setStringVelocity(index: number, velocity: number): void {
    if (index >= 0 && index < 6) {
      const arr = this.stringVelUniform.value as Float32Array;
      arr[index] = velocity;
    }
  }

  setJellyPosition(index: number, position: THREE.Vector3): void {
    if (index >= 0 && index < 6) {
      const arr = this.jellyPositionsUniform.value as Float32Array;
      arr[index * 3 + 0] = position.x;
      arr[index * 3 + 1] = position.y;
      arr[index * 3 + 2] = position.z;
    }
  }

  decayVelocities(deltaTime: number, decayRate: number = 2.0): void {
    const velocities = this.stringVelUniform.value as Float32Array;
    for (let i = 0; i < 6; i++) {
      if (velocities[i] > 0) {
        velocities[i] = Math.max(0, velocities[i] - decayRate * deltaTime);
      }
    }
  }

  // ... other methods matching existing API
}
```

### Backward Compatibility

Maintain full compatibility with existing `WaterMaterial` class:

```typescript
// src/entities/WaterMaterial.ts
// Keep all existing methods working the same way
export class WaterMaterial extends MeshPhysicalNodeMaterial {
  setTime(time: number): void { /* ... */ }
  setStringFrequency(index: number, freq: number): void { /* ... */ }
  setStringVelocity(index: number, vel: number): void { /* ... */ }
  setDuetProgress(progress: number): void { /* ... */ }
  setHarmonicResonance(resonance: number): void { /* ... */ }
  setShipPatience(patience: number): void { /* ... */ }
  setCameraPosition(pos: THREE.Vector3): void { /* ... */ }
  triggerStringRipple(index: number, intensity: number): void { /* ... */ }
  decayVelocities(dt: number, rate?: number): void { /* ... */ }
  // ... WaterBall methods
  setSphereRadius(radius: number): void { /* ... */ }
  setSphereCenter(center: THREE.Vector3): void { /* ... */ }
  setThickness(thickness: number): void { /* ... */ }
  setDensity(density: number): void { /* ... */ }
  setF0(f0: number): void { /* ... */ }
  // ... jelly methods
  setJellyPosition(index: number, pos: THREE.Vector3): void { /* ... */ }
  setJellyVelocity(index: number, vel: number): void { /* ... */ }
  setJellyActive(index: number, active: boolean): void { /* ... */ }
  updateJellyPositions(jellyManager: any): void { /* ... */ }
}
```

### File Structure Changes

**New file:**
```
src/shaders/tsl/WaterShader.ts   # TSL implementation
```

**Modified:**
```
src/entities/WaterMaterial.ts    # Convert to use TSL
src/shaders/index.ts              # Export TSL version
```

### Testing Requirements

1. Visual parity with GLSL version at all duet progress levels
2. Test all 6 string frequency responses individually
3. Verify jelly ripple effects for all 6 jellies
4. Test sphere constraint animation (0% → 100% duet progress)
5. Confirm fresnel edge glow intensity matches
6. Performance: no regression vs GLSL
7. Test water transmission and IOR visual appearance

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 142-176
- Existing water shader: `src/shaders/index.ts` lines 6-304
- Existing material: `src/entities/WaterMaterial.ts`
- WaterBall reference: https://github.com/matsuoka-601/WaterBall
- TSL compute reference: `.claude/skills/three-best-practices/rules/tsl-compute-shaders.md`

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
