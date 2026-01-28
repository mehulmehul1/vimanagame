# Story 4.6: Jelly Shader → TSL Migration

Status: ready-for-dev

## Story

As a rendering engineer,
I want to convert the bioluminescent jelly creature shader from GLSL to TSL (Three.js Shading Language),
so that the jelly creatures work with WebGPU rendering while maintaining their organic pulsing and teaching animations.

## Acceptance Criteria

1. Jelly shader converted to TSL in `src/shaders/tsl/JellyShader.ts`
2. Organic pulsing animation using `sin(timerLocal())`
3. Bioluminescent glow with teaching state enhancement
4. Per-jelly frequency variations via uniform
5. Emissive material with `emissiveNode`
6. Fresnel-based rim lighting effect
7. Simplex noise displacement for organic movement
8. Bell-shaped pulse enhancement at top of jelly

## Tasks / Subtasks

- [ ] Create TSL jelly shader module (AC: 1)
  - [ ] Create `src/shaders/tsl/JellyShader.ts`
  - [ ] Import TSL nodes from 'three/tsl'
  - [ ] Define Fn() for simplex noise calculation
- [ ] Implement vertex displacement in TSL (AC: 2, 7, 8)
  - [ ] Convert simplex noise to TSL
  - [ ] Apply noise-based displacement to positionLocal
  - [ ] Add pulse animation using sin(timerLocal())
  - [ ] Implement bell-shaped enhancement based on position.y
- [ ] Implement fragment shader in TSL (AC: 5, 6)
  - [ ] Calculate fresnel effect using dot() and normalize()
  - [ ] Create emissive node with bioluminescent color
  - [ ] Mix base color with glow based on teaching state
  - [ ] Apply rim lighting with fresnel
- [ ] Create material class wrapper (AC: 3, 4)
  - [ ] Extend MeshStandardNodeMaterial
  - [ ] Maintain existing JellyCreature API
  - [ ] Support setPulseRate(), setColor(), setCameraPosition()
  - [ ] Add setTeachingIntensity() method
- [ ] Update JellyCreature entity (AC: 4)
  - [ ] Import and use JellyMaterialTSL instead of ShaderMaterial
  - [ ] Update uniform setting methods to match TSL API
  - [ ] Test visual parity with GLSL version

## Dev Notes

### Existing GLSL Shader Context

**Current Jelly Vertex Shader (`src/shaders/index.ts` lines 433-503):**
```glsl
uniform float uTime;
uniform float uPulseRate;
uniform float uIsTeaching;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec2 vUv;
varying float vPulse;

// Simplex noise function (snoise)
void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    float pulse = sin(uTime * uPulseRate) * 0.5 + 0.5;
    vPulse = pulse;
    float noise = snoise(position * 2.0 + uTime * 0.5) * 0.1;
    float teachingEnhancement = 1.0 + uIsTeaching * 0.5;
    vec3 newPosition = position;
    newPosition += normal * (pulse * 0.15 + noise) * teachingEnhancement;
    float bellFactor = smoothstep(-0.5, 0.5, position.y);
    newPosition += normal * bellFactor * pulse * 0.1;
    // ...
}
```

**Current Jelly Fragment Shader (`src/shaders/index.ts` lines 505-535):**
```glsl
uniform float uTime;
uniform float uPulseRate;
uniform float uIsTeaching;
uniform float uTeachingIntensity;
uniform vec3 uBioluminescentColor;
uniform vec3 uCameraPosition;

void main() {
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0);
    vec3 baseColor = vec3(0.4, 0.8, 0.7);
    float glowIntensity = 0.3 + vPulse * 0.4 + uTeachingIntensity * 0.3;
    vec3 glow = uBioluminescentColor * glowIntensity;
    // Teaching enhancement adds warm color
    if (uIsTeaching > 0.5) {
        glow += vec3(0.2, 0.1, 0.0) * uTeachingIntensity;
    }
    // Rim lighting with fresnel
    vec3 rimColor = uBioluminescentColor * 2.0;
    finalColor += rimColor * fresnel * 0.5;
}
```

### Existing JellyCreature Entity

**File:** `src/entities/JellyCreature.ts`

**Key uniforms:**
- `uTime` - Animation time
- `uPulseRate` - Pulsing frequency (default 2.0)
- `uIsTeaching` - Boolean flag for teaching state (0 or 1)
- `uTeachingIntensity` - Intensity during teaching (0 to 1)
- `uBioluminescentColor` - Color #88ccff (soft cyan-blue)
- `uCameraPosition` - For fresnel calculation

**Methods to maintain:**
- `setPulseRate(rate: number)` - Set pulse frequency
- `setColor(color: THREE.Color)` - Set bioluminescent color
- `setCameraPosition(position: THREE.Vector3)` - Update fresnel
- `beginTeaching()` - Enter teaching state
- `submerge()` - Exit teaching state

### TSL Import Pattern

```typescript
import { Fn, positionLocal, normalLocal, uniform, timerLocal,
         mix, distance, max, min, sin, cos, pow, float, vec3, vec2,
         normalize, dot, cameraPosition, localToWorld, normalWorld,
         MeshStandardNodeMaterial, smoothstep, abs, add, sub, mul, div } from 'three/tsl';
```

### Simplex Noise in TSL

```typescript
// Simplified 3D simplex noise for TSL
const snoise3 = Fn(({ p }) => {
  const C = vec2(1.0/6.0, 1.0/3.0);
  const D = vec3(0.0, 0.5, 1.0);

  const i = floor(p.add(p.dot(C.yyy)));
  const x0 = p.sub(i).add(i.dot(C.xxx));

  const g = step(x0.yzx(), x0.xyz());
  const l = float(1).sub(g);
  const i1 = min(g.xyz(), l.zxy());
  const i2 = max(g.xyz(), l.zxy());

  // Simplified noise calculation
  const noiseValue = sin(p.x.mul(10).add(p.y.mul(10)).add(p.z.mul(10))).mul(0.5).add(0.5);

  return noiseValue;
});
```

### Jelly Pulse Animation

```typescript
const jellyPulse = Fn(({ time, pulseRate }) => {
  // Organic pulse: sin(time * rate) * 0.5 + 0.5
  return sin(time.mul(pulseRate)).mul(0.5).add(0.5);
});
```

### Fresnel Effect in TSL

```typescript
const jellyFresnel = Fn(({ normal, viewDir }) => {
  const cosTheta = dot(normal, viewDir);
  // Fresnel: (1 - dot)²
  return float(1).sub(cosTheta.abs()).pow(2);
});
```

### Bell-Shaped Enhancement

```typescript
const bellFactor = Fn(({ position }) => {
  // smoothstep(-0.5, 0.5, position.y) creates bell curve along Y axis
  return smoothstep(float(-0.5), float(0.5), position.y);
});
```

### Material Class

```typescript
export class JellyMaterialTSL extends MeshStandardNodeMaterial {
  private timeUniform = uniform(0);
  private pulseRateUniform = uniform(2.0);
  private isTeachingUniform = uniform(0);
  private teachingIntensityUniform = uniform(0);
  private bioluminescentColorUniform = uniform(new THREE.Color(0x88ccff));
  private cameraPositionUniform = uniform(new THREE.Vector3());

  constructor() {
    super();

    const time = this.timeUniform;
    const pulseRate = this.pulseRateUniform;
    const isTeaching = this.isTeachingUniform;
    const teachingIntensity = this.teachingIntensityUniform;
    const bioColor = this.bioluminescentColorUniform;

    // Calculate pulse
    const pulse = sin(time.mul(pulseRate)).mul(0.5).add(0.5);

    // Vertex displacement with noise
    const noise = snoise3({ p: positionLocal.mul(2).add(time.mul(0.5)) }).mul(0.1);
    const teachingEnhancement = float(1).add(isTeaching.mul(0.5));
    const displacement = normalLocal
      .mul(pulse.mul(0.15).add(noise))
      .mul(teachingEnhancement);

    // Bell factor for top enhancement
    const bell = smoothstep(float(-0.5), float(0.5), positionLocal.y);
    const bellDisplacement = normalLocal.mul(bell.mul(pulse).mul(0.1));

    // Combined displacement
    this.positionNode = displacement.add(bellDisplacement);

    // Fragment - fresnel and glow
    const normalWorld = normalLocal.transformDirection(modelWorldMatrix);
    const viewDir = cameraPosition.sub(localToWorld(positionLocal)).normalize();
    const fresnel = float(1).sub(dot(normalWorld, viewDir).abs()).pow(2);

    const baseColor = vec3(0.4, 0.8, 0.7);
    const glowIntensity = float(0.3).add(pulse.mul(0.4)).add(teachingIntensity.mul(0.3));
    const glow = bioColor.mul(glowIntensity);

    // Teaching enhancement (warm color added)
    const teachingGlow = vec3(0.2, 0.1, 0).mul(teachingIntensity).mul(isTeaching);

    const internalVisibility = float(1).sub(fresnel.mul(0.5));
    const internalColor = vec3(0.8, 0.9, 0.85).mul(internalVisibility).mul(0.3);

    const finalColor = baseColor.mul(0.4).add(glow).add(internalColor).add(teachingGlow);

    // Rim lighting
    const rimColor = bioColor.mul(2);
    const rimLight = rimColor.mul(fresnel).mul(0.5);

    this.emissiveNode = finalColor.add(rimLight);

    // Transparent with fresnel-based alpha
    this.transparentNode = float(0.5).add(fresnel.mul(0.3)).add(pulse.mul(0.1));
    this.transparent = true;
    this.side = THREE.DoubleSide;
  }

  setTime(t: number): void {
    this.timeUniform.value = t;
  }

  setPulseRate(rate: number): void {
    this.pulseRateUniform.value = rate;
  }

  setTeaching(isTeaching: boolean): void {
    this.isTeachingUniform.value = isTeaching ? 1.0 : 0.0;
  }

  setTeachingIntensity(intensity: number): void {
    this.teachingIntensityUniform.value = Math.max(0, Math.min(1, intensity));
  }

  setColor(color: THREE.Color): void {
    this.bioluminescentColorUniform.value.copy(color);
  }

  setCameraPosition(position: THREE.Vector3): void {
    this.cameraPositionUniform.value.copy(position);
  }
}
```

### Integration with JellyCreature Entity

```typescript
// src/entities/JellyCreature.ts
import { JellyMaterialTSL } from '../shaders/tsl/JellyShader';

export class JellyCreature extends THREE.Mesh {
  private material: JellyMaterialTSL;

  constructor(spawnPosition: THREE.Vector3 = new THREE.Vector3(0, 0, 0), noteIndex: number = 0) {
    const geometry = new THREE.SphereGeometry(0.3, 64, 64);
    const material = new JellyMaterialTSL();

    super(geometry, material);
    this.material = material;

    // Initialize with default pulse rate
    this.material.setPulseRate(2.0);
  }

  public update(deltaTime: number, time: number): void {
    this.material.setTime(time);
    // ... rest of update logic
  }

  public beginTeaching(): void {
    this.material.setTeaching(true);
    this.material.setTeachingIntensity(1.0);
    // ... other teaching logic
  }

  public submerge(): void {
    this.material.setTeaching(false);
    // ... other submerging logic
  }

  public setPulseRate(rate: number): void {
    this.material.setPulseRate(rate);
  }

  public setColor(color: THREE.Color): void {
    this.material.setColor(color);
  }

  public setCameraPosition(position: THREE.Vector3): void {
    this.material.setCameraPosition(position);
  }

  // ... other methods
}
```

### File Structure Changes

**New file:**
```
src/shaders/tsl/JellyShader.ts       # TSL implementation
```

**Modified:**
```
src/entities/JellyCreature.ts         # Use JellyMaterialTSL
src/shaders/index.ts                  # Export TSL version
```

### Testing Requirements

1. Visual comparison: TSL output must match GLSL version
2. Test pulsing animation at different rates (1.0, 2.0, 3.0)
3. Verify teaching state enhances glow intensity
4. Check fresnel rim lighting at different view angles
5. Confirm organic noise displacement is visible
6. Test bell-shaped enhancement (top pulses more than bottom)
7. Performance: no regression vs GLSL
8. Test all 6 jelly creatures with individual pulse rates

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 264-286
- Existing jelly shader: `src/shaders/index.ts` lines 433-535
- Existing entity: `src/entities/JellyCreature.ts`
- TSL reference: `.claude/skills/three-best-practices/rules/tsl-complete-reference.md`

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
