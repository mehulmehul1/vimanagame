# Story 1.2: Jelly Creatures - Musical Messengers

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player entering the Archive of Voices (Music Room)**,
I want **procedural jelly creatures to emerge from the water and demonstrate which strings to play**,
so that **I learn the ship's song through visual guidance rather than text instructions**.

## Acceptance Criteria

1. [ ] JellyCreature class with enhanced shader material featuring organic pulsing animation
2. [ ] Each jelly has unique pulse rate based on its associated note (C=1.0Hz, D=1.1Hz, E=1.2Hz, F=1.3Hz, G=1.4Hz, A=1.5Hz)
3. [ ] Bioluminescent glow intensifies when jelly is in "teaching" state (base 0.3, teaching 1.0)
4. [ ] Jump-out animation from water surface with arc trajectory (emerge → peak → land on string)
5. [ ] Target string creates ripple visualization when jelly demonstrates it
6. [ ] Smooth emergence and submersion animations (2-second emerge, 1.5-second submerge)
7. [ ] Six jellies total, one for each harp string

## Tasks / Subtasks

- [ ] Create JellyCreature class with Three.js mesh setup (AC: #1)
  - [ ] SphereGeometry with high segment count (64×64) for smooth displacement
  - [ ] EnhancedJellyMaterial with custom vertex/fragment shaders
  - [ ] Position at water surface level (y=0) when submerged
  - [ ] Associated note property (0-5 for C-A)
  - [ ] State tracking: submerged, emerging, teaching, submerging
- [ ] Create enhanced jelly shader with organic effects (AC: #1, #2, #3)
  - [ ] **Vertex shader**: Sine-wave displacement for breathing/pulsing effect
    - `pulseSpeed = 0.5 + (noteIndex * 0.1)` for unique rates per note
    - `displacement = sin(position.y * frequency + uTime * pulseSpeed) * amplitude`
  - [ ] **Fragment shader**: Fresnel-based bioluminescence
    - Rim lighting using view direction and normal
    - Emissive color based on teaching state (teaching = brighter)
    - Internal glow using depth-based alpha falloff
  - [ ] Uniforms: uTime, uTeachingState (0-1), uNoteIndex (0-5), uBaseColor, uGlowColor
- [ ] Implement jump-out animation system (AC: #4)
  - [ ] Parabolic arc trajectory from water to target string
  - [ ] Start: (x, 0, z) water surface → End: string position at y=1.5
  - [ ] Height offset: 0.5 units above water for arc peak
  - [ ] Duration: 1.5 seconds total emerge
  - [ ] Easing: easeOutQuad for ascent, easeInQuad for descent
  - [ ] Rotation: slight wobble during flight for organic feel
- [ ] Create teaching state behaviors (AC: #3, #5)
  - [ ] When teaching: intensify emissive to 1.0
  - [ ] Pulse rate increases by 50% during teaching
  - [ ] Create ripple effect on water surface at target string position
  - [ ] Teaching lasts 2 seconds per note demonstration
  - [ ] Scale up/down slightly (0.9 to 1.1) to draw attention
- [ ] Implement emergence and submersion animations (AC: #6)
  - [ ] **Emergence sequence:**
    - Submerged → Jump arc → Land on string → Teaching phase begins
    - Water splash particle effect on exit (simple VFX)
    - Drip particles during jump
  - [ ] **Submersion sequence:**
    - Fade emissive over 0.5 seconds
    - Shrink scale to 0.3 over 1 second
    - Move to water surface (y=0)
    - Fade alpha to 0 and disable rendering
  - [ ] Smooth easing functions: smoothstep for scale, lerp for position
- [ ] Create JellyManager to coordinate all six jellies (AC: #7)
  - [ ] Array of six JellyCreature instances
  - [ ] Note-to-jelly mapping (C=0, D=1, E=2, F=3, G=4, A=5)
  - [ ] demonstrateNote(noteIndex) method to trigger emergence
  - [ ] submergeJelly(noteIndex) method after teaching complete
  - [ ] Only one jelly teaching at a time
  - [ ] Queue system if multiple demonstrations requested
- [ ] Connect to harp string visualization (AC: #5)
  - [ ] Access HarpString mesh positions from scene
  - [ ] Create water ripple at string base when jelly demonstrates
  - [ ] Ripple: expanding ring shader effect on water surface
  - [ ] Sync ripple timing with jelly teaching duration
- [ ] Performance testing
  - [ ] Verify 60 FPS with all six jellies active
  - [ ] Test shader compilation time
  - [ ] Validate memory usage (dispose geometries/materials when done)

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Shader Language:** GLSL ES 3.0 (WebGPU) with fallback to GLSL ES 1.0 (WebGL2)
**Scene Format:** GLB with Gaussian Splatting via Shadow Engine

**File Organization:**
```
vimana/
├── src/
│   ├── shaders/
│   │   ├── jelly-vertex.glsl
│   │   └── jelly-fragment.glsl
│   ├── entities/
│   │   ├── JellyCreature.ts
│   │   └── JellyManager.ts
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Technical Requirements

**Jelly Creature Specifications:**

| Property | Value | Notes |
|----------|-------|-------|
| Base Geometry | Sphere (radius 0.3) | High segment count for smooth displacement |
| Position (submerged) | y = -0.2 | Below water surface |
| Position (teaching) | Above string (y = 1.5) | Hovering near associated harp string |
| Scale (idle) | 1.0 | Normal size |
| Scale (teaching) | 0.9 ↔ 1.1 | Pulsing scale for emphasis |
| Pulse Rate | 0.5 + (note × 0.1) Hz | Unique per note |

**Color Scheme:**
- Base color: `#88ccff` (soft cyan-blue)
- Teaching glow: `#00ffff` (bright cyan) to `#ff88ff` (magenta)
- Emissive intensity: 0.3 (idle) → 1.0 (teaching)
- Fresnel power: 2.0 for rim lighting

**Animation Timings:**
```
Emergence:        0.0s → 1.5s (jump arc to string)
Teaching:         1.5s → 3.5s (2 second demonstration)
Submerge start:   3.5s → 4.0s (fade and shrink)
Submerge end:     4.0s → 5.0s (return to water)
Total cycle:      ~5 seconds per note
```

**Jump Arc Formula:**
```javascript
// Parabolic arc from start to end with height offset
const t = progress; // 0 to 1
const arcHeight = 0.5;
const start = { x: startX, y: 0, z: startZ };
const end = { x: stringX, y: 1.5, z: stringZ };

const x = lerp(start.x, end.x, t);
const y = arcHeight * 4 * (t - t * t); // Parabola peaking at t=0.5
const z = lerp(start.z, end.z, t);
```

### Shader Implementation Details

**Jelly Vertex Shader:**
```glsl
uniform float uTime;
uniform float uPulseSpeed;
uniform float uPulseAmplitude;

varying vec3 vNormal;
varying vec3 vPosition;
varying float vDisplacement;

void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;

    // Breathing/pulsing displacement
    float pulse = sin(position.y * 3.0 + uTime * uPulseSpeed);
    vDisplacement = pulse * uPulseAmplitude;

    vec3 newPosition = position + normal * vDisplacement;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
}
```

**Jelly Fragment Shader:**
```glsl
uniform float uTeachingState; // 0 = idle, 1 = teaching
uniform vec3 uBaseColor;
uniform vec3 uGlowColor;

varying vec3 vNormal;
varying vec3 vPosition;
varying float vDisplacement;

void main() {
    // Fresnel rim lighting
    vec3 viewDir = normalize(cameraPosition - vPosition);
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0);

    // Internal glow from displacement
    float internalGlow = smoothstep(-0.1, 0.1, vDisplacement);

    // Mix base color with glow based on teaching state
    float glowIntensity = 0.3 + (uTeachingState * 0.7);
    vec3 color = mix(uBaseColor, uGlowColor, fresnel * glowIntensity);
    color += uGlowColor * internalGlow * glowIntensity;

    // Alpha for transparency
    float alpha = 0.6 + fresnel * 0.4;

    gl_FragColor = vec4(color, alpha);
}
```

### Memory Management Requirements

All JellyCreature instances MUST implement destroy() methods:

```javascript
class JellyCreature {
    destroy() {
        if (this.mesh) {
            this.scene.remove(this.mesh);
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
            this.mesh = null;
        }
        if (this.rippleEffect) {
            this.rippleEffect.destroy();
        }
    }
}
```

### Dependencies

**Previous Story:** 1.1 Visual Foundation (water surface must exist)

**Next Story:** 1.3 Gentle Feedback (jellies provide visual feedback for wrong notes)

**External Dependencies:**
- Three.js core: scene graph, materials, geometries
- Water shader from Story 1.1 (for ripple effects)
- Harp string positions (from GLB scene)

### Integration Points

**Harp Room Scene:**
```javascript
class HarpRoom {
    constructor() {
        this.jellyManager = new JellyManager(this.scene, this.waterMaterial);
    }

    demonstrateNote(noteIndex) {
        // Trigger jelly to emerge and teach
        this.jellyManager.demonstrateNote(noteIndex);
    }
}
```

### References

- [Source: music-room-proto-epic.md#Story 1.2]
- [Source: gdd.md#Archive of Voices chamber description]
- [Source: narrative-design.md#Musical teaching concept]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/shaders/jelly-vertex.glsl` (create)
- `src/shaders/jelly-fragment.glsl` (create)
- `src/entities/JellyCreature.ts` (create)
- `src/entities/JellyManager.ts` (create)
