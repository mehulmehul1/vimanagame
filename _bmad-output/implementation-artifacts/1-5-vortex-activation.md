# Story 1.5: Vortex Activation

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player completing the duet with the Vimana**,
I want **the vortex to intensify and the platform to carry me toward it**,
so that **I feel the ship is responding to our harmony and drawing me into the next chapter**.

## Acceptance Criteria

1. [ ] Vortex activation follows duet progress (0-1) from PatientJellyManager
2. [ ] Vortex emissive intensity scales from 0.0 → 3.0 as progress increases
3. [ ] Water harmonicResonance uniform updates with duet progress
4. [ ] Particle spin speed increases with activation (1x → 4x base speed)
5. [ ] Core white light intensifies at full activation (vortex center glows)
6. [ ] Platform detaches from floor and rides to vortex on completion
7. [ ] 5-second platform ride animation with smooth easing
8. [ ] Transition to shell spawn on arrival (Story 1.6)

## Tasks / Subtasks

- [ ] Create VortexActivationController class (AC: #1, #2, #3, #4, #5)
  - [ ] References to VortexMaterial and VortexParticles from Story 1.1
  - [ ] Reference to WaterMaterial for harmonicResonance updates
  - [ ] setActivation(progress: number) method (0-1 input)
  - [ ] Smooth interpolation over time (no abrupt changes)
  - [ ] Update loop for continuous animation
  - [ ] Event dispatch on full activation (platform ride trigger)
- [ ] Implement vortex material activation scaling (AC: #2)
  - [ ] VortexMaterial emissive intensity: 0 → 3.0
  - [ ] Formula: `emissiveIntensity = baseIntensity + (progress * 3.0)`
  - [ ] Base intensity: 0.2 (always slightly visible)
  - [ ] Color shift: Deep purple (#4400aa) → Bright cyan (#00ffff) → White
  - [ ] Uniform: uVortexActivation updated each frame
  - [ ] Swirl noise intensity increases with activation
- [ ] Update water surface harmonic resonance (AC: #3)
  - [ ] WaterMaterial uniform: uHarmonicResonance = progress
  - [ ] Bioluminescent color intensifies with progress
  - [ ] Ripple amplitude increases: base 0.1 → 0.5 at full activation
  - [ ] Water becomes more "alive" as duet progresses
  - [ ] At full activation: water glows with cyan-white light
- [ ] Implement particle system activation (AC: #4)
  - [ ] VortexParticles spin speed multiplier
  - [ ] Formula: `spinSpeed = baseSpeed * (1 + progress * 3)`
  - [ ] Base speed: 1.0, max speed: 4.0
  - [ ] Particle count increases with LOD:
    - Progress 0.0-0.3: 50% of particles active
    - Progress 0.3-0.6: 75% of particles active
    - Progress 0.6-1.0: 100% of particles active
  - [ ] Particle color shifts from purple → cyan → white
  - [ ] Add vertical motion to particles at high activation (rise toward center)
- [ ] Create core white light effect (AC: #5)
  - [ ] Point light at vortex center (0, 0.5, 2)
  - [ ] Intensity: 0 → 10.0 as progress → 1.0
  - [ ] Color: Pure white (#ffffff)
  - [ ] Range: 5 units (affects surrounding area)
  - [ ] Cast shadows: false (light is ethereal)
  - [ ] Add subtle bloom effect if post-processing available
- [ ] Implement platform detachment animation (AC: #6)
  - [ ] Find player platform mesh by name ("PlayerPlatform" or similar)
  - [ ] On full activation: trigger platform unlock
  - [ ] Visual: Platform shifts color (gray → warm amber)
  - [ ] Platform sinks slightly (0.1 units) then rises
  - [ ] Sound: Mechanical "unlock" tone (low to high frequency sweep)
  - [ ] Duration: 1 second for detachment phase
- [ ] Create platform ride animation (AC: #6, #7)
  - [ ] Target position: In front of vortex (0, 0.5, 1.0)
  - [ ] Start position: Original platform location
  - [ ] Duration: 5 seconds total
  - [ ] Easing: easeInOutCubic for smooth motion
  - [ ] Path: Slight arc (rise 0.3 units at midpoint)
  - [ ] Camera follows platform smoothly
  - [ ] Player remains centered on platform (no relative motion)
  - [ ] On arrival: trigger shell spawn event
- [ ] Connect to duet progress (AC: #1)
  - [ ] Subscribe to PatientJellyManager progress events
  - [ ] Update activation each frame with lerped value
  - [ ] Smooth factor: 0.1 (10% per frame toward target)
  - [ ] At progress 1.0: trigger platform ride sequence
  - [ ] Dispatch 'vortex-complete' event for Story 1.6
- [ ] Create VortexLightingManager (AC: #5)
  - [ ] Manage dynamic lights during activation
  - [ ] Core light (point) at vortex center
  - [ ] Ambient light intensification across scene
  - [ ] Bioluminescence boost on nearby surfaces
  - [ ] Cleanup method to remove lights on scene transition
- [ ] Performance and polish
  - [ ] Ensure 60 FPS during full activation
  - [ ] Test light shadow performance
  - [ ] Verify particle count LOD works correctly
  - [ ] Confirm smooth color transitions

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Scene Format:** GLB with Gaussian Splatting via Shadow Engine

**File Organization:**
```
vimana/
├── src/
│   ├── entities/
│   │   ├── VortexActivationController.ts
│   │   ├── VortexLightingManager.ts
│   │   ├── PlatformRideAnimator.ts
│   │   ├── VortexMaterial.ts (from Story 1.1)
│   │   ├── VortexParticles.ts (from Story 1.1)
│   │   └── WaterMaterial.ts (from Story 1.1)
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Activation Curve Specifications

**Progress → Visual Mapping:**

| Progress | Emissive | Spin Speed | Particle Count | Core Light |
|----------|----------|------------|----------------|------------|
| 0.0 | 0.2 | 1.0x | 50% | 0 (off) |
| 0.25 | 0.95 | 1.75x | 50% | 2.5 |
| 0.5 | 1.7 | 2.5x | 75% | 5.0 |
| 0.75 | 2.45 | 3.25x | 100% | 7.5 |
| 1.0 | 3.0 | 4.0x | 100% | 10.0 |

**Color Transition:**
```
0.0 - 0.33: Deep Purple (#4400aa) → Purple (#8800ff)
0.33 - 0.66: Purple (#8800ff) → Cyan (#00ffff)
0.66 - 1.0:  Cyan (#00ffff) → White (#ffffff)
```

### VortexActivationController Implementation

**Main Controller:**
```javascript
class VortexActivationController {
    private activation: number = 0; // Current lerped activation
    private targetActivation: number = 0; // Target from duet progress
    private vortexMaterial: VortexMaterial;
    private vortexParticles: VortexParticles;
    private waterMaterial: WaterMaterial;
    private coreLight: PointLight;
    private platform: PlatformRideAnimator;

    constructor(vortexMaterial, vortexParticles, waterMaterial, platform) {
        this.vortexMaterial = vortexMaterial;
        this.vortexParticles = vortexParticles;
        this.waterMaterial = waterMaterial;
        this.platform = platform;

        // Create core light
        this.coreLight = new PointLight(0xffffff, 0, 5);
        this.coreLight.position.set(0, 0.5, 2);
        this.scene.add(this.coreLight);
    }

    setActivation(progress: number) {
        this.targetActivation = Math.max(0, Math.min(1, progress));
    }

    update(deltaTime: number) {
        // Smooth lerping toward target
        const smoothing = 0.1;
        this.activation += (this.targetActivation - this.activation) * smoothing;

        // Update all systems
        this.updateVortexMaterial();
        this.updateWaterSurface();
        this.updateParticles();
        this.updateCoreLight();

        // Check for full activation
        if (this.activation >= 0.99 && !this.platform.rideStarted) {
            this.startPlatformRide();
        }
    }

    private updateVortexMaterial() {
        // Emissive intensity: 0.2 → 3.0
        const emissive = 0.2 + (this.activation * 2.8);
        this.vortexMaterial.uniforms.uVortexActivation.value = this.activation;
        this.vortexMaterial.emissiveIntensity = emissive;

        // Color transition
        const color = this.getActivationColor(this.activation);
        this.vortexMaterial.emissive.setRGB(color.r, color.g, color.b);
    }

    private updateWaterSurface() {
        // Harmonic resonance uniform
        this.waterMaterial.uniforms.uHarmonicResonance.value = this.activation;

        // Bioluminescent boost
        const bioBoost = 1.0 + (this.activation * 0.5);
        this.waterMaterial.uniforms.uBioluminescentIntensity.value = bioBoost;
    }

    private updateParticles() {
        // Spin speed multiplier
        const spinMultiplier = 1 + (this.activation * 3);
        this.vortexParticles.setSpinSpeed(spinMultiplier);

        // Particle count LOD
        const particleRatio = this.activation < 0.3 ? 0.5 :
                            this.activation < 0.6 ? 0.75 : 1.0;
        this.vortexParticles.setParticleRatio(particleRatio);
    }

    private updateCoreLight() {
        this.coreLight.intensity = this.activation * 10;
    }

    private getActivationColor(t: number) {
        // Purple → Cyan → White
        if (t < 0.33) {
            const localT = t / 0.33;
            return {
                r: this.lerp(0x44/255, 0x88/255, localT),
                g: this.lerp(0x00/255, 0x00/255, localT),
                b: this.lerp(0xaa/255, 0xff/255, localT)
            };
        } else if (t < 0.66) {
            const localT = (t - 0.33) / 0.33;
            return {
                r: this.lerp(0x88/255, 0x00/255, localT),
                g: this.lerp(0x00/255, 0xff/255, localT),
                b: this.lerp(0xff/255, 0xff/255, localT)
            };
        } else {
            const localT = (t - 0.66) / 0.34;
            return {
                r: this.lerp(0x00/255, 1.0, localT),
                g: this.lerp(0xff/255, 1.0, localT),
                b: this.lerp(0xff/255, 1.0, localT)
            };
        }
    }

    private lerp(a: number, b: number, t: number): number {
        return a + (b - a) * t;
    }

    private startPlatformRide() {
        this.platform.startRide();
        // Dispatch event for Story 1.6
        this.dispatchEvent('vortex-complete');
    }

    destroy() {
        this.scene.remove(this.coreLight);
        // Other cleanup...
    }
}
```

### Platform Ride Animation

**PlatformRideAnimator:**
```javascript
class PlatformRideAnimator {
    private platform: Mesh;
    private startPosition: Vector3;
    private endPosition: Vector3 = new Vector3(0, 0.5, 1.0);
    private rideProgress: number = 0;
    private rideStarted: boolean = false;
    private rideComplete: boolean = false;
    private rideDuration: number = 5.0;

    constructor(platform: Mesh) {
        this.platform = platform;
        this.startPosition = platform.position.clone();
    }

    startRide() {
        this.rideStarted = true;
    }

    update(deltaTime: number) {
        if (!this.rideStarted || this.rideComplete) return;

        this.rideProgress += deltaTime / this.rideDuration;

        if (this.rideProgress >= 1) {
            this.rideProgress = 1;
            this.rideComplete = true;
            this.onRideComplete();
        }

        // easeInOutCubic
        const t = this.rideProgress;
        const eased = t < 0.5
            ? 4 * t * t * t
            : 1 - Math.pow(-2 * t + 2, 3) / 2;

        // Arc path (rise at midpoint)
        const arcHeight = 0.3 * Math.sin(eased * Math.PI);

        this.platform.position.lerpVectors(
            this.startPosition,
            this.endPosition,
            eased
        );
        this.platform.position.y += arcHeight;
    }

    private onRideComplete() {
        // Trigger shell spawn
        this.dispatchEvent('platform-arrived');
    }
}
```

### Material Uniform Updates

**Vortex Material Shader Integration:**
```glsl
// In vortex fragment shader
uniform float uVortexActivation; // 0-1 from VortexActivationController

void main() {
    // Base emissive
    vec3 emissive = uEmissiveColor;

    // Activation-based intensity
    float intensity = 0.2 + (uVortexActivation * 2.8);
    emissive *= intensity;

    // Color shift with activation
    vec3 purple = vec3(0.27, 0.0, 0.67);
    vec3 cyan = vec3(0.0, 1.0, 1.0);
    vec3 white = vec3(1.0);

    vec3 activeColor;
    if (uVortexActivation < 0.33) {
        activeColor = mix(purple, cyan, uVortexActivation / 0.33);
    } else {
        activeColor = mix(cyan, white, (uVortexActivation - 0.33) / 0.67);
    }

    emissive = mix(emissive, activeColor, uVortexActivation);

    gl_FragColor = vec4(emissive, 1.0);
}
```

### Event Integration

**HarpRoom Integration:**
```javascript
class HarpRoom {
    private vortexController: VortexActivationController;

    constructor() {
        // Initialize systems
        this.vortexController = new VortexActivationController(...);

        // Connect to duet progress
        this.duetManager.on('progress', (progress) => {
            this.vortexController.setActivation(progress);
        });

        // Listen for completion
        this.vortexController.addEventListener('vortex-complete', () => {
            this.onVortexComplete();
        });
    }

    private onVortexComplete() {
        // Transition to shell spawn
        // Story 1.6 will handle this
    }
}
```

### Memory Management

```javascript
class VortexActivationController {
    destroy() {
        if (this.coreLight) {
            this.scene.remove(this.coreLight);
            this.coreLight.dispose();
        }
        this.vortexMaterial = null;
        this.vortexParticles = null;
        this.waterMaterial = null;
    }
}
```

### Dependencies

**Previous Story:** 1.4 Duet Mechanics (provides progress input)

**Next Story:** 1.6 Shell Collection (triggers on platform arrival)

**External Dependencies:**
- VortexMaterial (Story 1.1)
- VortexParticles (Story 1.1)
- WaterMaterial (Story 1.1)
- PatientJellyManager (Story 1.4)
- Three.js lights and animation

### Platform Ride Timing

**Ride Sequence:**
```
0.0s - 1.0s: Platform detachment (color shift, slight sink)
1.0s - 6.0s: Platform ride to vortex (5 seconds)
6.0s: Arrival at vortex, shell spawn triggered
```

**Camera Behavior:**
- Camera follows platform smoothly
- No jitter or sudden movements
- Player remains centered (platform moves, camera tracks)

### References

- [Source: music-room-proto-epic.md#Story 1.5]
- [Source: gdd.md#Ascension sequence]
- [Source: narrative-design.md#Vortex activation]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/entities/VortexActivationController.ts` (create)
- `src/entities/VortexLightingManager.ts` (create)
- `src/entities/PlatformRideAnimator.ts` (create)
