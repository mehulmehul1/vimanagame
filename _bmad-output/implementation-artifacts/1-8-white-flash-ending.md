# Story 1.8: White Flash Ending

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player completing the Archive of Voices chamber**,
I want **the vortex to engulf me in a transcendent white light**,
so that **I feel the experience has reached its climax and am ready to transition to the next chamber**.

## Acceptance Criteria

1. [ ] WhiteFlashEnding class with full-screen quad shader
2. [ ] 8-second total duration for the ending sequence
3. [ ] Phase 1 (0-60%): Multi-layered spiral intensifies from center
4. [ ] Phase 2 (40-100%): Fade to pure white
5. [ ] Color transition: cyan → purple → white
6. [ ] Vignette effect for focus at screen edges
7. [ ] Smooth alpha fade at end for scene transition
8. [ ] Scene completion callback for next scene trigger

## Tasks / Subtasks

- [ ] Create WhiteFlashEnding class structure (AC: #1)
  - [ ] Full-screen quad geometry (PlaneBufferGeometry, 2×2 units)
  - [ ] ShaderMaterial with custom vertex/fragment shaders
  - [ ] Parent to camera for proper positioning (always in view)
  - [ ] Render order: last (on top of everything)
  - [ ] Depth test: false (always visible)
  - [ ] Depth write: false (doesn't affect depth buffer)
  - [ ] Transparent blending for smooth fade
- [ ] Implement spiral SDF in fragment shader (AC: #3)
  - [ ] Multi-layered logarithmic spiral pattern
  - [ ] **Spiral formula:** `angle = atan(coord.y, coord.x) + log(length) * turns`
  - [ ] 3 layers with different rotation speeds and scales
  - [ ] Layer 1: Slow, large (base spiral)
  - [ ] Layer 2: Medium, medium (counter-rotation)
  - [ ] Layer 3: Fast, small (detail spiral)
  - [ ] Combine layers using additive blending
  - [ ] Spiral arms: 6-8 for visual complexity
- [ ] Create color transition system (AC: #5)
  - [ ] Phase-based color interpolation
  - [ ] **Phase 1 (0-40%):** Cyan (#00ffff) base
  - [ ] **Phase 2 (40-70%):** Purple (#8800ff) dominates
  - [ ] **Phase 3 (70-100%):** White (#ffffff) final
  - [ ] Smooth mixing using smoothstep between phases
  - [ ] Boost brightness in final phase for "blinding" effect
- [ ] Implement fade-to-white (AC: #2, #4)
  - [ ] Duration: 8 seconds total
  - [ ] Progress uniform (0-1) driven by time
  - [ ] **White intensity calculation:**
    - 0-0.6: 0 (spirals visible)
    - 0.6-0.8: Ramp to 0.5 (spirals + white wash)
    - 0.8-1.0: Ramp to 1.0 (full white)
  - [ ] Add white overlay color: `vec3(intensity)`
  - [ ] Use screen-space UVs for uniform coverage
- [ ] Add vignette effect (AC: #6)
  - [ ] Radial gradient from center to edges
  - [ ] **Vignette formula:** `vignette = 1.0 - length(uv - 0.5) * 2.0`
  - [ ] Apply to final color: `color *= vignette`
  - [ ] Soft edges using smoothstep
  - [ ] Strength increases with progress (0.3 → 0.8)
  - [ ] Creates focus on center spiral
- [ ] Implement alpha fade for transition (AC: #7)
  - [ ] Final fade starts at 95% progress
  - [ **Fade curve:** `alpha = smoothstep(0.95, 1.0, progress)`
  - [ ] Enables smooth scene transition
  - [ ] Callback triggered when fade completes
  - [ ] Optional: hold white for 0.5s before fade
- [ ] Create animation controller (AC: #2, #8)
  - [ ] start() method begins sequence
  - [ ] Update loop runs each frame during sequence
  - [ ] Track elapsed time since start
  - [ ] Calculate progress: `elapsed / duration`
  - [ ] Update shader uniforms each frame
  - [ ] On complete: dispatch callback, hide quad
- [ ] Add audio integration (optional but recommended)
  - [ ] Ascension tone: Rising sine wave sweep
  - [ ] Start: C5 (523 Hz) → End: C6 (1046 Hz) octave up
  - [ ] Duration: 6 seconds (matches visual)
  - [ ] Gain envelope: Fade in, sustain, fade out
  - [ ] Use Web Audio API for synthesis
- [ ] Implement scene completion callback (AC: #8)
  - [ ] onComplete callback function parameter
  - [ ] Triggered at 100% progress
  - [ ] Pass chamber ID for next scene logic
  - [ ] Cleanup resources before callback
  - [ ] Error handling for missing callback
- [ ] Create WhiteFlashManager for coordination
  - [ ] Singleton pattern for global access
  - [ ] triggerEnding(chamberId, callback) method
  - [ ] Ensure only one ending active at a time
  - [ ] Handle premature interruption
  - [ ] Debug mode for faster testing (1-second duration)
- [ ] Performance and polish
  - [ ] Test 60 FPS during full-screen effect
  - [ ] Verify mobile shader compatibility
  - [ ] Ensure smooth color transitions (no banding)
  - [ ] Test at different screen resolutions

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Shader Language:** GLSL ES 3.0 (WebGPU) with fallback to GLSL ES 1.0 (WebGL2)

**File Organization:**
```
vimana/
├── src/
│   ├── shaders/
│   │   ├── white-flash-vertex.glsl
│   │   └── white-flash-fragment.glsl
│   ├── entities/
│   │   ├── WhiteFlashEnding.ts
│   │   └── WhiteFlashManager.ts
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Technical Specifications

**Timing Breakdown:**

| Progress | Time | Phase | Visual State |
|----------|------|-------|--------------|
| 0.0 - 0.3 | 0-2.4s | Spiral emerge | Cyan spirals from center |
| 0.3 - 0.6 | 2.4-4.8s | Spiral intensify | Purple spirals, faster |
| 0.6 - 0.8 | 4.8-6.4s | White wash | Spirals fade, white enters |
| 0.8 - 0.95 | 6.4-7.6s | Full white | Pure white screen |
| 0.95 - 1.0 | 7.6-8.0s | Fade out | Alpha fade for transition |

**Color Stops:**
```
0.0: #00ffff (Cyan)
0.3: #4400ff (Blue-purple)
0.5: #8800ff (Purple)
0.7: #cc88ff (Light purple)
0.85: #ffffff (White, 50%)
1.0: #ffffff (Pure white)
```

### Shader Implementation

**White Flash Vertex Shader:**
```glsl
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
```

**White Flash Fragment Shader:**
```glsl
uniform float uTime;         // Elapsed time since start
uniform float uProgress;     // 0-1 progress through sequence
uniform vec3 uColor1;        // Start color (cyan)
uniform vec3 uColor2;        // Mid color (purple)
uniform vec3 uColor3;        // End color (white)

varying vec2 vUv;

// Logarithmic spiral SDF
float spiralSDF(vec2 uv, float turns, float scale) {
    vec2 centered = uv - 0.5;
    float angle = atan(centered.y, centered.x);
    float radius = length(centered) * 2.0;

    float spiral = sin(angle * turns + log(radius + 0.1) * scale);
    return spiral;
}

void main() {
    vec2 uv = vUv;

    // Multi-layered spiral
    float spiral1 = spiralSDF(uv, 6.0, 3.0 + uProgress * 2.0);
    float spiral2 = spiralSDF(uv, 8.0, -4.0 - uProgress * 3.0);
    float spiral3 = spiralSDF(uv, 10.0, 5.0 + uProgress * 4.0);

    // Combine layers
    float combined = (spiral1 + spiral2 * 0.5 + spiral3 * 0.25) / 1.75;

    // Normalize to 0-1
    float spiralValue = combined * 0.5 + 0.5;

    // Radial distance from center
    float dist = length(uv - 0.5) * 2.0;

    // Spiral intensity (brighter at center)
    float intensity = (1.0 - dist) * spiralValue;
    intensity = smoothstep(0.0, 1.0, intensity);
    intensity *= (1.0 - uProgress * 0.5); // Fade as white enters

    // Color transition
    vec3 color;
    if (uProgress < 0.5) {
        // Cyan to purple
        float t = uProgress * 2.0;
        color = mix(uColor1, uColor2, smoothstep(0.0, 1.0, t));
    } else {
        // Purple to white
        float t = (uProgress - 0.5) * 2.0;
        color = mix(uColor2, uColor3, smoothstep(0.0, 1.0, t));
    }

    // Add spiral color
    color += vec3(intensity);

    // White wash (0.6 - 1.0)
    float whiteWash = smoothstep(0.6, 1.0, uProgress);
    color = mix(color, vec3(1.0), whiteWash * whiteWash);

    // Vignette
    float vignette = 1.0 - dist * 0.5 * (0.3 + uProgress * 0.5);
    color *= vignette;

    // Alpha fade at end
    float alpha = 1.0;
    if (uProgress > 0.95) {
        alpha = 1.0 - smoothstep(0.95, 1.0, uProgress);
    }

    gl_FragColor = vec4(color, alpha);
}
```

### WhiteFlashEnding Class

**Main Implementation:**
```typescript
class WhiteFlashEnding extends THREE.Mesh {
    private material: THREE.ShaderMaterial;
    private duration: number = 8.0;
    private elapsedTime: number = 0;
    private isActive: boolean = false;
    private onCompleteCallback?: () => void;
    private uniforms: {
        uTime: { value: number };
        uProgress: { value: number };
        uColor1: { value: THREE.Color };
        uColor2: { value: THREE.Color };
        uColor3: { value: THREE.Color };
    };

    constructor(private camera: THREE.Camera) {
        // Full-screen quad
        const geometry = new THREE.PlaneGeometry(2, 2);

        // Shader material
        this.uniforms = {
            uTime: { value: 0 },
            uProgress: { value: 0 },
            uColor1: { value: new THREE.Color(0x00ffff) }, // Cyan
            uColor2: { value: new THREE.Color(0x8800ff) }, // Purple
            uColor3: { value: new THREE.Color(0xffffff) }  // White
        };

        this.material = new THREE.ShaderMaterial({
            vertexShader: whiteFlashVertexShader,
            fragmentShader: whiteFlashFragmentShader,
            uniforms: this.uniforms,
            transparent: true,
            depthTest: false,
            depthWrite: false
        });

        super(geometry, this.material);

        // Parent to camera
        camera.add(this);
        this.position.set(0, 0, -0.5); // Just in front of camera

        // Initially invisible
        this.visible = false;
    }

    start(onComplete?: () => void): void {
        this.isActive = true;
        this.elapsedTime = 0;
        this.onCompleteCallback = onComplete;
        this.visible = true;

        // Reset uniforms
        this.uniforms.uTime.value = 0;
        this.uniforms.uProgress.value = 0;
    }

    update(deltaTime: number): void {
        if (!this.isActive) return;

        this.elapsedTime += deltaTime;
        const progress = Math.min(this.elapsedTime / this.duration, 1.0);

        // Update uniforms
        this.uniforms.uTime.value = this.elapsedTime;
        this.uniforms.uProgress.value = progress;

        // Check completion
        if (progress >= 1.0) {
            this.complete();
        }
    }

    private complete(): void {
        this.isActive = false;
        this.visible = false;

        if (this.onCompleteCallback) {
            this.onCompleteCallback();
        }
    }

    destroy(): void {
        this.camera.remove(this);
        this.geometry.dispose();
        this.material.dispose();
    }
}
```

### WhiteFlashManager

**Coordination Layer:**
```typescript
class WhiteFlashManager {
    private static instance: WhiteFlashManager;
    private currentEnding: WhiteFlashEnding | null = null;
    private camera: THREE.Camera;

    private constructor(camera: THREE.Camera) {
        this.camera = camera;
    }

    static getInstance(camera?: THREE.Camera): WhiteFlashManager {
        if (!WhiteFlashManager.instance) {
            if (!camera) {
                throw new Error('Camera required for first initialization');
            }
            WhiteFlashManager.instance = new WhiteFlashManager(camera);
        }
        return WhiteFlashManager.instance;
    }

    triggerEnding(chamberId: string, onComplete?: () => void): void {
        // Don't interrupt if already running
        if (this.currentEnding) {
            console.warn('White flash ending already in progress');
            return;
        }

        // Create new ending
        this.currentEnding = new WhiteFlashEnding(this.camera);

        // Start sequence
        this.currentEnding.start(() => {
            // Cleanup and callback
            this.currentEnding?.destroy();
            this.currentEnding = null;

            if (onComplete) {
                onComplete();
            }

            // Dispatch event for other systems
            window.dispatchEvent(new CustomEvent('white-flash-complete', {
                detail: { chamber: chamberId }
            }));
        });
    }

    update(deltaTime: number): void {
        if (this.currentEnding) {
            this.currentEnding.update(deltaTime);
        }
    }

    // Debug mode for faster testing
    triggerDebugEnding(): void {
        if (this.currentEnding) {
            (this.currentEnding as any).duration = 1.0; // 1-second debug
        }
    }
}
```

### Audio Integration

**Optional Ascension Tone:**
```typescript
class WhiteFlashAudio {
    private audioContext: AudioContext;
    private oscillator: OscillatorNode;
    private gainNode: GainNode;

    constructor() {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    playAscendingTone(duration: number = 6.0): void {
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        this.oscillator = this.audioContext.createOscillator();
        this.gainNode = this.audioContext.createGain();

        // Frequency sweep: C5 → C6 (523 Hz → 1046 Hz)
        const now = this.audioContext.currentTime;
        this.oscillator.frequency.setValueAtTime(523, now);
        this.oscillator.frequency.exponentialRampToValueAtTime(1046, now + duration);

        // Gain envelope
        this.gainNode.gain.setValueAtTime(0, now);
        this.gainNode.gain.linearRampToValueAtTime(0.15, now + 0.5);
        this.gainNode.gain.linearRampToValueAtTime(0.15, now + duration - 0.5);
        this.gainNode.gain.linearRampToValueAtTime(0, now + duration);

        this.oscillator.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);

        this.oscillator.start(now);
        this.oscillator.stop(now + duration);
    }
}
```

### Usage Example

**In HarpRoom Scene:**
```typescript
class HarpRoom {
    private whiteFlashManager: WhiteFlashManager;

    constructor() {
        this.whiteFlashManager = WhiteFlashManager.getInstance(this.camera);
    }

    onShellCollectionComplete() {
        // Trigger white flash ending
        this.whiteFlashManager.triggerEnding('archive-of-voices', () => {
            // Transition to next scene
            this.sceneManager.loadNextScene();
        });
    }

    update(deltaTime: number) {
        this.whiteFlashManager.update(deltaTime);
    }
}
```

### Memory Management

```typescript
class WhiteFlashEnding {
    destroy(): void {
        // Remove from camera
        this.camera.remove(this);

        // Dispose resources
        this.geometry.dispose();
        this.material.dispose();

        // Clear references
        this.onCompleteCallback = undefined;
    }
}
```

### Dependencies

**Previous Story:** 1.7 UI Overlay System (shell collection triggers ending)

**Next Story:** 1.9 Performance & Polish (optimize this effect)

**External Dependencies:**
- Three.js core: Camera, Mesh, ShaderMaterial
- GLSL shaders for full-screen effect
- Optional: Web Audio API for ascension tone

### Performance Notes

**Optimization Strategies:**
- Use simple SDF math (avoid complex raymarching)
- Single full-screen quad (minimal geometry)
- Pre-compute color values in vertex shader when possible
- Mobile: Reduce spiral layers from 3 to 2
- Test on lowest target device (30 FPS mobile)

### References

- [Source: music-room-proto-epic.md#Story 1.8]
- [Source: gdd.md#Ascension sequence]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/shaders/white-flash-vertex.glsl` (create)
- `src/shaders/white-flash-fragment.glsl` (create)
- `src/entities/WhiteFlashEnding.ts` (create)
- `src/entities/WhiteFlashManager.ts` (create)
- `src/audio/WhiteFlashAudio.ts` (create, optional)
