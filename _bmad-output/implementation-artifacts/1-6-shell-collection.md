# Story 1.6: Shell Collection System

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player completing the Archive of Voices duet**,
I want **a beautiful nautilus shell to materialize before me**,
so that **I feel rewarded for harmonizing with the ship and receive a tangible keepsake of our connection**.

## Acceptance Criteria

1. [ ] SDFShellMaterial with procedural nautilus spiral SDF shader
2. [ ] 3-second materialize animation with dissolve effect
3. [ ] Iridescence shifts with view angle (color: pearlescent white/gold/pink)
4. [ ] Raycast click detection for shell interaction
5. [ ] 1.5-second fly-to-UI animation with scale shrink
6. [ ] Shell spawns at (0, 1.0, 1.0) in front of player
7. [ ] Bobbing idle animation (gentle float up/down)
8. [ ] Global event dispatch on collection for UI integration

## Tasks / Subtasks

- [ ] Create SDFShellMaterial with procedural shell shader (AC: #1, #3)
  - [ ] SphereGeometry with 128 segments for smooth displacement
  - [ ] Custom vertex/fragment shaders with SDF nautilus spiral
  - [ ] Nautilus spiral formula: logarithmic spiral with growing radius
  - [ ] Iridescence using view-dependent fresnel
  - [ ] Color layers: white (#fffff0) → gold (#ffd700) → pink (#ffc0cb)
  - [ ] Emissive glow on appear, settles to subtle
  - [ ] Transparency with alpha blending for ethereal look
- [ ] Implement nautilus spiral SDF (AC: #1)
  - [ ] **Vertex shader**: Displace vertices using spiral formula
  - [ ] **Spiral formula:**
    ```glsl
    float angle = atan(position.z, position.x);
    float radius = length(position.xz);
    float spiral = sin(angle * 3.0 + log(radius) * 5.0);
    float displacement = spiral * 0.15 * radius;
    ```
  - [ ] Add detail noise for organic surface texture
  - [ ] Create chamber ridges using stepped sine wave
  - [ ] Hollow out center for nautilus opening
- [ ] Create materialize animation (AC: #2)
  - [ ] Start with scale 0 and alpha 0
  - [ ] 3-second duration with custom easing
  - [ ] **Phase 1 (0-1s):** Scale up with elastic bounce
  - [ ] **Phase 2 (1-2s):** Dissolve effect reveals shell texture
  - [ ] **Phase 3 (2-3s):** Glow intensifies then settles
  - [ ] Use simplex noise for dissolve pattern
  - [ ] Uniform: uMaterializeProgress (0-1)
- [ ] Implement iridescence effect (AC: #3)
  - [ ] View-dependent color shifting in fragment shader
  - [ ] Fresnel calculation: `pow(1.0 - dot(viewDir, normal), power)`
  - [ ] Color mixing based on fresnel value
  - [ ] **Low angle:** White/pearl
  - [ ] **Medium angle:** Gold shimmer
  - [ ] **High angle:** Pink/magenta highlights
  - [ ] Add subtle sparkle with random noise threshold
- [ ] Create ShellCollectible entity class (AC: #4, #6, #7)
  - [ ] Position: (0, 1.0, 1.0) world coordinates
  - [ ] Scale: 0.15 units (small but visible)
  - [ ] Bobbing animation: `y = baseY + sin(time * 2) * 0.05`
  - [ ] Slow rotation: 15 degrees per second around Y axis
  - [ ] Click detection via raycaster
  - [ ] State: MATERIALIZING, IDLE, COLLECTING, COLLECTED
  - [ ] On click: trigger fly-to-UI animation
- [ ] Implement raycast click detection (AC: #4)
  - [ ] Use Three.js Raycaster for mouse/touch input
  - [ ] Click radius: 2.0 units (generous for accessibility)
  - [ ] Only active when state is IDLE
  - [ ] Visual feedback on hover: slight scale increase (1.0 → 1.1)
  - [ ] Cursor change to pointer on hover
  - [ ] Support both mouse click and touch tap
- [ ] Create fly-to-UI animation (AC: #5)
  - [ ] Target: Top-left UI slot position (screen space)
  - [ ] Duration: 1.5 seconds
  - [ ] Path: Quadratic bezier curve (shell → control point → UI slot)
  - [ ] Control point: Above and to left of shell (for arc)
  - [ ] Scale shrink: 1.0 → 0.2 (matches UI icon size)
  - [ ] Rotation: Accelerate to 720 degrees during flight
  - [ ] Alpha fade: 1.0 → 0.5 during flight
  - [ ] On arrival: dispatch 'shell-collected' event
  - [ ] Remove shell mesh from scene
- [ ] Implement particle trail during collection (AC: #5)
  - [ ] Small particles spawn behind shell during flight
  - [ ] Color: Gold/pink matching iridescence
  - [ ] Particle lifetime: 0.5 seconds
  - [ ] Additive blending for glow
  - [ ] 20-30 particles total
- [ ] Create ShellManager for coordination (AC: #8)
  - [ ] Singleton pattern for global access
  - [ ] spawnShell(position) method
  - [ ] collectShell() method
  - [ ] Event dispatch: 'shell-collected' with chamber ID
  - [ ] Track collected shells per chamber (0-4)
  - [ ] Integrate with ShellUIOverlay from Story 1.7
  - [ ] Save state to localStorage for persistence
- [ ] Add audio feedback
  - [ ] Spawn sound: Gentle chime (E5 + G5 harmony)
  - [ ] Hover sound: Subtle sparkle (high sine wave burst)
  - [ ] Collect sound: Ascending arpeggio (C5 → E5 → G5 → C6)
  - [ ] Use Web Audio API oscillators (no external files)
  - [ ] Volume: -12 dB for subtlety
- [ ] Performance and polish
  - [ ] Optimize SDF shader for mobile
  - [ ] LOD system: simpler shell on low-end devices
  - [ ] Test 60 FPS during materialize animation
  - [ ] Verify click detection works on touch devices

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
│   │   ├── shell-vertex.glsl
│   │   └── shell-fragment.glsl
│   ├── entities/
│   │   ├── ShellCollectible.ts
│   │   └── ShellManager.ts
│   ├── audio/
│   │   └── ShellAudioFeedback.ts
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Technical Requirements

**Shell Specifications:**

| Property | Value | Notes |
|----------|-------|-------|
| Geometry | Sphere (128 segments) | High poly for smooth SDF displacement |
| Base Scale | 0.15 units | Small but clearly visible |
| Spawn Position | (0, 1.0, 1.0) | In front of player at vortex edge |
| Bob Amplitude | ±0.05 units | Gentle float |
| Bob Frequency | 2 Hz | Relaxed breathing |
| Rotation Speed | 15 deg/sec | Slow, majestic spin |
| Click Radius | 2.0 units | Generous for accessibility |

**Color Palette:**
```glsl
vec3 shellWhite = vec3(1.0, 1.0, 0.94);  // #fffff0
vec3 shellGold = vec3(1.0, 0.84, 0.0);   // #ffd700
vec3 shellPink = vec3(1.0, 0.75, 0.8);   // #ffc0cb
```

**Animation Timings:**
```
Materialize:  0.0s → 3.0s (appear + settle)
Idle:         3.0s → (wait for click)
Collect:      Click → 1.5s (fly to UI)
Total:        ~5 seconds from spawn to collection
```

### SDF Shell Shader Implementation

**Shell Vertex Shader:**
```glsl
uniform float uTime;
uniform float uMaterializeProgress;
uniform vec3 uSpawnPosition;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec2 vUv;

// Nautilus spiral SDF function
float nautilusSpiral(vec3 p) {
    float angle = atan(p.z, p.x);
    float radius = length(p.xz);

    // Logarithmic spiral formula
    float spiralGrowth = log(radius + 0.1) * 3.0;
    float spiralWave = sin(angle * 5.0 - spiralGrowth);
    float chamberRidge = smoothstep(0.3, 0.35, fract(spiralGrowth * 0.5));

    return spiralWave * 0.15 * radius * (1.0 + chamberRidge * 0.5);
}

void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    vUv = uv;

    // Apply nautilus spiral displacement
    vec3 displaced = position;
    float spiral = nautilusSpiral(displaced);
    displaced += normal * spiral;

    // Materialize animation (scale from 0)
    float scale = smoothstep(0.0, 0.3, uMaterializeProgress);
    if (uMaterializeProgress > 0.3) {
        scale = 1.0;
    }
    displaced *= scale;

    // Bobbing animation (only after materialize)
    if (uMaterializeProgress >= 1.0) {
        float bob = sin(uTime * 2.0) * 0.05;
        displaced.y += bob;
    }

    vec4 worldPos = modelMatrix * vec4(displaced, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
```

**Shell Fragment Shader:**
```glsl
uniform float uTime;
uniform float uMaterializeProgress;
uniform vec3 uCameraPosition;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vWorldPosition;
varying vec2 vUv;

// Simplex noise function (abbreviated)
float simplexNoise(vec3 p);

void main() {
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);

    // Fresnel for iridescence
    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);

    // Iridescent color mixing
    vec3 shellWhite = vec3(1.0, 1.0, 0.94);
    vec3 shellGold = vec3(1.0, 0.84, 0.0);
    vec3 shellPink = vec3(1.0, 0.75, 0.8);

    vec3 iridescentColor = mix(shellWhite, shellGold, fresnel * 0.6);
    iridescentColor = mix(iridescentColor, shellPink, fresnel * fresnel);

    // Add sparkle
    float sparkle = step(0.97, simplexNoise(vWorldPosition * 50.0 + uTime));
    iridescentColor += sparkle * 0.3;

    // Dissolve effect during materialize
    float dissolveNoise = simplexNoise(vWorldPosition * 3.0);
    float dissolve = smoothstep(
        uMaterializeProgress - 0.2,
        uMaterializeProgress + 0.1,
        dissolveNoise
    );
    float alpha = dissolve;

    // Emissive glow during materialize
    float glow = 0.0;
    if (uMaterializeProgress < 0.5) {
        glow = (0.5 - uMaterializeProgress) * 2.0;
    }
    vec3 emissive = iridescentColor * glow * 0.5;

    // Final color
    vec3 finalColor = iridescentColor + emissive;

    // Transparency
    alpha = mix(alpha, 1.0, fresnel * 0.3);

    gl_FragColor = vec4(finalColor, alpha);
}
```

### ShellCollectible Class

**Main Entity:**
```javascript
class ShellCollectible extends THREE.Mesh {
    private state: 'materializing' | 'idle' | 'collecting' | 'collected';
    private materializeProgress: number = 0;
    private idleTime: number = 0;
    private collectProgress: number = 0;
    private startPosition: Vector3;
    private uiTargetPosition: Vector3;

    constructor(private scene: Scene, private camera: Camera) {
        // High-poly sphere for smooth displacement
        const geometry = new THREE.SphereGeometry(1, 128, 128);

        // SDF shell material
        const material = new THREE.ShaderMaterial({
            vertexShader: shellVertexShader,
            fragmentShader: shellFragmentShader,
            uniforms: {
                uTime: { value: 0 },
                uMaterializeProgress: { value: 0 },
                uCameraPosition: { value: new THREE.Vector3() }
            },
            transparent: true,
            side: THREE.DoubleSide
        });

        super(geometry, material);

        this.position.set(0, 1.0, 1.0);
        this.scale.setScalar(0.15);
        this.state = 'materializing';
        this.startPosition = this.position.clone();

        scene.add(this);
    }

    update(deltaTime: number, time: number) {
        const material = this.material as THREE.ShaderMaterial;

        material.uniforms.uTime.value = time;
        material.uniforms.uCameraPosition.value.copy(this.camera.position);

        switch (this.state) {
            case 'materializing':
                this.updateMaterializing(deltaTime);
                break;
            case 'idle':
                this.updateIdle(deltaTime, time);
                break;
            case 'collecting':
                this.updateCollecting(deltaTime);
                break;
        }
    }

    private updateMaterializing(deltaTime: number) {
        this.materializeProgress += deltaTime / 3.0; // 3 seconds

        const material = this.material as THREE.ShaderMaterial;
        material.uniforms.uMaterializeProgress.value = this.materializeProgress;

        if (this.materializeProgress >= 1.0) {
            this.state = 'idle';
        }
    }

    private updateIdle(deltaTime: number, time: number) {
        this.idleTime += deltaTime;

        // Bobbing handled in shader via uTime
        // Slow rotation
        this.rotation.y += THREE.MathUtils.degToRad(15) * deltaTime;
    }

    private updateCollecting(deltaTime: number) {
        this.collectProgress += deltaTime / 1.5; // 1.5 seconds

        if (this.collectProgress >= 1.0) {
            this.state = 'collected';
            this.onCollectComplete();
            return;
        }

        // Quadratic bezier to UI position
        const t = this.collectProgress;
        const easeT = t * t * (3 - 2 * t); // smoothstep

        // Control point for arc
        const controlPoint = new THREE.Vector3(-0.5, 1.5, 0.5);

        // Bezier interpolation
        this.position.lerpVectors(
            this.startPosition,
            controlPoint,
            easeT
        );
        this.position.lerp(
            this.uiTargetPosition,
            easeT * easeT
        );

        // Scale down
        const scale = 0.15 * (1 - easeT * 0.8);
        this.scale.setScalar(scale);

        // Spin during flight
        this.rotation.y += easeT * Math.PI * 4 * deltaTime;
    }

    collect(uiPosition: Vector3) {
        if (this.state !== 'idle') return;

        this.state = 'collecting';
        this.uiTargetPosition = uiPosition;
    }

    private onCollectComplete() {
        this.scene.remove(this);
        this.geometry.dispose();
        (this.material as THREE.Material).dispose();

        // Dispatch event
        window.dispatchEvent(new CustomEvent('shell-collected', {
            detail: { chamber: 'archive-of-voices' }
        }));
    }

    isHovered(raycaster: THREE.Raycaster): boolean {
        if (this.state !== 'idle') return false;

        const intersects = raycaster.intersectObject(this);
        return intersects.length > 0;
    }
}
```

### Raycast Detection

**Input Handler:**
```javascript
class ShellInputHandler {
    private raycaster: THREE.Raycaster;
    private mouse: THREE.Vector2;
    private shell: ShellCollectible;

    onPointerMove(event: PointerEvent) {
        // Convert to normalized device coordinates
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);

        if (this.shell.isHovered(this.raycaster)) {
            document.body.style.cursor = 'pointer';
            this.shell.scale.setScalar(0.165); // Slight increase
        } else {
            document.body.style.cursor = 'default';
            this.shell.scale.setScalar(0.15);
        }
    }

    onPointerClick(event: PointerEvent) {
        this.raycaster.setFromCamera(this.mouse, this.camera);

        if (this.shell.isHovered(this.raycaster)) {
            const uiPosition = this.getUIPosition();
            this.shell.collect(uiPosition);
        }
    }

    private getUIPosition(): THREE.Vector3 {
        // Top-left UI slot in world coordinates
        return new THREE.Vector3(-0.8, 0.5, 0.5);
    }
}
```

### Memory Management

```javascript
class ShellCollectible {
    destroy() {
        this.scene.remove(this);
        this.geometry.dispose();
        (this.material as THREE.Material).dispose();
    }
}
```

### Dependencies

**Previous Story:** 1.5 Vortex Activation (platform arrival triggers spawn)

**Next Story:** 1.7 UI Overlay System (receives collection event)

**External Dependencies:**
- Three.js core: Raycaster, geometries, shaders
- Web Audio API: Sound feedback
- Story 1.7: ShellUIOverlay integration

### Save State

**Persistence:**
```javascript
class ShellManager {
    private saveData = {
        shellsCollected: {
            'archive-of-voices': false,
            'gallery-of-forms': false,
            'hydroponic-memory': false,
            'engine-of-growth': false
        }
    };

    save() {
        localStorage.setItem('vimana-shells', JSON.stringify(this.saveData));
    }

    load() {
        const saved = localStorage.getItem('vimana-shells');
        if (saved) {
            this.saveData = JSON.parse(saved);
        }
    }

    isCollected(chamber: string): boolean {
        return this.saveData.shellsCollected[chamber] || false;
    }
}
```

### References

- [Source: music-room-proto-epic.md#Story 1.6]
- [Source: gdd.md#Shell collectibles]
- [Source: narrative-design.md#Collection mechanics]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/shaders/shell-vertex.glsl` (create)
- `src/shaders/shell-fragment.glsl` (create)
- `src/entities/ShellCollectible.ts` (create)
- `src/entities/ShellManager.ts` (create)
- `src/audio/ShellAudioFeedback.ts` (create)
