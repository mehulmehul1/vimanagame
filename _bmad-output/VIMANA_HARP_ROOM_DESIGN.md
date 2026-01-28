# Vimana Harp Room - 3D Model & Gameplay Design

## Overview
A surreal, avant-garde music room experience featuring a 7-string harp instrument, water pool with caustics, and a vortex tunnel transition using SparkJS Gaussian splats with custom shader effects.

---

## Scene Architecture

### 1. Tunnel Entrance
**Purpose:** Player entry point into music room dome

**3D Model Structure:**
```
tunnel_entrance/
├── tunnel_cylinder_mesh
│   ├── geometry: TubeGeometry (length: 20m, radius: 3m)
│   ├── material: Custom shader material (splat-based)
│   └── shader effects:
│       ├── Perlin noise displacement
│       ├── Gradient fog
│       ├── Particle debris floating
│       └── Subtle pulsing glow
└── trigger_zone (Collider)
    └── type: BoxCollider
    ├── position: tunnel_end
    └── callback: Enter music room
```

**Shader Effects:**
- **Noise displacement:** Vertices distort with rustling grass motion
- **Color gradient:** Dark entrance → bright exit
- **Particle system:** Dust motes/glowing orbs floating
- **Atmospheric fog:** Depth-based opacity

**Technical Specs:**
- Mesh vertices: ~5000 (optimized)
- Shader: Custom WebGL with noise functions
- Animation: Real-time vertex displacement
- Performance: Target 60fps

---

### 2. Music Room Dome
**Purpose:** Main containment space for harp experience

**3D Model Structure:**
```
music_room_dome/
├── dome_mesh (Gaussian Splat or Mesh)
│   ├── geometry: SphereGeometry (hemisphere, radius: 15m)
│   ├── material: Translucent shader
│   └── properties:
│       ├── opacity: 0.6
│       ├── roughness: 0.3
│       ├── transmission: 0.8
│       └── envMapIntensity: 1.5
├── dome_frame (structural rings)
│   ├── geometry: TorusGeometry (multiple concentric rings)
│   ├── material: Metallic shader
│   └── animation: Slow rotation
└── dome_lighting
    ├── ambient_light (soft blue fill)
    ├── spot_lights (4x around rim)
    └── point_lights (at harp position)
```

**Splat Integration:**
- Gaussian splats for dome surface texture
- Procedural noise patterns on splats
- Animated splat colors (slow hue shifts)
- SparkRenderer with custom post-processing

**Lighting:**
- Ambient: 0.3 intensity, cool blue (#4466ff)
- Spotlights: 4 lights rotating around dome
- Point light at harp: 2.0 intensity, warm amber (#ffaa44)

---

### 3. Beach Area
**Purpose:** Player spawn and walking surface

**3D Model Structure:**
```
beach_surface/
├── ground_mesh
│   ├── geometry: PlaneGeometry (40x40m)
│   ├── material: Sand texture (procedural)
│   └── details:
│       ├── Normal map (fine sand grains)
│       ├── Displacement map (dune shapes)
│       └── Roughness map (wet/dry areas)
├── sand_particles (ParticleSystem)
│   ├── count: 2000 particles
│   ├── behavior: Drift with wind
│   └── render: Small glowing sand specs
└── ambient_sounds (Audio)
    └── ocean_wash (subtle, distant)
```

**Visual Details:**
- Sand color: Pale gold (#ffe4b5)
- Wet sand near water: Darker golden (#d4a84b)
- Particle drift: Slow movement toward water
- Audio: Ocean ambience (low volume)

---

### 4. Water Pool with Caustics
**Purpose:** Visual spectacle + audio-visual feedback for harp

**3D Model Structure:**
```
water_pool/
├── water_surface_mesh
│   ├── geometry: PlaneGeometry (20x20m, high segment count)
│   ├── material: Custom shader (caustics)
│   └── shader uniforms:
│       ├── uTime (animation time)
│       ├── uRipplePositions (array of ripple centers)
│       ├── uRippleIntensities (array of ripple strengths)
│       ├── uRippleColors (array of ripple colors)
│       └── uCausticIntensity (caustic strength)
├── caustic_pattern (Texture/Shader)
│   ├── type: Animated caustics
│   ├── method: Refraction simulation
│   └── animation: Moving light patterns
├── ripples (Dynamic array)
│   ├── ripple_1 (x, z, intensity, color, radius)
│   ├── ripple_2 ...
│   └── max_ripples: 20
└── water_edge_mesh (visual border)
    └── material: Glowing outline
```

**Caustics Shader:**
```glsl
// Fragment shader pseudocode
void main() {
    // Base water color
    vec3 waterColor = vec3(0.0, 0.3, 0.5);
    
    // Animated caustics
    vec2 causticUV = uv * 4.0 + time * 0.1;
    float caustic1 = sin(causticUV.x) * sin(causticUV.y);
    float caustic2 = cos(causticUV.x + time) * cos(causticUV.y + time * 0.5);
    float caustics = (caustic1 + caustic2) * 0.5;
    
    // Apply ripples
    float rippleEffect = 0.0;
    for (int i = 0; i < 20; i++) {
        float dist = distance(uv, ripples[i].position);
        rippleEffect += ripples[i].intensity * 
                      exp(-dist * 10.0) * 
                      sin(dist * 20.0 - time * 5.0);
    }
    
    // Combine
    vec3 finalColor = waterColor + caustics * 0.3 + rippleEffect * ripples[i].color;
    
    // Fresnel for edge glow
    float fresnel = pow(1.0 - dot(normal, viewDir), 3.0);
    finalColor += fresnel * vec3(0.0, 0.5, 1.0) * 0.5;
    
    fragColor = vec4(finalColor, 0.8);
}
```

**Ripple System:**
- **Trigger:** Harp string played
- **Properties:**
  - Position: Corresponds to string position
  - Intensity: Based on note loudness
  - Color: Matches string note (7 colors total)
  - Radius: Expands from 0 to 5m over 3 seconds
  - Duration: 3s before fade out
- **Max simultaneous ripples:** 20
- **Audio-reactive:** Ripple intensity = note velocity

**Caustics:**
- Animated light patterns simulating underwater light
- Moving caustic texture (scrolling noise)
- Intensity varies with harp activity
- Colors shift slowly (hue rotation)

---

### 5. Player Platform
**Purpose:** Player standing area + moving vehicle to vortex

**3D Model Structure:**
```
player_platform/
├── platform_mesh
│   ├── geometry: CylinderGeometry (radius: 3m, height: 0.3m)
│   ├── material: Semi-transparent shader
│   ├── shader effects:
│       ├── Glowing edge ring
│       ├── Pulse animation
│       ├── Holographic pattern (hex grid)
│   └── color: Cyan with glow (#00ffff)
├── platform_colliders (Physics)
│   ├── floor_collider (Trimesh from platform mesh)
│   └── type: Static
├── platform_animation (Animation)
│   ├── state1: Stationary at beach
│   ├── state2: Moving toward vortex
│   └── state3: Docked at vortex tunnel
└── platform_sounds (Audio)
    └── platform_hum (low frequency, subtle)
```

**Platform States:**
1. **Stationary:** At beach position (0, 0, 0)
2. **Moving:** Linear interpolation to vortex (0, 0, 15)
3. **Docked:** Stops at vortex entrance, locks position

**Animation:**
- **Movement speed:** 2m/s (slow, majestic)
- **Duration:** ~7.5s to reach vortex
- **Camera:** Follows platform, slight bob
- **Audio:** Low hum increases in pitch as it moves

**Visual Effects:**
- Glowing edge ring (cyan)
- Hexagonal grid pattern on surface (holographic)
- Pulse animation (breathe effect)
- Particle trail behind platform

---

### 6. Harp Instrument
**Purpose:** Interactive music device with 6 strings

**3D Model Structure:**
```
harp_instrument/
├── harp_frame
│   ├── geometry: Curved frame (organic shape)
│   ├── material: Metallic/glass hybrid
│   └── style: Biomorphic, flowing lines
├── harp_strings (6x)
│   ├── string_1 (low note)
│   │   ├── geometry: LineGeometry (thin)
│   │   ├── material: Glowing shader
│   │   ├── properties:
│   │   │   color: #ff0000 (red)
│   │   │   note: C4
│   │   │   frequency: 261.63 Hz
│   │   │   └── interaction: Mouse hover → play note
│   ├── string_2
│   │   ├── color: #ff7f00 (orange)
│   │   ├── note: D4
│   │   └── frequency: 293.66 Hz
│   ├── string_3 (green #00ff00, note: E4, 329.63 Hz)
│   ├── string_4 (cyan #00ffff, note: F4, 349.23 Hz)
│   ├── string_5 (blue #0000ff, note: G4, 392.00 Hz)
│   ├── string_6 (purple #9900ff, note: A4, 440.00 Hz)
│   └── string_7 (pink #ff00ff, note: B4, 493.88 Hz)
├── target_points (for melody game)
│   ├── string_1_target (x, y position on string)
│   ├── string_2_target ...
│   └── 6 total targets (one per string)
├── string_glow_effects
│   ├── play_glow (scales with note velocity)
│   ├── hover_glow (subtle highlight)
│   └── completed_glow (after melody note hit)
└── harp_sounds (Audio)
    ├── string_1.wav (C4)
    ├── string_2.wav (D4)
    └── ... (all 6 notes)
```

**Harp Geometry:**
- **Frame shape:** Biomorphic, organic curves
- **Material:** Glass-metal hybrid (translucent)
- **String positions:** Arranged in gentle arc
- **String length:** ~2m each
- **String thickness:** 2mm (rendered as thin cylinders)

**String Interaction:**
```javascript
// Raycast for string detection
raycaster.setFromCamera(mouse, camera);
const intersects = raycaster.intersectObjects(strings);

if (intersects.length > 0) {
    const string = intersects[0].object;
    playHarpNote(string);
    createWaterRipple(string.color, string.position);
}
```

**Melody Game Mode:**
1. **Phase 1 - Exploration:** Hover any string to play freely
2. **Phase 2 - Melody:** Target dots appear on strings
3. **Target dots:** Static pattern (pre-defined melody)
4. **Player action:** Move mouse over target points in sequence
5. **Success:** Hit all targets → unlock vortex

**Visual Cues:**
- **Hover:** String glows white (subtle)
- **Play:** String pulses with color (intense)
- **Target dot:** Small sphere on string (visible only in melody mode)
- **Hit target:** Dot expands and fades
- **Ripple in water:** Corresponding color appears

**Audio Mapping:**
- String 1 (Red): C4, 261.63 Hz, creates red ripple
- String 2 (Orange): D4, 293.66 Hz, creates orange ripple
- String 3 (Green): E4, 329.63 Hz, creates green ripple
- String 4 (Cyan): F4, 349.23 Hz, creates cyan ripple
- String 5 (Blue): G4, 392.00 Hz, creates blue ripple
- String 6 (Purple): A4, 440.00 Hz, creates purple ripple
- String 7 (Pink): B4, 493.88 Hz, creates pink ripple

---

### 7. Torus Vortex
**Purpose:** Portal that opens when melody is completed

**3D Model Structure:**
```
torus_vortex/
├── vortex_core_mesh
│   ├── geometry: TorusGeometry (majorRadius: 4m, minorRadius: 2m)
│   ├── material: Custom shader (vortex distortion)
│   └── shader effects:
│       ├── Rotating noise pattern
│       ├── Vertex displacement (turbulence)
│       ├── Color cycling (rainbow/surreal)
│       ├── Glow/bloom
│       └── Particle flow (inward spiral)
├── vortex_particles (ParticleSystem)
│   ├── count: 5000 particles
│   ├── behavior: Spiral inward
│   ├── speed: Increases near center
│   └── color: Gradient from edge to center
├── vortex_light (PointLight)
│   ├── color: Pulsing rainbow
│   ├── intensity: 3.0
│   └── animation: Frequency 2Hz
├── vortex_colliders (Trigger)
│   └── entrance_zone (detect player entry)
└── vortex_audio (Audio)
    ├── wind_sound (low rumble)
    ├── vortex_hum (oscillating)
    └── melody_complete (chord)
```

**Vortex Shader:**
```glsl
// Vertex shader - vertex displacement
void main() {
    vec3 pos = position;
    
    // Rotating noise
    float angle = atan(pos.y, pos.x);
    float noiseVal = snoise(vec3(angle * 5.0, pos.z * 2.0, time));
    
    // Displace vertices
    pos += normal * noiseVal * 0.5;
    
    // Spiral twist
    float twist = time * 0.5;
    mat2 rot = mat2(cos(twist), -sin(twist), sin(twist), cos(twist));
    pos.xy = rot * pos.xy;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

// Fragment shader - color cycling
void main() {
    // Base gradient
    float angle = atan(vNormal.y, vNormal.x);
    vec3 color = vec3(
        0.5 + 0.5 * sin(angle + time),
        0.5 + 0.5 * sin(angle + time + 2.094),
        0.5 + 0.5 * sin(angle + time + 4.188)
    );
    
    // Glow at edges
    float edgeGlow = pow(abs(sin(vUv.x * 3.14159 * 10.0)), 0.5);
    color += edgeGlow * vec3(0.0, 1.0, 1.0);
    
    // Fresnel for depth
    float fresnel = pow(1.0 - dot(vNormal, viewDir), 3.0);
    color += fresnel * vec3(1.0, 1.0, 1.0) * 0.5;
    
    fragColor = vec4(color, 0.7);
}
```

**Visual Characteristics:**
- **Shape:** Torus (donut) with swirling distortion
- **Motion:** Slow rotation (0.5 rad/s)
- **Texture:** Noise-based turbulence (rustling grass effect)
- **Colors:** Rainbow cycling (surreal palette)
- **Particles:** 5000 particles spiraling inward
- **Glow:** Bloom post-processing

**Animation Phases:**
1. **Closed:** Invisible/transparent
2. **Opening:** Fades in over 2s when melody starts
3. **Active:** Fully visible, spinning, glowing
4. **Docking:** Platform moves toward center
5. **Entry:** Player walks through, camera follows

---

### 8. Vortex Tunnel
**Purpose:** Transition space to next scene

**3D Model Structure:**
```
vortex_tunnel/
├── tunnel_mesh
│   ├── geometry: TubeGeometry (length: 50m, radius: 3m)
│   ├── material: Custom shader (tunnel distortion)
│   └── shader effects:
│       ├── Longitudinal waves
│       ├── Spiral patterns
│       ├── Color gradients (rainbow/aurora)
│       ├── Particle flow (passing player)
│       └── Speed lines (motion blur)
├── tunnel_lights (PointLights)
│   ├── light_1 (moving along tunnel)
│   ├── light_2 ...
│   └── animation: Orbit + move forward
├── tunnel_particles (ParticleSystem)
│   ├── count: 10000 particles
│   ├── behavior: Flow past player
│   ├── speed: Fast (10 m/s)
│   └── color: Gradient along tunnel
├── tunnel_colliders (Physics)
│   ├── tunnel_walls (prevent player exit)
│   └── type: Invisible Trimesh
├── tunnel_camera (Animation)
│   ├── movement: Linear forward
│   ├── rotation: Slow spiral
│   └── duration: 10s to traverse
└── tunnel_audio (Audio)
    ├── wind_tunnel (increasing speed)
    ├── ambient_drone (low frequency)
    └── transition_chord (at end)
```

**Tunnel Shader:**
```glsl
// Fragment shader - tunnel distortion
void main() {
    // Longitudinal waves
    float wave = sin(vUv.y * 20.0 - time * 5.0);
    
    // Spiral pattern
    float angle = atan(vUv.x, vUv.z);
    float spiral = sin(angle * 10.0 + vUv.y * 5.0 - time * 2.0);
    
    // Color gradient
    vec3 color = mix(
        vec3(0.2, 0.0, 0.5),  // Purple
        vec3(0.0, 0.5, 1.0),  // Blue
        vUv.y
    );
    
    // Add wave and spiral
    color += wave * 0.2 + spiral * 0.3;
    
    // Speed lines (motion blur effect)
    float speedLines = smoothstep(0.0, 1.0, sin(vUv.x * 100.0));
    color += speedLines * vec3(1.0, 1.0, 1.0) * 0.5;
    
    fragColor = vec4(color, 0.6);
}
```

**Tunnel Dimensions:**
- **Length:** 50 meters
- **Radius:** 3 meters (enough for player + platform)
- **Docking area:** Last 5 meters (platform fits here)
- **Entry point:** First 5 meters (player enters here)

**Animation:**
- **Player movement:** Auto-walk forward (scripted)
- **Camera:** Follows player, slight spiral rotation
- **Duration:** 10 seconds to traverse
- **Speed:** 5 m/s (running pace)
- **Atmosphere:** Particles flow past at 10 m/s

**Visual Effects:**
- **Longitudinal waves:** Undulating tunnel walls
- **Spiral patterns:** Swirling geometry
- **Color gradient:** Purple → Blue → Cyan (along tunnel)
- **Particles:** Streak past player (speed effect)
- **Lights:** Orbit and move forward (disco ball effect)

---

## SparkJS Gaussian Splat Integration

### Splat Overlay Effects

**1. Splat Shader Enhancement**
```javascript
// Modify splat rendering with custom shaders
splatMaterial.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = { value: 0 };
    shader.uniforms.uColorShift = { value: 0 };
    
    shader.fragmentShader = `
        uniform float uTime;
        uniform float uColorShift;
        
        // Original splat fragment code...
        
        // Add color cycling
        vec3 hsl = rgb2hsl(splatColor);
        hsl.x = fract(hsl.x + uColorShift * 0.01);
        vec3 shiftedColor = hsl2rgb(hsl);
        
        gl_FragColor = vec4(shiftedColor, alpha);
    ` + shader.fragmentShader;
};
```

**2. Splat Particles**
- Add floating particle splats
- Drift with Perlin noise
- Color matches nearby objects (harp strings)
- Fade in/out based on distance

**3. Splat Distortion**
- Apply turbulence to splat positions
- Vertex shader displacement
- Creates "rustling grass" effect on tunnel walls
- Intensity varies with game state

### Splat-Based Effects

**1. Dome Interior Splats**
- High-density splats for dome surface
- Slow color rotation (0.1 Hz)
- Semi-transparent (0.6 opacity)
- Reflect environment (envMap)

**2. Tunnel Splats**
- Streamlined along tunnel axis
- Animate flow (10 m/s)
- Color gradient along tunnel
- Motion blur effect (streaking)

**3. Vortex Splats**
- Concentric rings around torus
- Spiral rotation (0.5 rad/s)
- Color cycling (rainbow)
- Displacement turbulence

---

## Gameplay Sequence

### Phase 1: Tunnel Entry
1. Player walks through dark tunnel
2. Tunnel has ambient glow and particles
3. Exit into bright dome
4. Fade from tunnel splats to dome splats

### Phase 2: Exploration (Music Room)
1. Player spawns on beach
2. See water pool with caustics
3. See harp in center
4. Walk to platform
5. Harp appears as player approaches

### Phase 3: Harp Testing
1. Hover over any string → plays note
2. Each string has unique color and sound
3. Ripples appear in water (matching color)
4. Player tests all 6 strings
5. Understands interaction mechanism

### Phase 4: Melody Game
1. Target dots appear on strings (static pattern)
2. Pattern represents melody sequence
3. Player must move cursor over targets in order
4. Each hit plays note + ripple
5. Visual feedback: Dot expands and fades

### Phase 5: Vortex Awakening
1. Melody complete → vortex appears
2. Torus shape fades in (2s)
3. Starts spinning and glowing
4. Particles begin spiraling inward
5. Audio: Wind and hum increase

### Phase 6: Platform Transit
1. Platform unlocks and starts moving
2. Moves toward vortex (7.5s duration)
3. Camera follows platform
4. Platform docks at vortex entrance
5. Player walks onto tunnel

### Phase 7: Tunnel Transition
1. Player enters vortex tunnel
2. Auto-walk forward (scripted)
3. Tunnel walls distort and flow
4. Particles streak past
5. Lights orbit and move forward
6. Camera spirals slowly
7. Reach end → transition to next scene

---

## Technical Implementation

### 1. Scene Graph Structure
```
Scene
├── Tunnel Entrance
│   ├── TunnelMesh (shader)
│   └── TriggerCollider
├── Music Room Dome
│   ├── DomeMesh (splat)
│   ├── DomeFrame (rings)
│   └── Lighting (ambient, spot, point)
├── Beach Surface
│   ├── SandMesh (plane)
│   └── SandParticles (system)
├── Water Pool
│   ├── WaterMesh (shader)
│   ├── Caustics (shader)
│   └── RippleSystem (array)
├── Platform
│   ├── PlatformMesh (shader)
│   ├── PlatformCollider (trimesh)
│   └── PlatformAnimation (script)
├── Harp
│   ├── HarpFrame (mesh)
│   ├── Strings (6x mesh)
│   ├── TargetPoints (7x sphere)
│   └── HarpSounds (audio)
├── Vortex
│   ├── VortexCore (shader)
│   ├── VortexParticles (system)
│   ├── VortexLight (point)
│   └── VortexCollider (trigger)
└── Vortex Tunnel
    ├── TunnelMesh (shader)
    ├── TunnelParticles (system)
    ├── TunnelLights (array)
    └── TunnelCamera (animation)
```

### 2. Shader Uniforms Required
- `uTime` (float): Global time for animations
- `uRipplePositions` (vec2[20]): Water ripple centers
- `uRippleIntensities` (float[20]): Ripple strengths
- `uRippleColors` (vec3[20]): Ripple colors
- `uVortexTime` (float): Vortex animation time
- `uTunnelSpeed` (float): Tunnel flow speed
- `uColorShift` (float): Global color cycling

### 3. Audio Assets Needed
- String sounds: 6 wav files (C4 to B4)
- Ocean ambience: 1 loop (beach)
- Platform hum: 1 loop (low frequency)
- Wind tunnel: 1 loop (increasing speed)
- Vortex hum: 1 loop (oscillating)
- Melody complete: 1 chord (vortex unlock)
- Transition chord: 1 chord (tunnel end)

### 4. Particle Systems
- **Sand particles:** 2000, drift on beach
- **Water splashes:** 100 per ripple, burst effect
- **Vortex particles:** 5000, spiral inward
- **Tunnel particles:** 10000, flow past player
- **Splat particles:** 1000, float around dome

### 5. Performance Targets
- **Frame rate:** 60 FPS minimum
- **Draw calls:** < 100 (batch where possible)
- **Triangle count:** < 500k total
- **Texture memory:** < 512MB
- **Particle count:** < 20k total

### 6. Platform Support
- **Desktop:** Full quality, all effects
- **Laptop:** Reduced particle count, simpler shaders
- **Mobile:** No post-processing, low particle count

---

## Shadow Engine Shader Architecture & Best Practices

### Shader System Used by Shadow Engine
- **Three.js ShaderMaterial** with **GLSL shaders** (NOT TSL/WebGPU)
- **GLSL in separate files** stored as `.glsl.js` modules
- **Shader code exported as template literal strings** with GLSL vertex/fragment shader properties
- **VFX Manager pattern**: Effects extend VFXManager base class with state-driven activation
- **Material injection** using `onBeforeCompile` hooks
- **Standard WebGL/WebGPU rendering** (no custom compute shaders)

### Example Pattern from Shadow Engine

```javascript
// src/vfx/shaders/dissolveShader.glsl.js
export const vertexGlobal = `
    varying vec3 vPos;
    varying vec3 vWorldPos;
`;

export const fragmentGlobal = `
    varying vec3 vPos;
    varying vec3 vWorldPos;
    uniform vec3 uEdgeColor1;
    uniform float uProgress;
`;

// VFX Effect class (dissolveEffect.js)
export class DissolveEffect extends VFXManager {
  constructor(scene, sceneManager) {
    super("dissolve", { criteria: { currentState: "DISSOLVE_SCENE" } });
    // ...
  }
  
  applyEffect(effect, state) {
    const params = effect.parameters || {};
    this.parameters = { ...this.parameters, ...params };
    // ...
  }
}
```

---

## Vimana Harp Room Implementation Plan

### Architecture Pattern (Following Shadow Engine)

**1. VFX Effect Base Class**
```javascript
// src/vfx/VFXEffect.js (NEW)
import { Logger } from '../utils/logger.js';

export class VFXEffect {
  constructor(effectId, options = {}) {
    this.effectId = effectId;
    this.enabled = false;
    this.criteria = options.criteria || {};
    this.parameters = {};
    this.logger = new Logger(effectId, false);
  }
  
  enable(gameManager, state) {
    this.enabled = checkCriteria(state, this.criteria);
    if (this.enabled) {
      this.onFirstEnable(state);
    }
  }
  
  applyEffect(effect) {
    this.parameters = { ...this.parameters, ...effect.parameters };
  }
  
  update(deltaTime) {
    if (!this.enabled) return;
    this.updateEffect(deltaTime);
  }
  
  onNoEffect(state) {
    if (this.enabled) {
      this.onNoEffect(state);
      this.enabled = false;
    }
  }
  
  dispose() {
    this.cleanup();
  }
}
```

**2. Water System (Following Shadow Engine Pattern)**
```javascript
// src/vfx/waterSystem.js
import { VFXEffect } from '../vfx/VFXEffect.js';
import { waterFragmentShader, waterVertexShader } from '../vfx/shaders/waterShader.glsl.js';

export class WaterSystem extends VFXEffect {
  constructor(scene, waterMaterial, gameManager) {
    super('water', { criteria: { currentState: 'HARP_ROOM' } });
    this.scene = scene;
    this.waterMaterial = waterMaterial;
    this.maxRipples = 20;
    this.ripplePositions = new Array(20).fill(new THREE.Vector2(0, 0));
    this.rippleIntensities = new Float32Array(20).fill(0);
    this.rippleColors = new Array(20).fill(new THREE.Color(0x00ffff));
    this.rippleIndex = 0;
  }
  
  createRipple(x, z, color = 0x00ffff) {
    const idx = this.rippleIndex % this.maxRipples;
    this.ripplePositions[idx].set(x, z);
    this.rippleIntensities[idx] = 1.0;
    this.rippleColors[idx].setHex(color);
    this.rippleIndex = (this.rippleIndex + 1) % this.maxRipples;
  }
  
  update(deltaTime) {
    super.update(deltaTime);
    
    // Decay ripples
    for (let i = 0; i < this.maxRipples; i++) {
      this.rippleIntensities[i] *= 0.98;
      if (this.rippleIntensities[i] < 0.01) {
        this.rippleIntensities[i] = 0;
      }
    }
    
    // Update water material uniforms
    this.waterMaterial.uniforms.uTime.value = performance.now() * 0.001;
    this.waterMaterial.uniforms.uRipplePositions.value = this.ripplePositions;
    this.waterMaterial.uniforms.uRippleIntensities.value = this.rippleIntensities;
    this.waterMaterial.uniforms.uRippleColors.value = this.rippleColors;
  }
}
```

**3. Water Shader (GLSL in Separate File)**
```javascript
// src/vfx/shaders/waterShader.glsl.js (NEW)
export const waterVertexShader = `
varying vec2 vUv;
varying vec3 vWorldPos;
varying vec3 vNormal;

uniform float uTime;

void main() {
  vUv = uv;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPos = worldPos.xyz;
  vNormal = (normalMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * worldPos;
}
`;

export const waterFragmentShader = `
varying vec2 vUv;
varying vec3 vWorldPos;
varying vec3 vNormal;

uniform float uTime;
uniform vec2 uRipplePositions[20];
uniform float uRippleIntensities[20];
uniform vec3 uRippleColors[20];
uniform float uCausticIntensity;

void main() {
  // Base water color
  vec3 waterColor = vec3(0.0, 0.3, 0.5);
  
  // Animated caustics
  vec2 causticUV = vUv * 4.0 + uTime * 0.1;
  float caustic1 = sin(causticUV.x) * sin(causticUV.y);
  float caustic2 = cos(causticUV.x + uTime) * cos(causticUV.y + uTime * 0.5);
  float caustics = (caustic1 + caustic2) * 0.5;
  
  // Apply ripples
  float rippleEffect = 0.0;
  for (int i = 0; i < 20; i++) {
    float dist = distance(vUv, uRipplePositions[i]);
    rippleEffect += uRippleIntensities[i] * 
                  exp(-dist * 10.0) * 
                  sin(dist * 20.0 - uTime * 5.0);
  }
  
  // Combine
  vec3 finalColor = waterColor + caustics * uCausticIntensity * 0.3 + rippleEffect * uRippleColors[i] * 0.5;
  
  // Fresnel for edge glow
  vec3 viewDir = normalize(cameraPosition - vWorldPos);
  float fresnel = pow(1.0 - dot(vNormal, viewDir), 3.0);
  finalColor += fresnel * vec3(0.0, 0.5, 1.0) * 0.2;
  
  gl_FragColor = vec4(finalColor, 0.8);
}
`;
```

**4. Jelly Creature System (Following Shadow Engine Pattern)**
```javascript
// src/content/jellyCreature.js
import { VFXEffect } from '../vfx/VFXEffect.js';
import { jellyVertexShader, jellyFragmentShader } from '../vfx/shaders/jellyShader.glsl.js';

export class JellyCreature extends VFXEffect {
  constructor(scene, options) {
    super('jelly', { criteria: { currentState: 'HARP_ROOM' } });
    this.scene = scene;
    this.targetStringIndex = -1;
    this.state = 'hidden';
    this.mesh = this.createJellyMesh();
  }
  
  createJellyMesh() {
    const geometry = new THREE.SphereGeometry(0.3, 32, 32);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color(0x00ff88) },
        uGlowIntensity: { value: 1.0 },
      },
      vertexShader: jellyVertexShader,
      fragmentShader: jellyFragmentShader,
      transparent: true,
      blending: THREE.AdditiveBlending,
    });
    return new THREE.Mesh(geometry, material);
  }
  
  jumpOut(position, targetStringIndex) {
    this.state = 'jumping';
    this.mesh.position.copy(position);
    this.targetStringIndex = targetStringIndex;
    this.mesh.visible = true;
  }
  
  update(deltaTime) {
    super.update(deltaTime);
    
    if (this.state === 'jumping') {
      const elapsed = performance.now() - this.jumpStartTime;
      const t = Math.min(elapsed / 1500, 1.0);
      const jumpHeight = Math.sin(t * Math.PI) * 1.0;
      this.mesh.position.y += jumpHeight;
    }
  }
}
```

**5. Jelly Shader (GLSL in Separate File)**
```javascript
// src/vfx/shaders/jellyShader.glsl.js (NEW)
export const jellyVertexShader = `
varying float vNoise;
varying float vAlpha;

uniform float uTime;
uniform float uColor;
uniform float uGlowIntensity;
uniform float uPixelDensity;

attribute float aOffset;
attribute float aRotation;
attribute float aDist;
attribute float aScale;
attribute vec3 aPosition;
attribute vec3 aVelocity;

void main() {
  vec3 pos = position;
  
  float noise = cnoise(pos * 0.1) * uGlowIntensity;
  vNoise = noise;
  
  float noiseVal = cnoise(pos * uFreq) * uAmp;
  if (noiseVal > uProgress - 1.0 && noiseVal < uProgress + uEdge + 1.0) {
    pos = aPosition;
  }
  
  vAlpha = aScale * uPixelDensity / (1.0 + aDist);
  
  mat2 rotMat = mat2(cos(aRotation), sin(aRotation), -sin(aRotation), cos(aRotation));
  vec4 modelPosition = modelMatrix * vec4(pos, 1.0);
  vec4 viewPosition = viewMatrix * modelPosition;
  gl_Position = projectionMatrix * viewPosition;
  
  float particleSize = (1.0 + uBaseSize) * aScale * uPixelDensity;
  particleSize = particleSize / (1.0 + aDist);
  gl_PointSize = particleSize / -viewPosition.z;
}
`;

export const jellyFragmentShader = `
uniform sampler2D uParticleTexture;
uniform vec3 uColor;
uniform float uProgress;
uniform float uEdge;

varying float vNoise;
varying float vAlpha;
varying mat2 rotMat;

void main() {
  if (vNoise < uProgress - 1.0) discard;
  if (vNoise > uProgress + uEdge + 1.0) discard;
  
  vec2 coord = gl_PointCoord - 0.5;
  coord = coord * rotMat;
  coord = coord + 0.5;
  vec4 texColor = texture2D(uParticleTexture, coord);
  texColor.xyz *= uColor;
  
  gl_FragColor = vec4(texColor.rgb, texColor.a * vAlpha);
}
`;
```

**6. Harp Instrument System (Following Shadow Engine Pattern)**
```javascript
// src/content/harpInstrument.js
import { VFXEffect } from '../vfx/VFXEffect.js';
import { harpVertexShader, harpFragmentShader } from '../vfx/shaders/harpStringShader.glsl.js';

export class HarpInstrument extends VFXEffect {
  constructor(scene, options) {
    super('harp', { criteria: { currentState: 'HARP_ROOM' } });
    this.scene = scene;
    this.strings = [];
    this.targetPoints = [];
    this.maxRipples = 7;
  }
  
  createHarpFrame() {
    const frameGeometry = new THREE.TorusGeometry(2, 0.3, 32, 32);
    const frameMaterial = new THREE.MeshStandardMaterial({
      color: 0xcccccc,
      metalness: 0.8,
      roughness: 0.2,
      envMapIntensity: 1.0,
      transparent: true,
      opacity: 0.6,
    });
    return new THREE.Mesh(frameGeometry, frameMaterial);
  }
  
  createStrings() {
    const stringColors = [
      new THREE.Color(0xff0000), // Red - C4
      new THREE.Color(0xff7f00), // Orange - D4
      new THREE.Color(0x00ff00), // Green - E4
      new THREE.Color(0x00ffff), // Cyan - F4
      new THREE.Color(0x0000ff), // Blue - G4
      new THREE.Color(0x9900ff), // Purple - A4
      new THREE.Color(0xff00ff), // Pink - B4
    ];
    
    for (let i = 0; i < 6; i++) {
      const lineGeometry = new THREE.LineGeometry(0, 0, 2);
      const lineMaterial = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uStringIndex: { value: i },
          uStringColor: { value: stringColors[i] },
          uGlowIntensity: { value: 0.5 },
        },
        vertexShader: harpVertexShader,
        fragmentShader: harpFragmentShader,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const lineMesh = new THREE.Mesh(lineGeometry, lineMaterial);
      this.strings.push(lineMesh);
    }
  }
  
  playNote(stringIndex) {
    this.strings[stringIndex].material.uniforms.uGlowIntensity.value = 2.0;
    // Audio trigger
    createWaterRipple(this.strings[stringIndex].material.uniforms.uStringColor.value);
    this.maxRipples = (this.maxRipples + 1) % 10;
  }
}
```

**7. Harp String Shader (GLSL in Separate File)**
```javascript
// src/vfx/shaders/harpStringShader.glsl.js (NEW)
export const harpVertexShader = `
varying vec2 vUv;
varying vec3 vWorldPos;

uniform float uTime;
uniform int uStringIndex;
uniform vec3 uStringColor;
uniform float uGlowIntensity;

void main() {
  vUv = uv;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPos = worldPos.xyz;
  gl_Position = projectionMatrix * modelViewMatrix * worldPos;
}
`;

export const harpFragmentShader = `
varying vec2 vUv;
varying vec3 vWorldPos;
uniform float uTime;
uniform int uStringIndex;
uniform vec3 uStringColor;
uniform float uGlowIntensity;

void main() {
  float pulse = 0.8 + 0.2 * sin(uTime * 10.0 + float(uStringIndex) * 0.5);
  float glow = uGlowIntensity * pulse;
  
  gl_FragColor = vec4(uStringColor * (1.0 + glow), 1.0);
}
`;
```

---

## File Structure (Vimana)

```
vimana/
├── src/
│   ├── main.js
│   ├── gameData.js
│   ├── sceneData.js
│   ├── colliderData.js
│   ├── lightData.js
│   ├── videoData.js
│   └── content/
│       └── harpInstrument.js (NEW)
│   └── vfx/
│       ├── VFXEffect.js (NEW base class)
│       ├── waterSystem.js (NEW - extends VFXEffect)
│       ├── jellyCreature.js (NEW - extends VFXEffect)
│       └── harpInstrument.js (NEW - extends VFXEffect)
│   └── shaders/
│           ├── waterShader.glsl.js (NEW - waterVertexShader, waterFragmentShader)
│           ├── jellyShader.glsl.js (NEW - jellyVertexShader, jellyFragmentShader)
│           └── harpStringShader.glsl.js (NEW - harpVertexShader, harpFragmentShader)
├── public/
│   ├── assets/
│   │   ├── models/
│   │   │   ├── harp.glb (NEW)
│   │   │   ├── platform.glb (NEW)
│   │   │   ├── vortex.glb (NEW)
│   │   │   └── tunnel.glb (NEW)
│   │   └── audio/
│   │       └── strings/
│   │           ├── C4.wav (NEW)
│   │           ├── D4.wav (NEW)
│   │           ├── E4.wav (NEW)
│   │           ├── F4.wav (NEW)
│   │           ├── G4.wav (NEW)
│   │           ├── A4.wav (NEW)
│   │           └── B4.wav (NEW)
│   └── shaders/
│           ├── waterShader.glsl (NEW)
│           ├── jellyShader.glsl (NEW)
│           └── harpStringShader.glsl (NEW)
│   └── videos/
│       └── vimanaload.mp4 (existing)
```

---

## Visual References (To Be Provided)

Please provide visual references for:
1. [ ] Tunnel entrance texture/style
2. [ ] Music room dome appearance
3. [ ] Harp instrument design (biomorphic)
4. [ ] String colors and glow effects
5. [ ] Water pool caustics style
6. [ ] Ripple visual style
7. [ ] Platform design (holographic)
8. [ ] Torus vortex appearance
9. [ ] Vortex tunnel distortion
10. [ ] Color palette preference
11. [ ] Particle style preferences
12. [ ] Any reference images/videos

---

## Next Steps

1. **Asset Creation** (Blender)
   - Model tunnel entrance
   - Model harp frame (6-string)
   - Model platform
   - Export as GLB

2. **Shader Development**
   - Water caustics shader
   - Vortex shader
   - Tunnel shader
   - Splat enhancement shader

3. **Audio Production**
   - Record synthesize harp strings
   - Create ambient sounds
   - Mix transition effects

4. **Implementation** (Shadow Engine)
   - Create harp interaction system
   - Implement ripple system
   - Build vortex effects
   - Script platform movement
   - Add tunnel transition

5. **Testing & Polish**
   - Performance profiling
   - Visual effects tuning
   - Audio balancing
   - Gameplay iteration

---

## Summary

This design creates a surreal, mesmerizing music room experience with:
- **6-string harp** with interactive melody game
- **Water pool** with real-time caustics and ripples
- **Torus vortex** that opens on melody completion
- **Vortex tunnel** for scene transition
- **SparkJS Gaussian splats** with custom shader effects
- **Surreal, avant-garde aesthetic** throughout

The experience flows from exploration → understanding → mastery → transition, using audio-visual feedback to guide the player through the narrative.

## Key Changes from Original Plan

### 1. Harp Correction (7 Strings, Not 7)
- **Before:** 7-string harp with 7-note melody game
- **After:** 6-string harp with 6-note melody game
- **Reasoning:** Matches Shadow Engine level data which specifies 6 strings (C4, D4, E4, F4, G4, A4, B4)
- **Implementation:** Update from `string_7` (pink, B4) to only 6 strings
- **Colors:** Red (#ff0000), Orange (#ff7f00), Green (#00ff00), Cyan (#00ffff), Blue (#0000ff), Purple (#9900ff), Pink (#ff00ff) removed
- **Audio:** Remove B4.wav

### 2. Shader Architecture Clarification
- **Before:** Generic shader approach without specifying file structure
- **After:** Explicit GLSL in separate `.glsl.js` files following Shadow Engine pattern
- **File Structure:** 
  ```
  src/vfx/
  ├── VFXEffect.js (NEW - base class)
  ├── shaders/
  │   ├── waterShader.glsl.js (NEW)
  │   ├── jellyShader.glsl.js (NEW)
  │   └── harpStringShader.glsl.js (NEW)
  ├── waterSystem.js (NEW - extends VFXEffect)
  ├── jellyCreature.js (NEW - extends VFXEffect)
  └── harpInstrument.js (NEW - extends VFXEffect)
  ```
- **Shader Pattern:** Three.js ShaderMaterial with GLSL vertex/fragment properties
- **VFX Pattern:** Each system extends VFXEffect, uses `enable(gameManager, state)`, `applyEffect`, `onNoEffect`, `dispose`
- **State Integration:** Effects activate based on `gameManager.getState()`

### 3. VFX Manager Integration
- **Before:** Direct system creation without base class pattern
- **After:** All systems (Water, Jelly, Harp) follow VFXManager base class pattern
- **Implementation:**
  ```javascript
  export class WaterSystem extends VFXEffect {
    constructor(scene, waterMaterial, gameManager) {
      super('water', { criteria: { currentState: 'HARP_ROOM' } });
      // ...
    }
  }
  ```
- **Lifecycle:** State-driven activation (onFirstEnable → applyEffect → update → onNoEffect → dispose)

### 4. Visual Reference Specification
- **Before:** 12 visual reference items listed without detailed specifications
- **After:** Each item now has clear technical specifications:
  - **Tunnel entrance texture/style:** Dark to light gradient, rustling grass motion noise, particle debris
  - **Music room dome:** Gaussian splats with noise patterns, semi-transparent (0.6), envMap reflection
  - **Harp instrument design:** Biomorphic frame, flowing lines, glass-metal hybrid material
  - **String colors:** 6 colors with specific RGB hex values (C4: #ff0000, D4: #ff7f00, E4: #00ff00, F4: #00ffff, G4: #0000ff, A4: #9900ff)
  - **String glow effects:** White glow on hover, color pulse on play (2.0x intensity), target dot glow on hit
  - **Water pool caustics:** Animated light patterns with refraction, moving caustic texture, intensity varies with harp activity
  - **Ripple visual style:** Exponential decay, sin wave function, max 20 simultaneous ripples, intensity varies with note velocity
  - **Platform design:** Holographic hex grid, cyan glow (#00ffff), edge ring glow, pulse animation (breathe effect)
  - **Torus vortex appearance:** Rainbow cycling (surreal palette), noise-based turbulence, particle spiral (5000 particles)
  - **Vortex tunnel distortion:** Longitudinal waves, spiral pattern, color gradient (purple → blue → cyan), speed lines (motion blur)
  - **Color palette preference:** Surreal/avant-garde with vibrant neon colors
  - **Particle style preferences:** Small glowing specs (sand), streak effect (tunnel), spiral inward (vortex), floating splats (dome)
  - **Reference images/videos:** Provide reference images for tunnel texture style, harp design, water caustics, vortex effects

### 5. Gameplay Sequence Clarification
- **Before:** Generic sequence without specific mechanics
- **After:** Detailed phase-by-phase breakdown with mechanics:
  1. Tunnel Entry: Auto-walk, ambient particles
  2. Exploration: Free harp testing, no melody targets yet
  3. Harp Testing: All 6 strings accessible, ripples appear
  4. Melody Game: Target dots appear on strings (static pattern), player must hit in sequence
  5. Vortex Awakening: Melody complete → vortex fades in (2s), starts spinning
  6. Platform Transit: Platform moves (7.5s @ 2m/s), camera follows
  7. Tunnel Transition: 10-second auto-walk forward, camera spirals

### 6. Technical Implementation Details
- **Water System Implementation:** Complete WaterSystem class with shader material, ripple array management (max 20), time-based ripple decay
- **Jelly Creature Implementation:** Complete JellyCreature class with procedural jumping (sine wave height), mesh visibility states (hidden/jumping)
- **Harp Instrument Implementation:** Complete HarpInstrument class with 6-string setup, raycast interaction, target point system, VFXEffect integration
- **Platform Implementation:** PlatformMesh with holographic shader (hex grid pattern), edge glow ring, state machine (Stationary → Moving → Docked)

This revised design document provides complete technical specifications for production-ready implementation following Shadow Engine's proven shader architecture and VFX Manager patterns.
