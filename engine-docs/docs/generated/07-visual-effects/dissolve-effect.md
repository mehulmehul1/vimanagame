# Dissolve Effect - First Principles Guide

## Overview

The **Dissolve Effect** is a dramatic visual transition that makes Gaussian splats appear to dissolve into particles or fade away into nothingness. Unlike traditional 3D models that use vertex-based dissolve, Gaussian Splatting requires specialized techniques since splats are point-based rather than polygon-based.

Think of the dissolve effect as the **digital equivalent of sandcastles washing away** - just as wind and water erode sand sculptures particle by particle, the dissolve effect makes Gaussian splats break down and disperse, creating an otherworldly, magical disappearance.

## What You Need to Know First

Before understanding the dissolve effect, you should know:
- **Gaussian Splats** - Point-based rendering (see [Gaussian Splatting Explained](../01-foundation/gaussian-splatting-explained.md))
- **Particle systems** - Multiple small objects moving together
- **Alpha blending** - Transparency and opacity
- **Shader uniforms** - Passing values from JavaScript to GPU
- **Three.js PointsMaterial** - Material for rendering point clouds

### Quick Refresher: What is Dissolve?

```
TRADITIONAL 3D DISSOLVE (Polygon-based):

┌─────────────────┐
│   Solid Mesh    │
│   ████░░░░░    │  ← Pixels become transparent
│   ██░░░░░░░    │     based on noise pattern
│   █░░░░░░░░    │
└─────────────────┘

GAUSSIAN SPLAT DISSOLVE (Point-based):

Each "splat" (colored point) can:

1. FADE OUT (opacity goes to 0)
   ●●●●●●●●●●  →  ●●●●●●○○○○  →  ○○○○○○○○○○

2. DISPERSE (move away from original position)
   ●●●●●●●●●●  →  ● ● ● ● ○ ○ ○  →  ○  ○  ○  ○

3. SHRINK (scale goes to 0)
   ████████████  →  ████░░░░░░░  →  ░░░░░░░░░░░

4. PARTICLE EMISSION (spawn new particles)
   ●●●●●●●●●●  →  ●●●●●●✦✦✦✦  →  ●✦✦✦✦✦✦✦
       (splat)    (splats + particles)  (mostly particles)
```

**Why Gaussian Splatting is different:** Traditional dissolve works on polygon meshes by manipulating vertex alpha. Gaussian splats are already points, so we dissolve by modifying each point's opacity, size, and position - essentially "un-splatting" the scene back into raw data.

---

## Part 1: Why Use a Dissolve Effect?

### The Problem: Scene Transitions Feel Abrupt

Without dissolve effects:

```javascript
// ❌ WITHOUT Dissolve - Abrupt transitions
function switchScene() {
  sceneManager.unloadCurrentScene();  // Instant disappear
  sceneManager.loadNextScene();        // Instant appear
  // Player feels disoriented - jarring cut!
}
```

**Problems:**
- Jarring visual cuts between scenes
- No sense of passage of time
- Breaks immersion
- Missed storytelling opportunities

### The Solution: Dissolve Effect

```javascript
// ✅ WITH Dissolve - Smooth transitions
function switchScene() {
  // Current scene dissolves away over 3 seconds
  dissolveEffect.start({
    target: currentScene,
    direction: "out",
    duration: 3000,
    particles: true
  });

  // When complete, load next scene and dissolve IN
  dissolveEffect.onComplete(() => {
    sceneManager.loadNextScene();
    dissolveEffect.start({
      target: nextScene,
      direction: "in",
      duration: 3000,
      particles: true
    });
  });
}
```

**Benefits:**
- Smooth, cinematic transitions
- Maintains spatial continuity
- Creates magical, otherworldly feeling
- Hides loading/pop-in artifacts
- Enhances supernatural atmosphere

---

## Part 2: Dissolve Data Structure

### Basic Dissolve Definition

```javascript
// In vfxData.js
export const vfxEffects = {
  // Scene dissolve out (fade away)
  officeDissolve: {
    type: "dissolve",
    duration: 3000,
    target: "officeSplat",
    direction: "out",           // "out" = disappear, "in" = appear
    particles: true,            // Spawn particles during dissolve
    particleCount: 500,         // Number of particles
    particleColor: "#ffffff",   // Particle color
    particleSpeed: 2.0,         // How fast particles move
    easing: "easeInCubic",
    criteria: {
      currentState: OFFICE_EXIT,
      currentZone: "office"
    }
  },

  // Scene dissolve in (fade in)
  plazaAppear: {
    type: "dissolve",
    duration: 4000,
    target: "plazaSplat",
    direction: "in",
    particles: true,
    particleCount: 800,
    particleColor: "#88ccff",
    easing: "easeOutCubic",
    criteria: {
      currentState: PLAZA_ENTER,
      currentZone: "plaza"
    }
  },

  // Object dissolve (specific splat)
  runeDisappear: {
    type: "dissolve",
    duration: 2000,
    target: "runeObject",
    direction: "out",
    particles: true,
    particleCount: 200,
    particleColor: "#ff0000",
    criteria: {
      currentState: RUNE_VANISH,
      sawRune: true
    }
  }
};
```

### Dissolve Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | Must be "dissolve" |
| `target` | string | Yes | Splat object to dissolve |
| `direction` | string | Yes | "in" (appear) or "out" (disappear) |
| `duration` | number | Yes | Transition duration in ms |
| `particles` | boolean | No | Whether to spawn particles |
| `particleCount` | number | No | Number of particles to spawn |
| `particleColor` | string | No | Particle color (hex) |
| `particleSpeed` | number | No | Particle movement speed |
| `noiseScale` | number | No | Noise texture scale (variation) |
| `easing` | string | No | Easing function for transition |

---

## Part 3: How Dissolve Works for Gaussian Splats

### Approach 1: Opacity-Based Dissolve

The simplest method - gradually reduce splat opacity:

```javascript
class OpacityDissolve {
  constructor(splatObject, config) {
    this.splat = splatObject;
    this.duration = config.duration || 3000;
    this.direction = config.direction || "out";
    this.startTime = performance.now();

    // Get the splat material
    this.material = this.splat.material;
    this.originalOpacity = this.material.opacity;
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);

    // Apply easing
    const eased = this.easeInOutCubic(progress);

    // Calculate target opacity
    let targetOpacity;
    if (this.direction === "out") {
      // Fade out: 1 → 0
      targetOpacity = this.originalOpacity * (1 - eased);
    } else {
      // Fade in: 0 → 1
      targetOpacity = this.originalOpacity * eased;
    }

    // Apply to material
    this.material.opacity = targetOpacity;
    this.material.transparent = true;
    this.material.needsUpdate = true;

    return progress >= 1;  // Return true when complete
  }

  easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }
}
```

### Approach 2: Point Size Dissolve

Reduce splat size to zero - makes them "shrink away":

```javascript
class SizeDissolve {
  constructor(splatObject, config) {
    this.splat = splatObject;
    this.duration = config.duration || 3000;
    this.direction = config.direction || "out";
    this.startTime = performance.now();

    // Get original point sizes
    this.geometry = this.splat.geometry;
    this.originalSizes = this.geometry.attributes.size.clone();
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);
    const eased = this.easeInOutCubic(progress);

    // Calculate size multiplier
    let sizeMultiplier;
    if (this.direction === "out") {
      sizeMultiplier = 1 - eased;  // Shrink to 0
    } else {
      sizeMultiplier = eased;      // Grow from 0
    }

    // Apply to all points
    const sizes = this.geometry.attributes.size;
    for (let i = 0; i < sizes.count; i++) {
      const originalSize = this.originalSizes.array[i];
      sizes.array[i] = originalSize * sizeMultiplier;
    }

    sizes.needsUpdate = true;

    return progress >= 1;
  }
}
```

### Approach 3: Noise-Based Dissolve (Patterned)

Use a noise pattern for natural-looking dissolve:

```javascript
class NoiseDissolve {
  constructor(splatObject, config) {
    this.splat = splatObject;
    this.duration = config.duration || 3000;
    this.direction = config.direction || "out";
    this.noiseScale = config.noiseScale || 1.0;
    this.startTime = performance.now();

    // Pre-generate noise values for each point
    this.noiseValues = this.generateNoise(this.splat.geometry);
    this.originalOpacity = this.splat.material.opacity;
  }

  generateNoise(geometry) {
    const count = geometry.attributes.position.count;
    const noise = new Float32Array(count);

    // Generate Perlin-like noise for each point
    for (let i = 0; i < count; i++) {
      const position = geometry.attributes.position;
      const x = position.getX(i) * this.noiseScale;
      const y = position.getY(i) * this.noiseScale;
      const z = position.getZ(i) * this.noiseScale;

      // Simple noise function
      noise[i] = this.perlinNoise(x, y, z);
    }

    return noise;
  }

  perlinNoise(x, y, z) {
    // Simplified 3D noise
    const dot = x * 12.9898 + y * 78.233 + z * 37.719;
    return (Math.sin(dot) * 43758.5453) % 1;
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);
    const eased = this.easeInOutCubic(progress);

    // Calculate noise threshold
    // Points with noise < threshold become invisible
    const threshold = this.direction === "out" ? eased : (1 - eased);

    // Apply to each point
    const sizes = this.splat.geometry.attributes.size;
    const originalSizes = this.originalSizes || sizes.clone();

    for (let i = 0; i < sizes.count; i++) {
      const noiseValue = this.noiseValues[i];

      if (noiseValue < threshold) {
        // This point is "dissolved"
        sizes.array[i] = 0;
      } else {
        // This point is still visible
        const fadeAmount = (noiseValue - threshold) / (1 - threshold);
        sizes.array[i] = originalSizes.array[i] * fadeAmount;
      }
    }

    sizes.needsUpdate = true;

    return progress >= 1;
  }
}
```

### Approach 4: Particle Dissolve (Full Effect)

Combine opacity fade with particle emission:

```javascript
class ParticleDissolve {
  constructor(splatObject, config, scene) {
    this.splat = splatObject;
    this.config = config;
    this.scene = scene;
    this.duration = config.duration || 3000;
    this.direction = config.direction || "out";
    this.startTime = performance.now();

    // Create particle system
    this.particles = this.createParticles();
    this.scene.add(this.particles);

    // Initialize splat opacity
    this.originalOpacity = this.splat.material.opacity;
  }

  createParticles() {
    const count = this.config.particleCount || 500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const velocities = [];
    const lifetimes = new Float32Array(count);

    // Get splat bounds for particle spawn area
    const splatBox = new THREE.Box3().setFromObject(this.splat);

    for (let i = 0; i < count; i++) {
      // Random position within splat bounds
      positions[i * 3] = this.randomRange(splatBox.min.x, splatBox.max.x);
      positions[i * 3 + 1] = this.randomRange(splatBox.min.y, splatBox.max.y);
      positions[i * 3 + 2] = this.randomRange(splatBox.min.z, splatBox.max.z);

      // Random velocity
      velocities.push({
        x: (Math.random() - 0.5) * this.config.particleSpeed,
        y: Math.random() * this.config.particleSpeed * 0.5,
        z: (Math.random() - 0.5) * this.config.particleSpeed
      });

      // Random lifetime
      lifetimes[i] = Math.random();
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: this.config.particleColor || "#ffffff",
      size: 0.1,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const points = new THREE.Points(geometry, material);
    points.userData.velocities = velocities;
    points.userData.lifetimes = lifetimes;

    return points;
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);
    const eased = this.easeInOutCubic(progress);

    // Update splat opacity
    let splatOpacity;
    if (this.direction === "out") {
      splatOpacity = this.originalOpacity * (1 - eased);
    } else {
      splatOpacity = this.originalOpacity * eased;
    }
    this.splat.material.opacity = splatOpacity;

    // Update particles
    this.updateParticles(eased, deltaTime);

    return progress >= 1;
  }

  updateParticles(progress, deltaTime) {
    const positions = this.particles.geometry.attributes.position;
    const velocities = this.particles.userData.velocities;
    const lifetimes = this.particles.userData.lifetimes;
    const material = this.particles.material;

    // Fade particles in based on progress
    material.opacity = Math.sin(progress * Math.PI);  // Fade in then out

    for (let i = 0; i < velocities.length; i++) {
      // Update position
      positions.array[i * 3] += velocities[i].x * deltaTime;
      positions.array[i * 3 + 1] += velocities[i].y * deltaTime;
      positions.array[i * 3 + 2] += velocities[i].z * deltaTime;

      // Update lifetime
      lifetimes[i] += deltaTime * 0.5;

      // Respawn dead particles
      if (lifetimes[i] > 1 && progress < 0.8) {
        lifetimes[i] = 0;
        // Reset position to splat bounds
        const splatBox = new THREE.Box3().setFromObject(this.splat);
        positions.array[i * 3] = this.randomRange(splatBox.min.x, splatBox.max.x);
        positions.array[i * 3 + 1] = this.randomRange(splatBox.min.y, splatBox.max.y);
        positions.array[i * 3 + 2] = this.randomRange(splatBox.min.z, splatBox.max.z);
      }
    }

    positions.needsUpdate = true;
  }

  cleanup() {
    this.scene.remove(this.particles);
    this.particles.geometry.dispose();
    this.particles.material.dispose();
  }

  randomRange(min, max) {
    return min + Math.random() * (max - min);
  }

  easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }
}
```

---

## Part 4: Dissolve Coordinator

### Managing Multiple Dissolves

```javascript
class DissolveCoordinator {
  constructor(vfxManager, sceneManager) {
    this.vfxManager = vfxManager;
    this.sceneManager = sceneManager;
    this.activeDissolves = new Map();
  }

  startDissolve(config) {
    const target = this.getTarget(config.target);
    if (!target) {
      console.warn(`Dissolve target not found: ${config.target}`);
      return;
    }

    // Create appropriate dissolve type
    const dissolve = this.createDissolve(target, config);
    if (dissolve) {
      this.activeDissolves.set(config.target, dissolve);
    }
  }

  createDissolve(target, config) {
    // Choose dissolve method based on config
    if (config.particles) {
      return new ParticleDissolve(target, config, this.vfxManager.scene);
    } else if (config.noiseScale) {
      return new NoiseDissolve(target, config);
    } else if (config.useSize) {
      return new SizeDissolve(target, config);
    } else {
      return new OpacityDissolve(target, config);
    }
  }

  update(deltaTime) {
    // Update all active dissolves
    for (const [name, dissolve] of this.activeDissolves) {
      const complete = dissolve.update(deltaTime);

      if (complete) {
        this.onDissolveComplete(name, dissolve);
      }
    }
  }

  onDissolveComplete(name, dissolve) {
    // Clean up
    if (dissolve.cleanup) {
      dissolve.cleanup();
    }

    this.activeDissolves.delete(name);

    // Emit event
    this.vfxManager.emit('dissolve:complete', { target: name });
  }

  getTarget(targetName) {
    // Find splat object by name
    return this.sceneManager.scene.getObjectByName(targetName);
  }
}
```

---

## Part 5: Practical Dissolve Examples

### Example 1: Scene Transition (Chain Dissolves)

```javascript
export const dissolveTransitions = {
  // Zone transition: office → plaza
  zoneTransition: {
    dissolveOut: {
      target: "officeSplat",
      direction: "out",
      duration: 3000,
      particles: true,
      particleCount: 1000,
      particleColor: "#aaaaaa",
      easing: "easeInCubic"
    },

    dissolveIn: {
      target: "plazaSplat",
      direction: "in",
      duration: 4000,
      particles: true,
      particleCount: 1500,
      particleColor: "#aaddff",
      easing: "easeOutCubic"
    }
  }
};

// Usage
function transitionToPlaza() {
  // Start dissolve out
  dissolveCoordinator.startDissolve(zoneTransition.dissolveOut);

  // Wait for completion, then load and dissolve in
  setTimeout(() => {
    sceneManager.loadZone("plaza");
    dissolveCoordinator.startDissolve(zoneTransition.dissolveIn);
  }, 3000);
}
```

### Example 2: Object Disappear (Rune Vanishing)

```javascript
export const runeDissolve = {
  // Rune fades away after viewing
  runeVanish: {
    type: "dissolve",
    target: "runeObject",
    direction: "out",
    duration: 5000,
    particles: true,
    particleCount: 300,
    particleColor: "#ff3333",
    particleSpeed: 1.5,
    noiseScale: 2.0,
    easing: "easeInCubic",
    criteria: {
      currentState: RUNE_VANISH,
      runeViewed: true
    }
  }
};
```

### Example 3: Dramatic Reveal (Boss Appearance)

```javascript
export const bossReveal = {
  // Boss materializes from particles
  bossAppear: {
    type: "dissolve",
    target: "bossSplat",
    direction: "in",
    duration: 6000,
    particles: true,
    particleCount: 2000,
    particleColor: "#ff0000",
    particleSpeed: 3.0,
    noiseScale: 0.5,
    easing: "easeOutBack",
    criteria: {
      currentState: BOSS_INTRO,
      bossActivated: true
    }
  }
};
```

---

## Part 6: Performance Optimization

### Particle Count Limits

```javascript
class DissolveOptimizer {
  constructor() {
    this.maxParticles = {
      desktop: 2000,
      laptop: 1000,
      mobile: 500
    };

    this.currentProfile = 'desktop';
  }

  adjustForPerformance(profile) {
    this.currentProfile = profile;

    // Reduce particle count for lower-end devices
    if (profile === 'mobile') {
      this.maxParticles.mobile = 500;
    }
  }

  getOptimalParticleCount(baseCount) {
    const max = this.maxParticles[this.currentProfile];
    return Math.min(baseCount, max);
  }
}
```

### Geometry Updates Optimization

```javascript
// Instead of updating every point every frame
class OptimizedDissolve {
  constructor(splatObject, config) {
    // Group points by noise value
    this.updateGroups = this.createUpdateGroups(splatObject, config);
  }

  createUpdateGroups(splatObject, config) {
    // Divide points into groups that dissolve at different times
    const pointCount = splatObject.geometry.attributes.position.count;
    const groupCount = 10;  // 10 update groups
    const groupSize = Math.floor(pointCount / groupCount);

    const groups = [];
    for (let i = 0; i < groupCount; i++) {
      groups.push({
        start: i * groupSize,
        end: (i + 1) * groupSize,
        threshold: i / groupCount
      });
    }

    return groups;
  }

  update(progress) {
    // Only update points in current active group
    const groupIndex = Math.floor(progress * this.updateGroups.length);
    const group = this.updateGroups[groupIndex];

    // Update only points in this group
    this.updateGroupPoints(group, progress);

    // Other groups remain at their previous state
    // Much faster than updating all points every frame!
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Dissolving Too Fast

```javascript
// ❌ WRONG: Too abrupt
{
  duration: 500,  // Half second - barely visible!
  particles: false
}

// ✅ CORRECT: Give time for the effect
{
  duration: 3000,  // 3 seconds - players can appreciate it
  particles: true
}
```

### 2. No Particles (Boring Dissolve)

```javascript
// ❌ WRONG: Simple fade
{
  direction: "out",
  duration: 3000
  // Just opacity change - boring!
}

// ✅ CORRECT: Add particles for magic feel
{
  direction: "out",
  duration: 3000,
  particles: true,
  particleCount: 800,
  particleColor: "#ffffff"
}
```

### 3. Wrong Direction

```javascript
// ❌ WRONG: Scene appears but we wanted it to disappear
{
  direction: "in",  // Wrong!
  target: "oldScene"
}

// ✅ CORRECT: Match direction to intent
{
  direction: "out",  // Fade away old scene
  target: "oldScene"
}
```

### 4. Forgetting to Clean Up

```javascript
// ❌ WRONG: Particles stay forever
function startDissolve() {
  const particles = createParticles();
  scene.add(particles);
  // Never removed - memory leak!
}

// ✅ CORRECT: Clean up when done
function startDissolve() {
  const dissolve = new ParticleDissolve(config);

  const cleanup = () => {
    dissolve.cleanup();
    scene.remove(particles);
  };

  dissolve.on('complete', cleanup);
}
```

---

## Performance Considerations

### Dissolve Cost by Method

```
DISSOLVE METHOD PERFORMANCE (per 100k points):

Opacity Fade:
├── Cost: Low
├── CPU: Uniform update (single draw call)
└── GPU: Simple alpha change

Size Shrink:
├── Cost: Low-Medium
├── CPU: Per-point size update
└── GPU: Geometry update required

Noise Pattern:
├── Cost: Medium
├── CPU: Per-point comparison with noise value
└── GPU: Size update with conditionals

Particle Dissolve:
├── Cost: High
├── CPU: Particle system + splat updates
├── GPU: Two draw calls + particle updates
└── Memory: Additional particle buffers

RECOMMENDATION: Use opacity fade for simple transitions,
                particle dissolve for dramatic moments.
```

---

## 🎮 Game Design Perspective

### Dissolve as Storytelling

```javascript
// Principle: Dissolve style tells a story

Magical disappearance:
{
  particles: true,
  particleColor: "#ffffff",
  particleSpeed: 2.0,
  easing: "easeOutCubic"
  // Graceful, magical fade away
}

Dark consumption:
{
  particles: true,
  particleColor: "#330000",
  particleSpeed: 0.5,
  easing: "easeInQuad"
  // Slow, ominous draining away
}

Digital derezzing:
{
  particles: false,
  noiseScale: 10.0,
  easing: "linear"
  // Digital breakdown effect
}

Natural decay:
{
  particles: true,
  particleColor: "#553322",
  particleSpeed: 0.2,
  easing: "easeInOutSine"
  // Like dust blowing away
}
```

### Emotional Impact

```javascript
// Dissolve creates emotional response through timing and style

Sudden disappearance (horror):
{
  duration: 500,
  direction: "out",
  particles: true,
  particleCount: 50,
  particleSpeed: 8.0
  // Quick and shocking
}

Peaceful fade (whimsical):
{
  duration: 6000,
  direction: "out",
  particles: true,
  particleCount: 2000,
  particleSpeed: 1.0,
  easing: "easeInOutSine"
  // Gentle and dreamlike
}

Building tension:
{
  duration: 1000,
  direction: "in",
  particles: true,
  particleCount: 500,
  particleSpeed: 4.0,
  easing: "easeOutBack"
  // Explosive arrival
}
```

---

## Next Steps

Now that you understand the dissolve effect:

- [VFXManager](./vfx-manager.md) - Base visual effects system
- [Gaussian Splatting Explained](../01-foundation/gaussian-splatting-explained.md) - Understanding splats
- [Glitch Post-Processing](./glitch-post-processing.md) - Digital distortion effects
- [SceneManager](../03-scene-rendering/scene-manager.md) - Scene loading and management

---

## References

- [Three.js PointsMaterial](https://threejs.org/docs/#api/en/materials/PointsMaterial) - Point cloud rendering
- [Three.js BufferAttribute](https://threejs.org/docs/#api/en/core/BufferAttribute) - Per-vertex data
- [Particle Systems](https://threejs.org/docs/#api/en/objects/Points) - Point particle rendering
- [Custom Shaders](https://threejs.org/docs/#api/en/materials/ShaderMaterial) - Custom GLSL shaders

*Documentation last updated: January 12, 2026*
