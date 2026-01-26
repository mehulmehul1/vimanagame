# ParticleCanvas3D - First Principles Guide

## Overview

**ParticleCanvas3D** is the visual feedback system for the drawing minigame. As players draw symbols in 3D space, particles trail behind their cursor/stroke, creating magical visual feedback that makes the drawing feel alive and responsive. These particles aren't just cosmetic - they confirm input, guide the player, and enhance the magical atmosphere.

Think of ParticleCanvas3D as the **"sparkle trail"** system - just as a magic wand leaves sparkles in its wake, the player's drawing gesture leaves a trail of glowing particles, providing immediate visual confirmation that their input is being registered.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create wonder through visual feedback. Every gesture should feel magical - particles make simple input feel like casting a spell.

**Why Particle Feedback?**
- **Immediate Confirmation**: Players see their input is working
- **Magical Atmosphere**: Glowing particles = magic happening
- **Spatial Awareness**: 3D particles help players "feel" the drawing plane
- **Reward Feedback**: Burst effects on successful recognition feel great

**Player Psychology**:
```
Start Drawing → Particles Appear → "I'm doing it!"
     ↓
Continue Drawing → Trail Follows → Immersion
     ↓
Complete Drawing → Particles Burst → Satisfaction
```

### Visual Design Principles

**1. Particle Appearance**
```
Color: Cyan/green for active drawing
     Gold/white for recognition success
     Red/fade for failure

Size: Small (0.02-0.05 units) for subtle effect
     Larger burst (0.1 units) for success

Shape: Simple sprites or points
     Soft glow via additive blending
```

**2. Particle Behavior**
```
Spawn: At cursor position during draw
Life: 1-3 seconds
Fade: Linear opacity decay
Motion: Slight drift, gravity optional
```

**3. Burst Effect**
```
On Recognition Success:
- Spawn 50-100 particles at stroke center
- Expand outward with velocity
- Fade quickly (0.5-1 second)
- Use success color (gold/white)
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding ParticleCanvas3D, you should know:
- **THREE.Points** - Efficient particle rendering
- **BufferGeometry** - Storing particle positions
- **ShaderMaterial** - Custom particle appearance
- **AdditiveBlending** - Glowing overlap effect
- **Object pooling** - Reusing particles for performance

### Core Architecture

```
PARTICLE CANVAS 3D ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    PARTICLE CANVAS 3D                    │
│  - Manages particle pool                                │
│  - Spawns particles at positions                        │
│  - Updates particle physics each frame                  │
│  - Renders particles efficiently                        │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   SPAWNER     │  │  PHYSICS     │  │   RENDERER   │
│  - Create     │  │  - Update     │  │  - Draw      │
│    particles  │  │    positions  │  │    batches   │
└───────────────┘  └───────────────┘  └───────────────┘
```

### ParticleCanvas3D Class

```javascript
import * as THREE from 'three';

class ParticleCanvas3D {
  constructor(options = {}) {
    this.scene = options.scene;
    this.logger = options.logger || console;

    // Particle pool
    this.maxParticles = options.maxParticles || 500;
    this.activeParticles = [];
    this.inactiveParticles = [];

    // Configuration
    this.config = {
      particleSize: 0.03,
      particleLife: 2.0,      // Seconds
      particleColor: 0x00ff88,
      burstColor: 0xffd700,    // Gold for success
      gravity: new THREE.Vector3(0, -0.5, 0),
      driftAmount: 0.02,
      blending: THREE.AdditiveBlending
    };

    // Create particle system
    this.createParticleSystem();

    // Trail state
    this.isTrailing = false;
    this.trailInterval = 0.02;  // Spawn every 20ms
    this.lastTrailSpawn = 0;
  }

  /**
   * Create particle rendering system
   */
  createParticleSystem() {
    // Create geometry for all particles
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.maxParticles * 3);
    const colors = new Float32Array(this.maxParticles * 3);
    const sizes = new Float32Array(this.maxParticles);
    const alphas = new Float32Array(this.maxParticles);

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));

    // Create shader material for particles
    const material = new THREE.ShaderMaterial({
      uniforms: {
        pointTexture: { value: this.createParticleTexture() }
      },
      vertexShader: `
        attribute float size;
        attribute vec3 color;
        attribute float alpha;
        varying vec3 vColor;
        varying float vAlpha;

        void main() {
          vColor = color;
          vAlpha = alpha;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform sampler2D pointTexture;
        varying vec3 vColor;
        varying float vAlpha;

        void main() {
          vec4 texColor = texture2D(pointTexture, gl_PointCoord);
          gl_FragColor = vec4(vColor, vAlpha * texColor.a);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: this.config.blending
    });

    this.particleSystem = new THREE.Points(geometry, material);
    this.particleSystem.frustumCulled = false;
    this.scene.add(this.particleSystem);
  }

  /**
   * Create particle texture (soft circle)
   */
  createParticleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');

    // Draw soft gradient circle
    const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.3, 'rgba(255, 255, 255, 0.8)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 32, 32);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  /**
   * Spawn a single particle
   */
  spawnParticle(position, options = {}) {
    // Get particle from pool or create new
    let particle;

    if (this.inactiveParticles.length > 0) {
      particle = this.inactiveParticles.pop();
    } else if (this.activeParticles.length < this.maxParticles) {
      particle = this.createParticle();
    } else {
      // Pool exhausted, recycle oldest
      particle = this.activeParticles.shift();
    }

    // Initialize particle
    const color = options.color || this.config.particleColor;
    const colorObj = new THREE.Color(color);

    particle.position.copy(position);
    particle.velocity = options.velocity || new THREE.Vector3(
      (Math.random() - 0.5) * this.config.driftAmount,
      (Math.random() - 0.5) * this.config.driftAmount,
      (Math.random() - 0.5) * this.config.driftAmount
    );
    particle.life = options.life || this.config.particleLife;
    particle.maxLife = particle.life;
    particle.size = options.size || this.config.particleSize;
    particle.color.set(colorObj.x, colorObj.y, colorObj.z);
    particle.alpha = 1;

    this.activeParticles.push(particle);
    return particle;
  }

  /**
   * Create a particle object
   */
  createParticle() {
    return {
      position: new THREE.Vector3(),
      velocity: new THREE.Vector3(),
      life: 0,
      maxLife: 0,
      size: 0.03,
      color: new THREE.Color(),
      alpha: 1
    };
  }

  /**
   * Spawn burst of particles
   */
  burst(position, options = {}) {
    const count = options.count || 50;
    const color = options.color || this.config.burstColor;
    const speed = options.speed || 1.0;

    for (let i = 0; i < count; i++) {
      // Random direction on sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      const velocity = new THREE.Vector3(
        Math.sin(phi) * Math.cos(theta),
        Math.sin(phi) * Math.sin(theta),
        Math.cos(phi)
      ).multiplyScalar(speed * (0.5 + Math.random() * 0.5));

      this.spawnParticle(position, {
        velocity,
        color,
        life: 0.5 + Math.random() * 0.5,
        size: this.config.particleSize * (0.5 + Math.random())
      });
    }
  }

  /**
   * Start trailing (for drawing)
   */
  startTrail() {
    this.isTrailing = true;
    this.lastTrailSpawn = performance.now();
  }

  /**
   * Stop trailing
   */
  stopTrail() {
    this.isTrailing = false;
  }

  /**
   * Add trail particle at position
   */
  addParticle(position) {
    if (!this.isTrailing) return;

    const now = performance.now();
    if (now - this.lastTrailSpawn >= this.trailInterval * 1000) {
      this.spawnParticle(position, {
        velocity: new THREE.Vector3(0, 0.2, 0),  // Slight upward drift
        life: this.config.particleLife
      });
      this.lastTrailSpawn = now;
    }
  }

  /**
   * Update particles
   */
  update(dt) {
    const positions = this.particleSystem.geometry.attributes.position.array;
    const colors = this.particleSystem.geometry.attributes.color.array;
    const sizes = this.particleSystem.geometry.attributes.size.array;
    const alphas = this.particleSystem.geometry.attributes.alpha.array;

    // Update active particles
    for (let i = this.activeParticles.length - 1; i >= 0; i--) {
      const particle = this.activeParticles[i];

      // Update life
      particle.life -= dt;

      if (particle.life <= 0) {
        // Particle died, return to pool
        this.inactiveParticles.push(particle);
        this.activeParticles.splice(i, 1);
        continue;
      }

      // Update position
      particle.position.add(particle.velocity.clone().multiplyScalar(dt));
      particle.velocity.add(this.config.gravity.clone().multiplyScalar(dt));

      // Update alpha based on life
      particle.alpha = particle.life / particle.maxLife;
    }

    // Update geometry
    let particleIndex = 0;

    // Active particles
    for (const particle of this.activeParticles) {
      positions[particleIndex * 3] = particle.position.x;
      positions[particleIndex * 3 + 1] = particle.position.y;
      positions[particleIndex * 3 + 2] = particle.position.z;

      colors[particleIndex * 3] = particle.color.r;
      colors[particleIndex * 3 + 1] = particle.color.g;
      colors[particleIndex * 3 + 2] = particle.color.b;

      sizes[particleIndex] = particle.size;
      alphas[particleIndex] = particle.alpha;

      particleIndex++;
    }

    // Remaining slots (hide inactive particles)
    for (let i = particleIndex; i < this.maxParticles; i++) {
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;
      alphas[i] = 0;
    }

    // Mark attributes as needing update
    this.particleSystem.geometry.attributes.position.needsUpdate = true;
    this.particleSystem.geometry.attributes.color.needsUpdate = true;
    this.particleSystem.geometry.attributes.size.needsUpdate = true;
    this.particleSystem.geometry.attributes.alpha.needsUpdate = true;
  }

  /**
   * Clear all particles
   */
  clear() {
    this.activeParticles.forEach(p => this.inactiveParticles.push(p));
    this.activeParticles = [];
  }

  /**
   * Get particle count
   */
  getParticleCount() {
    return this.activeParticles.length;
  }

  /**
   * Clean up
   */
  destroy() {
    this.clear();
    this.scene.remove(this.particleSystem);
    this.particleSystem.geometry.dispose();
    this.particleSystem.material.dispose();
  }
}

export default ParticleCanvas3D;
```

---

## 📝 How To Build A Particle System Like This

### Step 1: Define Your Particle Properties

```javascript
const particleConfig = {
  size: 0.03,           // How big
  life: 2.0,            // How long (seconds)
  color: 0x00ff88,       // What color
  gravity: -0.5,         // Fall speed
  blending: "additive"   // How they overlap
};
```

### Step 2: Create the Geometry

```javascript
// Use THREE.Points for efficiency
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(maxParticles * 3);
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
```

### Step 3: Update Each Frame

```javascript
function update(dt) {
  for (particle of particles) {
    particle.life -= dt;
    particle.position += particle.velocity * dt;
    particle.alpha = particle.life / particle.maxLife;
  }

  // Remove dead particles
  particles = particles.filter(p => p.life > 0);
}
```

---

## 🔧 Variations For Your Game

### Fire Particles

```javascript
{
  color: 0xff4400,
  gravity: new THREE.Vector3(0, 0.5, 0),  // Fall down
  drift: 0.01,
  life: 1.0,
  size: 0.02
}
```

### Magic Sparkles

```javascript
{
  color: 0x00ffff,
  gravity: new THREE.Vector3(0, 0, 0),   // No gravity
  drift: 0.05,                             // Float away
  life: 3.0,
  size: 0.01,
  pulse: true
}
```

### Snowflakes

```javascript
{
  color: 0xffffff,
  gravity: new THREE.Vector3(0, -0.1, 0),
  drift: 0.02,
  life: 5.0,
  rotation: true,
  size: 0.02
}
```

---

## Common Mistakes Beginners Make

### 1. Too Many Particles

```javascript
// ❌ WRONG: Performance killer
{ maxParticles: 10000 }
// Runs at 5 FPS

// ✅ CORRECT: Balanced
{ maxParticles: 500 }
// Runs smoothly
```

### 2. No Fade Out

```javascript
// ❌ WRONG: Pop out of existence
particle.alpha = 1;
if (life <= 0) remove();

// ✅ CORRECT: Smooth fade
particle.alpha = life / maxLife;
// Natural disappearance
```

### 3. Wrong Blend Mode

```javascript
// ❌ WRONG: Opaque stacking
{ blending: THREE.NormalBlending }
// Particles look like solid circles

// ✅ CORRECT: Glowing overlap
{ blending: THREE.AdditiveBlending }
// Particles glow and blend
```

---

## Performance Considerations

```
PARTICLE SYSTEM PERFORMANCE:

Per-Particle Cost:
├── Position update: Minimal
├── Physics calculation: Minimal
├── Alpha fade: Minimal
└── Total per particle: ~0.01ms

Total Cost:
├── 100 particles: ~1ms/frame (negligible)
├── 500 particles: ~5ms/frame (minor)
├── 1000 particles: ~10ms/frame (noticeable)
└── 5000 particles: ~50ms/frame (major)

Optimization:
- Use object pooling (recycle particles)
- Limit max particles
- Use simple geometry (Points)
- Avoid complex shader calculations
- Update only visible particles
```

---

## Related Systems

- [DrawingManager](./drawing-manager.md) - Input coordination
- [DrawingRecognitionManager](./drawing-recognition-manager.md) - ML recognition
- [VFXManager](../07-visual-effects/vfx-manager.md) - Effect orchestration
- [Three.js Points](https://threejs.org/docs/#api/en/objects/Points) - Particle rendering

---

## Source File Reference

**Primary Files**:
- `../src/content/ParticleCanvas3D.js` - Particle system (estimated)

**Key Classes**:
- `ParticleCanvas3D` - Main particle system
- Particle pool management

**Dependencies**:
- Three.js (Points, BufferGeometry, ShaderMaterial)
- Object pooling pattern

---

## References

- [Three.js Points](https://threejs.org/docs/#api/en/objects/Points) - Particle rendering
- [BufferGeometry](https://threejs.org/docs/#api/en/core/BufferGeometry) - Efficient geometry
- [ShaderMaterial](https://threejs.org/docs/#api/en/materials/ShaderMaterial) - Custom shaders
- [Additive Blending](https://www.khronos.org/opengl/wiki/Blending) - Glow effects
- [Particle Systems](https://www.youtube.com/watch?v=v7Yfgne4z7U) - Tutorial video

*Documentation last updated: January 12, 2026*
