# Selective Bloom Composer - First Principles Guide

## Overview

**Bloom** is a post-processing effect that makes bright areas of your scene glow, creating a dreamy, magical atmosphere. The **Selective Bloom Composer** takes this further by allowing you to control which objects glow and which don't - letting you highlight important items like runes, candles, or magical artifacts while keeping the rest of the scene sharp.

Think of selective bloom as the **"magical highlighter"** for your game world - just as a spotlight draws attention to a stage performer, bloom makes important objects visually pop and glow, telling players "this matters."

## What You Need to Know First

Before understanding bloom and selective bloom, you should know:
- **Post-processing pipeline** - Effects applied after the main render
- **Brightness thresholds** - Only pixels above certain brightness glow
- **Render layers** - Separating objects into different rendering passes
- **Framebuffers** - Off-screen textures for intermediate rendering
- **Gaussian blur** - The blurring technique that creates the glow

### Quick Refresher: What is Bloom?

```
NORMAL RENDERING:
┌────────────────────────────────┐
│ Scene rendered as-is            │
│ ████████████████████████████   │
│ ████████████████████████████   │
│ ████████████████████████████   │
│ Bright areas just look bright    │
└────────────────────────────────┘

WITH BLOOM:
┌────────────────────────────────┐
│ Bright areas GLOW and bleed     │
│ ████████████████████████████   │
│ ████████████████████████████   │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │ ← Glow spreads
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │
└────────────────────────────────┘
Bloom makes bright things GLOW

SELECTIVE BLOOM:
┌────────────────────────────────┐
│ Only SPECIFIC objects glow    │
│ ████████████░░░░░░░░░░░░░░░   │ ← This object glows
│ ██████████████░░░░░░░░░░░░░   │
│ ████████████████░░░░░░░░░░░░   │
│ ░░░░░░░░░░███████████████░░░░   │ ← That object glows
└────────────────────────────────┘
You control EXACTLY what glows!
```

**Why selective bloom matters:** Regular bloom applies to everything - bright skies, lights, shiny surfaces. Selective bloom lets you choose exactly what glows, creating magical highlights on important objects without the whole scene looking hazy.

---

## Part 1: Why Use Bloom?

### The Problem: Flat, Lifeless Scenes

Without bloom:

```javascript
// ❌ WITHOUT Bloom - Scene looks flat
const candleScene = {
  candle: { emissive: white, intensity: 1.0 },
  rune: { emissive: red, intensity: 1.0 }
};
// Bright areas just look... bright.
// No magical feeling.
```

**Problems:**
- Lights don't feel "light-filled"
- Magical objects look ordinary
- Missing atmosphere and mood
- Can't guide player attention visually

### The Solution: Bloom

```javascript
// ✅ WITH Bloom - Scene comes alive
const candleScene = {
  candle: {
    emissive: white,
    intensity: 1.0,
    bloom: { strength: 1.5, radius: 0.5 }
  },
  rune: {
    emissive: red,
    intensity: 1.0,
    bloom: { strength: 3.0, radius: 0.8 }  // Stronger glow!
  }
};
// Candle flickers with warmth.
// Rune pulses with magical energy.
// Scene feels ALIVE.
```

**Benefits:**
- Creates atmosphere and mood
- Highlights important objects
- Makes lights feel more realistic
- Adds magical, supernatural quality
- Guides player attention

---

## Part 2: How Bloom Works

### The Bloom Algorithm

```
BLOOM PIPELINE:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Scene     │ →  │  Extract    │ →  │    Blur     │ →  │    Combine   │
│   Render    │    │  Bright     │    │    Glow     │    │   Original   │
└─────────────┘    │  Areas      │    └─────────────┘    └─────────────┘
                   └─────────────┘              │
                       (Threshold)             (+ Add glow)
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   Final     │
                                          │   Image     │
                                          │ (with glow)  │
                                          └─────────────┘

STEP 1: RENDER SCENE
└── Render entire 3D scene to texture

STEP 2: EXTRACT BRIGHT AREAS
└── Copy scene texture
└── For each pixel: if brightness > threshold → keep, else → black

STEP 3: BLUR THE BRIGHT AREAS
└── Apply Gaussian blur horizontally
└── Apply Gaussian blur vertically
└── This creates the "glow" spread effect

STEP 4: COMBINE WITH ORIGINAL
└── Add blurred glow texture on top of original
└── Result: bright areas now glow!
```

### Gaussian Blur Explained

```javascript
// Gaussian blur uses a bell curve to weight pixels
// Center pixel contributes most, neighbors contribute less

    1D Gaussian Kernel (for horizontal/vertical blur)
    ───┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
    │  │  │  │  │  │  │  │  │  │  │  │  │  Weights
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
     Low weights      Peak (center)

// Typical kernel (5x5 shown, but larger used)
const kernel = [
  1,  4,  7,  4,  1,
  4, 16, 26, 16, 4,
  7, 26, 41, 26, 7,  ← Center = highest weight
  4, 16, 26, 16, 4,
  1,  4,  7,  4,  1
];
```

---

## Part 3: Selective Bloom Implementation

### Using Three.js Layers

```javascript
class SelectiveBloomComposer {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;

    // Create render targets
    this.renderTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    // Create layers for selective bloom
    this.bloomLayer = new THREE.Layers();
    this.bloomLayer.set(1);  // Layer 1 for bloom objects

    // Create bloom composer
    this.composer = this.createComposer();
  }

  createComposer() {
    const composer = new EffectComposer(this.renderer);

    // 1. Render pass (main scene)
    const renderPass = new RenderPass(this.scene, this.camera);
    composer.addPass(renderPass);

    // 2. Extract bright areas (only from bloom layer)
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      1.5,   // strength
      0.4,   // radius
      0.85   // threshold
    );

    // Configure bloom to only affect specific layer
    this.configureSelectiveBloom(bloomPass);

    composer.addPass(bloomPass);

    return composer;
  }

  configureSelectiveBloom(bloomPass) {
    // Set up layer-based rendering
    // Objects on layer 0 (default) → no bloom
    // Objects on layer 1 (bloomLayer) → bloom

    // This is handled by setting object layers:
    // object.layers.enable(1);  // Make object bloom

    // And rendering the scene twice:
    // Once for layer 0 (no bloom), once for layer 1 (bloom)
  }

  render() {
    // Render scene normally (layer 0)
    this.renderSceneLayer(0);

    // Render bloom layer (layer 1)
    // This goes through bloom pass
    this.renderSceneLayer(1);
  }

  renderSceneLayer(layer) {
    // Set camera to only see this layer
    this.camera.layers.set(layer);

    // Render
    this.renderer.render(this.scene, this.camera);

    // Reset camera to see all layers
    this.camera.layers.enableAll();
  }
}
```

### Marking Objects for Bloom

```javascript
// When creating objects, assign them to bloom layer
function createBloomObject(mesh, config) {
  // Add to scene
  scene.add(mesh);

  // Enable bloom layer
  mesh.layers.enable(1);

  // Also keep in main layer (0) for visibility
  mesh.layers.enable(0);

  // Adjust material for bloom
  if (config.bloomStrength) {
    mesh.material.emissive = new THREE.Color(config.bloomColor || "#ffffff");
    mesh.material.emissiveIntensity = config.bloomStrength;
  }
}

// Example: Creating a glowing rune
const runeGeometry = new THREE.PlaneGeometry(1, 1);
const runeMaterial = new THREE.MeshBasicMaterial({
  color: 0xff0000,
  emissive: 0xff0000,
  emissiveIntensity: 2.0  // High emission = more bloom
});

const rune = new THREE.Mesh(runeGeometry, runeMaterial);
rune.position.set(0, 1, -5);
rune.layers.set(1);  // Set to bloom layer only (optional)

scene.add(rune);
```

### Alternative: Multi-Pass Rendering

```javascript
class MultiPassBloomRenderer {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;

    // Create render targets for each pass
    this.mainTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight,
      { format: THREE.RGBAFormat }
    );

    this.bloomTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight,
      { format: THREE.RGBAFormat }
    );

    // Create bloom effect
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      1.5,
      0.4,
      0.85
    );

    // Create composer for bloom
    this.bloomComposer = new EffectComposer(this.renderer);
    this.bloomComposer.addPass(new RenderPass(scene, camera));
    this.bloomComposer.addPass(this.bloomPass);
  }

  render(bloomObjects) {
    // 1. Render main scene without bloom objects
    bloomObjects.forEach(obj => obj.visible = false);
    this.renderer.setRenderTarget(this.mainTarget);
    this.renderer.render(this.scene, this.camera);
    bloomObjects.forEach(obj => obj.visible = true);

    // 2. Render bloom pass with only bloom objects
    this.renderer.setRenderTarget(this.bloomTarget);
    this.bloomComposer.render();

    // 3. Combine both results
    this.combinedRender();
  }

  combinedRender() {
    // Use custom shader to combine both textures
    const combineMaterial = new THREE.ShaderMaterial({
      uniforms: {
        mainTexture: { value: this.mainTarget.texture },
        bloomTexture: { value: this.bloomTarget.texture }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D mainTexture;
        uniform sampler2D bloomTexture;
        varying vec2 vUv;

        void main() {
          vec4 mainColor = texture2D(mainTexture, vUv);
          vec4 bloomColor = texture2D(bloomTexture, vUv);

          // Additive blending for glow effect
          gl_FragColor = mainColor + bloomColor;
        }
      `
    });

    // Render fullscreen quad
    const quad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      combineMaterial
    );

    this.renderer.render(quad, this.camera);
  }
}
```

---

## Part 4: Bloom Data Structure

### Bloom Configuration in VFX Data

```javascript
// In vfxData.js
export const vfxEffects = {
  // Simple bloom on single object
  runeGlow: {
    type: "bloom",
    target: "runeObject",
    threshold: 0.7,      // Brightness threshold (0-1)
    strength: 1.5,       // Glow intensity
    radius: 0.5,         // Glow spread (blur amount)
    color: "#ff0000",     // Tint color (optional)
    criteria: {
      sawRune: true
    }
  },

  // Bloom on multiple objects
  ritualGlow: {
    type: "bloom",
    targets: ["candle1", "candle2", "candle3", "mysticOrb"],
    threshold: 0.6,
    strength: 2.0,
    radius: 0.3,
    criteria: {
      currentState: RITUAL_ACTIVE,
      ritualProgress: { $gte: 0.5 }
    }
  },

  // Pulsing bloom (animated intensity)
  pulsingRune: {
    type: "bloom",
    target: "runeObject",
    threshold: 0.5,
    strength: 2.0,
    radius: 0.6,
    pulse: {
      min: 1.0,
      max: 3.0,
      speed: 2.0  // Pulse speed
    },
    criteria: {
      runeActive: true
    }
  },

  // Area-wide bloom (all emissive objects)
  magicalArea: {
    type: "bloom",
    threshold: 0.4,
    strength: 1.0,
    radius: 0.4,
    criteria: {
      currentZone: "magical_realm"
    }
  }
};
```

### Bloom Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | Must be "bloom" |
| `target` | string | Conditional | Single object to bloom |
| `targets` | array | Conditional | Multiple objects to bloom |
| `threshold` | number | Yes | Brightness threshold (0-1) |
| `strength` | number | Yes | Glow intensity |
| `radius` | number | Yes | Glow spread amount |
| `color` | string | No | Tint the glow with this color |
| `pulse` | object | No | Pulse animation config |
| `criteria` | object | Yes | When bloom triggers |

---

## Part 5: Animated Bloom

### Pulsing Bloom Effect

```javascript
class PulsingBloomEffect {
  constructor(config, bloomPass) {
    this.config = config;
    this.bloomPass = bloomPass;
    this.startTime = performance.now();

    this.minStrength = config.pulse.min;
    this.maxStrength = config.pulse.max;
    this.pulseSpeed = config.pulse.speed;
  }

  update(deltaTime) {
    // Calculate pulse using sine wave
    const elapsed = (performance.now() - this.startTime) / 1000;
    const sineValue = Math.sin(elapsed * this.pulseSpeed);

    // Map sine (-1 to 1) to strength range
    const strength = this.minStrength +
      (this.maxStrength - this.minStrength) * (sineValue * 0.5 + 0.5);

    // Apply to bloom pass
    this.bloomPass.strength = strength;
  }

  isComplete() {
    return false;  // Pulsing is continuous
  }
}
```

### Ramping Bloom (Fade In/Out)

```javascript
class RampingBloomEffect {
  constructor(config, bloomPass) {
    this.config = config;
    this.bloomPass = bloomPass;
    this.targetStrength = config.strength;
    this.currentStrength = 0;
    this.duration = config.duration || 2000;
    this.startTime = performance.now();
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);
    const eased = this.easeInOutCubic(progress);

    this.currentStrength = THREE.MathUtils.lerp(
      0,
      this.targetStrength,
      eased
    );

    this.bloomPass.strength = this.currentStrength;
    this.bloomPass.radius = THREE.MathUtils.lerp(
      0,
      this.config.radius,
      eased
    );
  }

  easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  isComplete() {
    const elapsed = performance.now() - this.startTime;
    return elapsed >= this.duration;
  }
}
```

---

## Part 6: Practical Bloom Examples

### Example 1: Candle Flicker

```javascript
// Candles that flicker with bloom
export const vfxEffects = {
  candleFlicker: {
    type: "bloom",
    targets: ["candle1", "candle2", "candle3"],
    threshold: 0.5,
    strength: 1.2,
    radius: 0.3,
    flicker: {
      intensity: 0.3,    // How much brightness varies
      speed: 10,        // Flicker speed
      noiseSeed: 12345   // For consistent pattern
    },
    criteria: {
      currentZone: "ritual_room",
      candlesLit: true
    }
  }
};

// Flicker implementation
class FlickeringBloom {
  constructor(config, bloomPass) {
    this.config = config;
    this.bloomPass = bloomPass;
    this.baseStrength = config.strength;
    this.flickerIntensity = config.flicker.intensity;
    this.flickerSpeed = config.flicker.speed;
  }

  update(deltaTime) {
    const time = performance.now() / 1000;

    // Create flicker using noise
    const noise = this.noise(time * this.flickerSpeed);
    const flicker = 1 + (noise - 0.5) * this.flickerIntensity;

    this.bloomPass.strength = this.baseStrength * flicker;
  }

  noise(t) {
    // Simple noise function
    return (Math.sin(t) * 0.5 + 0.5) * (Math.cos(t * 1.3) * 0.5 + 0.5);
  }
}
```

### Example 2: Rune Activation Sequence

```javascript
// Rune powers up when player approaches
export const vfxEffects = {
  // Stage 1: Dim glow when visible
  runeDormant: {
    type: "bloom",
    target: "runeObject",
    threshold: 0.8,
    strength: 0.5,
    radius: 0.2,
    criteria: {
      runeVisible: true,
      playerDistance: { $gt: 5 }
    }
  },

  // Stage 2: Brighter when closer
  runeAwakening: {
    type: "bloom",
    target: "runeObject",
    threshold: 0.5,
    strength: 1.5,
    radius: 0.4,
    criteria: {
      runeVisible: true,
      playerDistance: { $lte: 5, $gt: 2 }
    }
  },

  // Stage 3: Full intensity when interacting
  runeActive: {
    type: "bloom",
    target: "runeObject",
    threshold: 0.3,
    strength: 3.0,
    radius: 0.6,
    pulse: {
      min: 2.0,
      max: 4.0,
      speed: 3.0
    },
    criteria: {
      runeInteracting: true
    }
  }
};
```

### Example 3: Supernatural Atmosphere

```javascript
// Eerie ambient glow throughout area
export const vfxEffects = {
  spookyAtmosphere: {
    type: "bloom",
    targets: ["ambientOrb1", "ambientOrb2", "ambientOrb3"],
    threshold: 0.4,
    strength: 0.8,
    radius: 0.8,
    color: "#6644ff",  // Slightly purple tint
    criteria: {
      currentZone: "haunted_area",
      nightTime: true
    }
  }
};
```

---

## Part 7: Bloom Material Setup

### Making Objects Bloom

```javascript
// For an object to bloom, it needs to emit light
// This is done through the material

// Method 1: Using Emissive (recommended)
const bloomMaterial = new THREE.MeshStandardMaterial({
  color: 0x00ff00,           // Green color
  emissive: 0x00ff00,         // Same green emission
  emissiveIntensity: 1.0    // Full intensity
});

// Method 2: Using BasicMaterial (always bright)
const alwaysBrightMaterial = new THREE.MeshBasicMaterial({
  color: 0x00ff00
  // Basic materials don't react to lighting
  // So they always appear "bright" = bloom
});

// Method 3: Using point lights nearby
const pointLight = new THREE.PointLight(0x00ff00, 1, 10);
pointLight.position.set(0, 2, 0);
scene.add(pointLight);
```

### Threshold Tuning

```javascript
// Threshold determines what blooms
// 0 = everything blooms, 1 = only very bright things bloom

class BloomThresholdGuide {
  // 0.0 - 0.3: Almost everything blooms (dreamy, hazy)
  // 0.3 - 0.5: Lights and emissive objects bloom (magical)
  // 0.5 - 0.7: Bright lights only (realistic)
  // 0.7 - 0.9: Very bright only (subtle)
  // 0.9 - 1.0: Almost no bloom (minimal)

  getRecommendedThreshold(purpose) {
    switch (purpose) {
      case 'magical':
        return 0.4;  // Lower threshold = more blooms

      case 'realistic':
        return 0.7;  // Higher threshold = selective

      case 'dreamy':
        return 0.2;  // Very low = hazy glow

      case 'subtle':
        return 0.8;  // High = minimal bloom

      default:
        return 0.5;
    }
  }
}
```

---

## Part 8: Advanced Techniques

### Colored Bloom

```javascript
// Tint the bloom with a specific color
class ColoredBloomEffect {
  constructor(bloomPass, color) {
    this.bloomPass = bloomPass;
    this.tintColor = new THREE.Color(color);
  }

  apply() {
    // UnrealBloomPass doesn't directly support tinting
    // So we add a tint pass after bloom
    this.tintPass = new ShaderPass(ColorTintShader);
    this.tintPass.uniforms['tintColor'].value = this.tintColor;
  }
}

const ColorTintShader = {
  uniforms: {
    tDiffuse: { value: null },
    tintColor: { value: new THREE.Vector3(1, 1, 1) }
  },

  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform vec3 tintColor;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      // Tint only the bright areas (bloom)
      float brightness = max(max(color.r, color.g), color.b);
      float tintAmount = smoothstep(0.5, 1.0, brightness);
      vec3 finalColor = mix(color.rgb, color.rgb * tintColor, tintAmount);
      gl_FragColor = vec4(finalColor, color.a);
    }
  `
};
```

### Different Bloom per Object

```javascript
// Using multiple bloom passes for different objects
class MultiBloomComposer {
  constructor() {
    this.bloomPasses = new Map();

    // Create separate bloom pass for each object/group
    this.createBloomPass('strong', { threshold: 0.3, strength: 3.0 });
    this.createBloomPass('medium', { threshold: 0.6, strength: 1.5 });
    this.createBloomPass('weak', { threshold: 0.8, strength: 0.5 });
  }

  createBloomPass(name, config) {
    const pass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      config.strength,
      0.4,
      config.threshold
    );
    this.bloomPasses.set(name, pass);
  }

  render(bloomAssignments) {
    // bloomAssignments = {
    //   objectName: 'strong' | 'medium' | 'weak'
    // }

    for (const [objectName, bloomLevel] of Object.entries(bloomAssignments)) {
      const bloomPass = this.bloomPasses.get(bloomLevel);
      const object = scene.getObjectByName(objectName);

      if (object && bloomPass) {
        // Render object with its specific bloom
        this.renderObjectWithBloom(object, bloomPass);
      }
    }
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Threshold Too Low

```javascript
// ❌ WRONG: Everything glows
{
  threshold: 0.1,  // Too low!
  strength: 2.0
}
// Scene looks hazy and washed out

// ✅ CORRECT: Selective glow
{
  threshold: 0.7,  // Only bright things glow
  strength: 2.0
}
// Clear, with intentional highlights
```

### 2. Strength Too High

```javascript
// ❌ WRONG: Overwhelming bloom
{
  threshold: 0.6,
  strength: 5.0  // Way too high!
}
// Can't see details, everything blown out

// ✅ CORRECT: Balanced glow
{
  threshold: 0.6,
  strength: 1.5  // Noticeable but not overwhelming
}
// Pleasant glow, still visible
```

### 3. Radius Too Large

```javascript
// ❌ WRONG: Giant glow blobs
{
  threshold: 0.5,
  strength: 1.5,
  radius: 1.0  // Glow spreads too far!
}
// Objects lose definition, messy look

// ✅ CORRECT: Controlled spread
{
  threshold: 0.5,
  strength: 1.5,
  radius: 0.3  // Tight, controlled glow
}
// Clean, purposeful glow
```

### 4. Forgetting to Enable Bloom Layer

```javascript
// ❌ WRONG: Object won't bloom
const rune = new THREE.Mesh(geometry, material);
scene.add(rune);
// Forgot to set layer - no bloom!

// ✅ CORRECT: Enable bloom layer
const rune = new THREE.Mesh(geometry, material);
rune.layers.enable(0);  // Main layer
rune.layers.enable(1);  // Bloom layer ← Important!
scene.add(rune);
```

---

## Performance Considerations

### Bloom Performance Cost

```
BLOOM COST FACTORS:

Resolution (BIGGEST IMPACT):
├── 1080p:  1x cost (baseline)
├── 1440p:  1.8x cost
└── 4K:      8x cost (very expensive!)

Radius:
├── 0.2:   Low cost (few blur iterations)
├── 0.5:   Medium cost
└── 1.0+:  High cost (many blur iterations)

Threshold:
├── Low:   More pixels to blur (slower)
└── High:  Fewer pixels to blur (faster)

Number of Bloom Objects:
├── 1-3:   Negligible impact
├── 4-10:  Minor impact
└── 10+:   Noticeable performance hit

RECOMMENDATION:
- Use 0.5-1.0px render resolution for mobile
- Keep radius under 0.5 for performance
- Limit concurrent bloom objects to ~10
- Use threshold > 0.5 to reduce bloom area
```

### Optimization Techniques

```javascript
class BloomOptimizer {
  adjustForPerformance(profile, bloomPass) {
    switch (profile) {
      case 'mobile':
        bloomPass.resolution.set(256, 256);  // Low res bloom
        bloomPass.radius = 0.3;
        bloomPass.threshold = 0.8;
        break;

      case 'laptop':
        bloomPass.resolution.set(512, 512);
        bloomPass.radius = 0.4;
        bloomPass.threshold = 0.7;
        break;

      case 'desktop':
        bloomPass.resolution.set(1024, 1024);
        bloomPass.radius = 0.5;
        bloomPass.threshold = 0.6;
        break;
    }
  }

  // Skip bloom every other frame on low-end
  shouldRenderBloom(frameCount) {
    return frameCount % 2 === 0;  // Bloom at 30fps for 60fps game
  }
}
```

---

## 🎮 Game Design Perspective

### Bloom as Storytelling

```javascript
// Bloom communicates importance and magic

Mundane object (no bloom):
{
  ordinaryRock: { bloom: false }
  // Just a rock, nothing special
}

Important object (subtle bloom):
{
  questItem: {
    bloom: { strength: 0.8, threshold: 0.7 }
  }
  // Slight glow - "this matters"
}

Magical object (strong bloom):
{
  magicalArtifact: {
    bloom: { strength: 2.5, threshold: 0.4, pulse: true }
  }
  // Strong glow - "this is POWERFUL"
}

Supernatural entity (intense bloom):
{
  ghostEntity: {
    bloom: { strength: 4.0, threshold: 0.2, pulse: { speed: 5.0 } }
  }
  // Overwhelming glow - "this is OTHERWORLDLY"
}
```

### Emotional Progression

```javascript
// Bloom intensity tracks with emotional beats

Stage 1 - Normal:
{
  bloom: { strength: 0.5 }
  // Subtle, atmosphere
}

Stage 2 - Magical reveal:
{
  bloom: { strength: 1.5, pulse: true }
  // Wonder and awe
}

Stage 3 - Power building:
{
  bloom: { strength: 2.5, pulse: { speed: 2.0 } }
  // Growing intensity
}

Stage 4 - Climax:
{
  bloom: { strength: 4.0, pulse: { speed: 4.0 } }
  // Overwhelming power
}
```

---

## Next Steps

Now that you understand selective bloom:

- [VFXManager](./vfx-manager.md) - Base visual effects system
- [Dissolve Effect](./dissolve-effect.md) - Object transitions
- [Glitch Post-Processing](./glitch-post-processing.md) - Digital distortion
- [Audio-Reactive Lighting](./audio-reactive-lighting.md) - Light that responds to music

---

## References

- [Three.js UnrealBloomPass](https://threejs.org/docs/#examples/en/postprocessing/UnrealBloomPass) - Bloom effect
- [Three.js Layers](https://threejs.org/docs/#api/en/core/Layers) - Object layering system
- [Post-Processing](https://threejs.org/docs/#examples/en/postprocessing/EffectComposer) - Effect composition
- [WebGL Framebuffers](https://www.khronos.org/opengl/wiki/Framebuffer_Object) - Render targets

*Documentation last updated: January 12, 2026*
