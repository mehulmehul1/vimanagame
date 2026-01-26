# Splat Fractal Effect - First Principles Guide

## Overview

The **Splat Fractal Effect** applies procedural fractal patterns to Gaussian splats, creating organic, self-similar visual effects like swirling smoke, crystalline structures, or supernatural energy fields. Unlike traditional texture-based effects, fractal effects are generated mathematically, creating infinite detail at any scale.

Think of the splat fractal effect as **"procedural geometry"** - just as fractals in nature (ferns, snowflakes, coastlines) show similar patterns at different scales, this effect creates self-similar visual complexity on your Gaussian splats using mathematical formulas.

## What You Need to Know First

Before understanding splat fractal effects, you should know:
- **Gaussian Splatting** - Point-based rendering with gaussian blobs
- **Fragment shaders** - Pixel-level GPU programming
- **UV coordinates** - Texture mapping coordinates (0-1)
- **Noise functions** - Generating pseudo-random values
- **Fractal mathematics** - Self-similarity and iteration
- **WebGPU compute shaders** (optional) - For GPU-based calculation

### Quick Refresher: What is a Fractal?

```
TRADITIONAL GEOMETRY:
┌─────────────────────────────────────────┐
│  Smooth, simple shapes                  │
│                                         │
│      ┌─────────┐                        │
│     ╱          ╲                       │
│    │            │                      │
│     ╲          ╱                       │
│      └─────────┘                        │
│  Circle: smooth at any scale            │
└─────────────────────────────────────────┘

FRACTAL GEOMETRY:
┌─────────────────────────────────────────┐
│  Rough, detailed, self-similar          │
│                                         │
│      ╱╲╱╲╱╲╱╲╱╲                        │
│     ╱╲╱╲╱╲╱╲╱╲╱╲                       │
│    ╱╲╱╲╱╲╱╲╱╲╱╲╱╲                      │
│                                         │
│  Pattern repeats at different scales    │
│  (zoom in → see similar pattern)        │
└─────────────────────────────────────────┘

FRACTAL BROWNIAN MOTION (FBM):
┌─────────────────────────────────────────┐
│  Layered noise for organic detail       │
│                                         │
│  Layer 1 (base):  ═══════════════       │
│  Layer 2 (detail): ══╬═╬═╬═╬═══╬═       │
│  Layer 3 (fine):   ═╬┼╬┼╬╬┼╬╬┼╬┼       │
│  ─────────────────────────────────      │
│  Combined:       ═╬┼╬┼╫┼╬┼╬╫╬┼╬┼       │
│                                         │
│  Result: Natural-looking detail!        │
└─────────────────────────────────────────┘

Why fractals for splats?
├── Infinite detail at any zoom level
├── No texture memory needed (procedural)
├── Creates organic, natural patterns
└── Perfect for supernatural effects
```

---

## Part 1: Why Use Fractal Effects on Splats?

### The Problem: Boring, Flat Splats

Without fractal effects:

```javascript
// ❌ WITHOUT Fractal Effects - Plain splats
const splatMaterial = {
  color: 0xff0000,
  opacity: 1.0
};
// Every splat looks identical.
// Boring, artificial appearance.
```

**Problems:**
- Repetitive, tiled appearance
- No organic variation
- Missing supernatural quality
- Can't create complex textures procedurally

### The Solution: Fractal Effects

```javascript
// ✅ WITH Fractal Effects - Organic splats
const splatMaterial = {
  color: 0xff0000,
  opacity: 1.0,
  fractalEffect: {
    type: "fbm",           // Fractal Brownian Motion
    octaves: 4,            // Detail layers
    persistence: 0.5,      // Detail amplitude
    lacunarity: 2.0,       // Detail frequency
    scale: 1.0             // Pattern scale
  }
};
// Each splat has unique, organic pattern.
// Natural, magical appearance.
```

**Benefits:**
- Procedural infinite detail
- Organic, natural variation
- No texture memory required
- Creates supernatural atmospheres
- Animated and dynamic patterns

---

## Part 2: Fractal Noise Fundamentals

### Value Noise vs. Gradient Noise

```javascript
// VALUE NOISE - Simple but blocky
function valueNoise(x, y) {
  const i = Math.floor(x);
  const j = Math.floor(y);
  const fx = x - i;
  const fy = y - j;

  // Get random values at grid points
  const a = hash(i, j);
  const b = hash(i + 1, j);
  const c = hash(i, j + 1);
  const d = hash(i + 1, j + 1);

  // Smooth interpolation
  const ux = smoothstep(fx);
  const uy = smoothstep(fy);

  return mix(
    mix(a, b, ux),
    mix(c, d, ux),
    uy
  );
}

// GRADIENT NOISE (Perlin) - Smoother, more natural
function gradientNoise(x, y) {
  const i = Math.floor(x);
  const j = Math.floor(y);
  const fx = x - i;
  const fy = y - j;

  // Get gradient vectors at grid points
  const ga = gradientHash(i, j);
  const gb = gradientHash(i + 1, j);
  const gc = gradientHash(i, j + 1);
  const gd = gradientHash(i + 1, j + 1);

  // Dot product with distance vectors
  const a = dot(ga, [fx, fy]);
  const b = dot(gb, [fx - 1, fy]);
  const c = dot(gc, [fx, fy - 1]);
  const d = dot(gd, [fx - 1, fy - 1]);

  // Smooth interpolation
  const ux = smoothstep(fx);
  const uy = smoothstep(fy);

  return mix(mix(a, b, ux), mix(c, d, ux), uy);
}
```

### Fractal Brownian Motion (FBM)

```javascript
// FBM combines multiple octaves of noise
// Each octave adds finer detail

function fbm(x, y, octaves, persistence, lacunarity, scale) {
  let total = 0;
  let frequency = scale;
  let amplitude = 1;
  let maxValue = 0;

  for (let i = 0; i < octaves; i++) {
    // Add noise at current frequency
    total += gradientNoise(x * frequency, y * frequency) * amplitude;

    // Track for normalization
    maxValue += amplitude;

    // Increase frequency (detail) for next octave
    frequency *= lacunarity;

    // Decrease amplitude for next octave
    amplitude *= persistence;
  }

  // Normalize to 0-1 range
  return total / maxValue;
}

// Example usage:
const value = fbm(
  0.5,           // x coordinate
  0.3,           // y coordinate
  4,             // octaves: detail layers
  0.5,           // persistence: detail amplitude
  2.0,           // lacunarity: detail frequency
  1.0            // scale: pattern size
);
```

### Domain Warping

```javascript
// Domain warping distorts the input coordinates
// Creates swirling, liquid-like patterns

function domainWarpFBM(x, y, time) {
  // First FBM for warp amount
  const warpAmount = 0.5;

  // Warp coordinates with first FBM
  const wx = x + fbm(x, y, 3, 0.5, 2.0, 1.0) * warpAmount;
  const wy = y + fbm(x + 5.2, y + 1.3, 3, 0.5, 2.0, 1.0) * warpAmount;

  // Second FBM using warped coordinates
  const pattern = fbm(wx, wy, 4, 0.5, 2.0, 1.0);

  return pattern;
}
```

---

## Part 3: GPU Shader Implementation

### Vertex Shader: Pass UV Coordinates

```glsl
// fractal.vert.glsl
attribute vec3 position;
attribute vec2 uv;

varying vec2 vUv;
varying vec3 vPosition;

uniform float time;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

void main() {
  vUv = uv;
  vPosition = position;

  // Optional: Animate position for wave effects
  vec3 animatedPos = position;
  animatedPos.y += sin(position.x * 2.0 + time) * 0.1;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(animatedPos, 1.0);

  // Set point size for Gaussian splat
  gl_PointSize = 20.0;
}
```

### Fragment Shader: FBM on Splats

```glsl
// fractal.frag.glsl
precision highp float;

varying vec2 vUv;
varying vec3 vPosition;

uniform float time;
uniform vec3 baseColor;
uniform float opacity;

// Fractal parameters
uniform int octaves;
uniform float persistence;
uniform float lacunarity;
uniform float scale;
uniform float warpAmount;

// Hash function for gradients
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)),
           dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// 2D Perlin noise
float perlinNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);

  // Smooth interpolation curve
  vec2 u = f * f * (3.0 - 2.0 * f);

  // Get gradient vectors
  vec2 ga = hash2(i);
  vec2 gb = hash2(i + vec2(1.0, 0.0));
  vec2 gc = hash2(i + vec2(0.0, 1.0));
  vec2 gd = hash2(i + vec2(1.0, 1.0));

  // Dot products
  float a = dot(ga, f);
  float b = dot(gb, f - vec2(1.0, 0.0));
  float c = dot(gc, f - vec2(0.0, 1.0));
  float d = dot(gd, f - vec2(1.0, 1.0));

  // Mix
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion
float fbm(vec2 p, int oct, float pers, float lac) {
  float value = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  float maxValue = 0.0;

  for (int i = 0; i < 8; i++) {
    if (i >= oct) break;
    value += amplitude * perlinNoise(p * frequency);
    maxValue += amplitude;
    amplitude *= pers;
    frequency *= lac;
  }

  return value / maxValue;
}

// Domain warping
vec2 domainWarp(vec2 p, float amount) {
  float q = fbm(p, 4, 0.5, 2.0);
  float r = fbm(p + q + vec2(1.7, 9.2), 4, 0.5, 2.0);
  return p + r * amount;
}

void main() {
  // Calculate distance from center for point shape
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);

  // Discard pixels outside the splat
  if (dist > 0.5) {
    discard;
  }

  // Gaussian falloff for soft edge
  float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

  // Apply fractal pattern
  vec2 warpedUV = domainWarp(vUv * scale + time * 0.1, warpAmount);
  float fractalValue = fbm(warpedUV * 3.0, octaves, persistence, lacunarity);

  // Remap from [-1, 1] to [0, 1]
  fractalValue = fractalValue * 0.5 + 0.5;

  // Apply fractal to color
  vec3 fractalColor = baseColor * (0.8 + fractalValue * 0.4);

  // Add color variation based on fractal
  fractalColor.r += fractalValue * 0.2;
  fractalColor.g += fractalValue * 0.1;
  fractalColor.b += (1.0 - fractalValue) * 0.2;

  // Final output
  gl_FragColor = vec4(fractalColor, alpha * opacity);
}
```

---

## Part 4: Fractal Effect Data Structure

### VFX Configuration

```javascript
// In vfxData.js
export const vfxEffects = {
  // Simple FBM pattern on splats
  smokeSplatPattern: {
    type: "splatFractal",
    target: "smokeSplatGroup",
    fractalType: "fbm",
    octaves: 4,
    persistence: 0.5,
    lacunarity: 2.0,
    scale: 2.0,
    warpAmount: 0.0,
    colorMix: 0.3,
    speed: 0.5,
    criteria: {
      currentState: RITUAL_ACTIVE,
      smokeEmitting: true
    }
  },

  // Domain warped fractal (swirling)
  supernaturalEnergy: {
    type: "splatFractal",
    target: "energySplatGroup",
    fractalType: "domainWarp",
    octaves: 5,
    persistence: 0.6,
    lacunarity: 2.2,
    scale: 3.0,
    warpAmount: 0.7,
    colorMix: 0.5,
    speed: 1.0,
    colorPalette: ["#4400ff", "#ff00ff", "#00ffff"],
    criteria: {
      currentState: SUPERNATURAL_EVENT,
      portalOpen: true
    }
  },

  // Voronoi-based fractal (cellular)
  crystallinePattern: {
    type: "splatFractal",
    target: "crystalSplatGroup",
    fractalType: "voronoi",
    octaves: 3,
    persistence: 0.7,
    lacunarity: 1.5,
    scale: 4.0,
    cellSize: 0.3,
    edgeSharpness: 0.8,
    criteria: {
      currentState: CRYSTAL_FORMING,
      runeActive: true
    }
  },

  // Animated flowing fractal
  flowingMist: {
    type: "splatFractal",
    target: "mistSplatGroup",
    fractalType: "flowNoise",
    octaves: 4,
    persistence: 0.5,
    lacunarity: 2.0,
    scale: 1.5,
    flowDirection: [1.0, 0.5],
    speed: 0.3,
    criteria: {
      currentZone: "graveyard",
      nightTime: true
    }
  }
};
```

### Fractal Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | Must be "splatFractal" |
| `target` | string | Yes | Splat group or object name |
| `fractalType` | string | Yes | "fbm", "domainWarp", "voronoi", "flowNoise" |
| `octaves` | number | Yes | Detail layers (1-8) |
| `persistence` | number | Yes | Detail amplitude (0-1) |
| `lacunarity` | number | Yes | Detail frequency (1-3) |
| `scale` | number | Yes | Pattern scale |
| `warpAmount` | number | No | Domain warp intensity (0-1) |
| `colorMix` | number | No | How much fractal affects color (0-1) |
| `speed` | number | No | Animation speed |
| `colorPalette` | array | No | Colors for fractal mapping |
| `cellSize` | number | No | For Voronoi: cell size |
| `flowDirection` | array | No | For flow noise: [x, y] direction |
| `criteria` | object | Yes | When effect triggers |

---

## Part 5: Splat Fractal Manager

```javascript
class SplatFractalManager {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.activeEffects = new Map();
  }

  applyFractalEffect(config) {
    const target = this.scene.getObjectByName(config.target);
    if (!target) {
      console.warn(`Splat fractal target not found: ${config.target}`);
      return;
    }

    // Create shader material with fractal
    const material = this.createFractalMaterial(config);
    target.material = material;

    // Store effect for updates
    this.activeEffects.set(config.target, {
      config,
      material,
      startTime: performance.now()
    });
  }

  createFractalMaterial(config) {
    const shaderCode = this.getFractalShader(config.fractalType);

    return new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        baseColor: { value: new THREE.Color(config.color || 0xffffff) },
        opacity: { value: config.opacity || 1.0 },
        octaves: { value: config.octaves || 4 },
        persistence: { value: config.persistence || 0.5 },
        lacunarity: { value: config.lacunarity || 2.0 },
        scale: { value: config.scale || 1.0 },
        warpAmount: { value: config.warpAmount || 0.0 }
      },
      vertexShader: shaderCode.vertex,
      fragmentShader: shaderCode.fragment,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });
  }

  getFractalShader(type) {
    const shaders = {
      fbm: this.getFBMShader(),
      domainWarp: this.getDomainWarpShader(),
      voronoi: this.getVoronoiShader(),
      flowNoise: this.getFlowNoiseShader()
    };

    return shaders[type] || shaders.fbm;
  }

  getFBMShader() {
    return {
      vertex: `
        attribute vec3 position;
        varying vec2 vUv;
        varying vec3 vPosition;

        void main() {
          vUv = uv;
          vPosition = position;
          gl_PointSize = 20.0;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragment: `
        precision highp float;
        varying vec2 vUv;

        uniform float time;
        uniform vec3 baseColor;
        uniform float opacity;
        uniform int octaves;
        uniform float persistence;
        uniform float lacunarity;
        uniform float scale;

        vec2 hash2(vec2 p) {
          p = vec2(dot(p, vec2(127.1, 311.7)),
                   dot(p, vec2(269.5, 183.3)));
          return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
        }

        float perlinNoise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);

          vec2 ga = hash2(i);
          vec2 gb = hash2(i + vec2(1.0, 0.0));
          vec2 gc = hash2(i + vec2(0.0, 1.0));
          vec2 gd = hash2(i + vec2(1.0, 1.0));

          float a = dot(ga, f);
          float b = dot(gb, f - vec2(1.0, 0.0));
          float c = dot(gc, f - vec2(0.0, 1.0));
          float d = dot(gd, f - vec2(1.0, 1.0));

          return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
        }

        float fbm(vec2 p, int oct, float pers, float lac) {
          float value = 0.0;
          float amplitude = 1.0;
          float frequency = 1.0;
          float maxValue = 0.0;

          for (int i = 0; i < 8; i++) {
            if (i >= oct) break;
            value += amplitude * perlinNoise(p * frequency);
            maxValue += amplitude;
            amplitude *= pers;
            frequency *= lac;
          }
          return value / maxValue;
        }

        void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          float dist = length(coord);
          if (dist > 0.5) discard;

          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

          vec2 noiseUV = vUv * scale + time * 0.1;
          float fractalValue = fbm(noiseUV * 3.0, octaves, persistence, lacunarity);
          fractalValue = fractalValue * 0.5 + 0.5;

          vec3 finalColor = baseColor * (0.8 + fractalValue * 0.4);

          gl_FragColor = vec4(finalColor, alpha * opacity);
        }
      `
    };
  }

  getDomainWarpShader() {
    // Similar to FBM but with coordinate warping
    const fbmShader = this.getFBMShader();
    // ... enhanced with domain warping logic
    return fbmShader;
  }

  getVoronoiShader() {
    return {
      vertex: `
        // Standard vertex shader...
      `,
      fragment: `
        precision highp float;
        varying vec2 vUv;

        uniform float time;
        uniform vec3 baseColor;
        uniform float opacity;
        uniform float scale;
        uniform float cellSize;

        vec2 voronoiRandom(vec2 coord) {
          float dotVal = dot(coord, vec2(12.9898, 78.233));
          return vec2(
            fract(sin(dotVal) * 43758.5453),
            fract(cos(dotVal) * 43758.5453)
          );
        }

        float voronoi(vec2 coord) {
          vec2 gridCoord = floor(coord / cellSize);
          vec2 localCoord = fract(coord / cellSize) - 0.5;

          float minDist = 1.0;

          for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
              vec2 offset = vec2(float(x), float(y));
              vec2 cellPoint = voronoiRandom(gridCoord + offset);
              vec2 pointPos = offset + cellPoint - localCoord;
              minDist = min(minDist, length(pointPos));
            }
          }

          return minDist;
        }

        void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          float dist = length(coord);
          if (dist > 0.5) discard;

          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

          float v = voronoi(vUv * scale + time * 0.05);
          vec3 finalColor = mix(baseColor, baseColor * 0.5, v);

          gl_FragColor = vec4(finalColor, alpha * opacity);
        }
      `
    };
  }

  getFlowNoiseShader() {
    // Flow noise with directional movement
    return {
      vertex: `
        // Standard vertex shader...
      `,
      fragment: `
        // Flow noise implementation with directional bias
        // ... similar to FBM but with flowDirection uniform
      `
    };
  }

  update(deltaTime) {
    for (const [targetName, effect] of this.activeEffects) {
      const elapsed = (performance.now() - effect.startTime) / 1000;
      effect.material.uniforms.time.value = elapsed * effect.config.speed;
    }
  }

  removeEffect(targetName) {
    const effect = this.activeEffects.get(targetName);
    if (effect) {
      this.activeEffects.delete(targetName);
      // Material will be garbage collected
    }
  }
}
```

---

## Part 6: Practical Examples

### Example 1: Supernatural Mist

```javascript
export const vfxEffects = {
  // Eerie mist that flows across the scene
  supernaturalMist: {
    type: "splatFractal",
    target: "mistSplats",
    fractalType: "domainWarp",
    octaves: 5,
    persistence: 0.6,
    lacunarity: 2.0,
    scale: 1.5,
    warpAmount: 0.8,
    colorMix: 0.4,
    speed: 0.2,
    color: "#8866aa",
    opacity: 0.3,
    criteria: {
      currentZone: "haunted_forest",
      nightTime: true
    }
  }
};

// JavaScript setup
class SupernaturalMist {
  constructor(scene) {
    this.scene = scene;

    // Create mist splat particles
    this.mistGroup = new THREE.Group();
    this.mistGroup.name = "mistSplats";

    const particleCount = 500;
    const geometry = new THREE.BufferGeometry();
    const positions = [];

    for (let i = 0; i < particleCount; i++) {
      positions.push(
        (Math.random() - 0.5) * 50,  // x
        Math.random() * 3,            // y (height)
        (Math.random() - 0.5) * 50    // z
      );
    }

    geometry.setAttribute('position',
      new THREE.Float32BufferAttribute(positions, 3)
    );

    this.mistGroup.geometry = geometry;
    this.scene.add(this.mistGroup);
  }

  applyFractal(fractalManager) {
    fractalManager.applyFractalEffect({
      target: "mistSplats",
      fractalType: "domainWarp",
      octaves: 5,
      persistence: 0.6,
      lacunarity: 2.0,
      scale: 1.5,
      warpAmount: 0.8,
      color: 0x8866aa,
      opacity: 0.3,
      speed: 0.2
    });
  }
}
```

### Example 2: Crystalline Rune Effect

```javascript
export const vfxEffects = {
  // Voronoi-based crystalline pattern
  crystallineRune: {
    type: "splatFractal",
    target: "runeSplatGroup",
    fractalType: "voronoi",
    octaves: 3,
    persistence: 0.7,
    lacunarity: 1.5,
    scale: 4.0,
    cellSize: 0.2,
    edgeSharpness: 0.9,
    colorPalette: ["#ff0000", "#ff4400", "#ff8800"],
    speed: 0.1,
    criteria: {
      runeActive: true,
      runePower: { $gte: 0.5 }
    }
  }
};

// Implementation
class CrystallineRuneEffect {
  applyVoronoiToSplats(runeGroup, config) {
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        cellSize: { value: config.cellSize },
        edgeSharpness: { value: config.edgeSharpness },
        palette: { value: this.createPaletteTexture(config.colorPalette) }
      },
      vertexShader: this.voronoiVertexShader,
      fragmentShader: this.voronoiFragmentShader,
      transparent: true,
      blending: THREE.AdditiveBlending
    });

    runeGroup.material = material;
  }

  createPaletteTexture(colors) {
    const canvas = document.createElement('canvas');
    canvas.width = colors.length;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');

    colors.forEach((color, i) => {
      ctx.fillStyle = color;
      ctx.fillRect(i, 0, 1, 1);
    });

    return new THREE.CanvasTexture(canvas);
  }
}
```

### Example 3: Energy Portal Effect

```javascript
export const vfxEffects = {
  // Swirling portal with multiple fractal layers
  energyPortal: {
    type: "splatFractal",
    target: "portalSplats",
    fractalType: "domainWarp",
    layers: [
      {
        octaves: 6,
        persistence: 0.5,
        lacunarity: 2.0,
        scale: 2.0,
        warpAmount: 1.0,
        color: "#0000ff",
        speed: 1.0
      },
      {
        octaves: 4,
        persistence: 0.6,
        lacunarity: 2.5,
        scale: 4.0,
        warpAmount: 0.5,
        color: "#ff00ff",
        speed: 0.5
      },
      {
        octaves: 3,
        persistence: 0.7,
        lacunarity: 3.0,
        scale: 8.0,
        warpAmount: 0.2,
        color: "#00ffff",
        speed: 0.25
      }
    ],
    criteria: {
      portalOpen: true
    }
  }
};
```

---

## Part 7: Advanced Techniques

### Time-Varying Fractals

```javascript
class AnimatedFractalSplat {
  constructor() {
    this.timeOffset = 0;
  }

  // Animate fractal parameters over time
  animateFractal(material, time) {
    const config = material.userData.fractalConfig;

    // Pulse the octaves for breathing effect
    const pulse = Math.sin(time * 0.5) * 0.5 + 0.5;
    material.uniforms.octaves.value = Math.floor(
      config.baseOctaves + pulse * 2
    );

    // Slowly rotate the fractal
    material.uniforms.rotation.value = time * 0.1;

    // Modulate scale for pulse effect
    material.uniforms.scale.value = config.baseScale *
      (1.0 + Math.sin(time * config.pulseSpeed) * 0.2);

    // Color cycle through palette
    if (config.colorPalette) {
      const colorIndex = (time * 0.1) % config.colorPalette.length;
      const nextIndex = (colorIndex + 1) % config.colorPalette.length;
      const blend = (time * 0.1) % 1.0;

      const color1 = new THREE.Color(config.colorPalette[Math.floor(colorIndex)]);
      const color2 = new THREE.Color(config.colorPalette[Math.floor(nextIndex)]);

      material.uniforms.baseColor.value.copy(color1).lerp(color2, blend);
    }
  }
}
```

### Multi-Channel Fractals

```javascript
// Use different fractal values for different color channels
class MultiChannelFractal {
  getMultiChannelShader() {
    return {
      vertex: `
        // Standard vertex shader
      `,
      fragment: `
        precision highp float;
        varying vec2 vUv;

        uniform float time;
        uniform float scale;

        // FBM function...

        void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          float dist = length(coord);
          if (dist > 0.5) discard;

          float alpha = 1.0 - smoothstep(0.0, 0.5, dist);

          // Different fractal for each channel
          float r = fbm(vUv * scale + time * 0.1, 4, 0.5, 2.0);
          float g = fbm(vUv * scale + time * 0.1 + vec2(5.2, 1.3), 5, 0.6, 2.2);
          float b = fbm(vUv * scale + time * 0.1 + vec2(8.7, 3.4), 3, 0.4, 1.8);

          vec3 color = vec3(r * 0.5 + 0.5, g * 0.5 + 0.5, b * 0.5 + 0.5);

          gl_FragColor = vec4(color, alpha);
        }
      `
    };
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Too Many Octaves

```javascript
// ❌ WRONG: Performance killer
{
  octaves: 16  // Way too many!
}
// Runs at 5 FPS

// ✅ CORRECT: Balanced
{
  octaves: 4  // Good quality/performance
}
// Runs smoothly
```

### 2. Scale Too Small

```javascript
// ❌ WRONG: Pattern looks like noise
{
  scale: 100.0  // Too small, looks like TV static
}

// ✅ CORRECT: Visible pattern
{
  scale: 2.0  // Nice visible pattern
}
```

### 3. No Time Variation

```javascript
// ❌ WRONG: Static fractal
static material = new ShaderMaterial({
  uniforms: {
    time: { value: 0 }  // Never changes!
  }
});
// Boring, looks like texture

// ✅ CORRECT: Animated fractal
function update(time) {
  material.uniforms.time.value = time;
}
// Alive and magical
```

### 4. Wrong Persistence Value

```javascript
// ❌ WRONG: Too much detail
{
  persistence: 0.9  // Each octave is almost as strong
}
// Chaotic, messy look

// ✅ CORRECT: Balanced detail
{
  persistence: 0.5  // Each octave is half as strong
}
// Natural, pleasing detail
```

---

## Performance Considerations

```
FRACTAL EFFECT COST:

Per-splat calculation:
├── 1-2 octaves:   Low cost (mobile OK)
├── 3-4 octaves:   Medium cost (balanced)
├── 5-6 octaves:   High cost (desktop)
└── 7+ octaves:    Very high cost (avoid)

Splat count:
├── < 1000:       Negligible
├── 1000-5000:    Minor impact
├── 5000-10000:   Noticeable
└── 10000+:       Significant impact

Optimization tips:
├── Use lower octaves for distant splats
├── Cache fractal values when possible
├── Use lower precision (mediump) on mobile
└── Consider pre-baking for static patterns
```

---

## 🎮 Game Design Perspective

### Fractals for Mood

```javascript
// Fractal parameters affect emotional tone

Calm, peaceful:
{
  octaves: 3,
  persistence: 0.4,
  speed: 0.1,
  warpAmount: 0.2
  // Gentle, slow variation
}

Mysterious, eerie:
{
  octaves: 5,
  persistence: 0.6,
  speed: 0.3,
  warpAmount: 0.6
  // Complex, swirling patterns
}

Chaotic, dangerous:
{
  octaves: 6,
  persistence: 0.8,
  speed: 1.0,
  warpAmount: 1.0
  // Intense, turbulent
}
```

### Visual Storytelling

```javascript
// Fractal evolution tells a story

Stage 1 - Normal:
{
  fractalType: "fbm",
  octaves: 2,
  warpAmount: 0.0
  // Plain, ordinary
}

Stage 2 - Something wrong:
{
  fractalType: "domainWarp",
  octaves: 4,
  warpAmount: 0.3
  // Subtle distortion
}

Stage 3 - Reality breaking:
{
  fractalType: "domainWarp",
  octaves: 6,
  warpAmount: 0.8,
  speed: 2.0
  // Severe distortion
}

Stage 4 - Transformed:
{
  fractalType: "voronoi",
  octaves: 5,
  scale: 10.0
  // Completely different reality
}
```

---

## Next Steps

Now that you understand splat fractal effects:

- [VFXManager](./vfx-manager.md) - Base visual effects system
- [Dissolve Effect](./dissolve-effect.md) - Transition effects
- [Contact Shadows](./contact-shadows.md) - Grounding shadows
- [Audio-Reactive Lighting](./audio-reactive-lighting.md) - Sound-driven visuals

---

## References

- [Perlin Noise](https://mrl.nyu.edu/~perlin/noise/) - Original noise paper
- [Simplex Noise](https://web.archive.org/web/20150318033316/http://www.staff.science.uu.nl/~fleet001/5-acgor-15-simplex_noise-slides.pdf) - Improved noise algorithm
- [GPU Gems - Chapter 5](https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch05.html) - Implementing noise in shaders
- [The Book of Shaders](https://thebookofshaders.com/) - Shader programming guide
- [WebGPU Shading Language](https://www.w3.org/TR/WGSL/) - Shader syntax reference

*Documentation last updated: January 12, 2026*
