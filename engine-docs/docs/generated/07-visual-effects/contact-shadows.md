# Contact Shadows - First Principles Guide

## Overview

**Contact shadows** are localized shadows that appear where objects touch or come close to surfaces. Unlike cascaded shadow maps that cover large distances, contact shadows focus on the immediate area around object-ground contact points, adding crucial depth cues that make objects feel "grounded" in the world.

Think of contact shadows as the **"grounding effect"** - just as your feet cast a small shadow on the pavement when you stand, objects in your game need contact shadows to feel like they exist in 3D space rather than floating above it.

## What You Need to Know First

Before understanding contact shadows, you should know:
- **Shadow mapping** - The technique of rendering shadows from light's perspective
- **Render targets** - Off-screen buffers for intermediate rendering
- **Depth buffers** - Store distance information for each pixel
- **Bias values** - Prevent shadow acne (self-shadowing artifacts)
- **Light sources** - How directional, point, and spot lights work

### Quick Refresher: Contact vs. Cascaded Shadows

```
CASCADED SHADOW MAPS (CSM):
┌─────────────────────────────────────────────────────────┐
│  Shadows for EVERYTHING, far and near                   │
│                                                         │
│      ╱╲        ╱╲        ╱╲                           │
│     ╱  ╲      ╱  ╲      ╱  ╲                          │
│    █████    █████    █████                            │
│   ██████   ██████   ██████                            │
│  ████████ █████████████████ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│                                                         │
│  Great for: Player shadow, building shadows            │
│  Problems with: Perspective aliasing, bleeding edges    │
└─────────────────────────────────────────────────────────┘

CONTACT SHADOWS:
┌─────────────────────────────────────────────────────────┐
│  Shadows ONLY where objects touch surfaces              │
│                                                         │
│      ╱╲        ╱╲        ╱╲                           │
│     ╱  ╲      ╱  ╲      ╱  ╲                          │
│    █████    █████    █████                            │
│   ██████   ██████   ██████                            │
│  ████████ █████████████████                            │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░  │
│           ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                      ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░  │
│                                                         │
│  Great for: Small objects, grounding, detail            │
│  Problems with: Limited range, requires proximity      │
└─────────────────────────────────────────────────────────┘

BOTH TOGETHER:
┌─────────────────────────────────────────────────────────┐
│  Cascaded for far shadows + Contact for near detail     │
│                                                         │
│      ╱╲        ╱╲        ╱╲                           │
│     ╱  ╲      ╱  ╲      ╱  ╲                          │
│    █████    █████    █████                            │
│   ██████   ██████   ██████                            │
│  ████████ █████████████████ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│                                                         │
│  Best of both: Depth + Detail!                         │
└─────────────────────────────────────────────────────────┘
```

**Why contact shadows matter:** Regular shadow maps struggle with close-range shadows due to perspective aliasing and resolution limits. Contact shadows solve this by focusing only on the immediate contact area, using screen-space techniques that work at any distance. They're essential for making small objects like keys, runes, and props feel physically present in the scene.

---

## Part 1: Why Use Contact Shadows?

### The Problem: Floating Objects

Without contact shadows:

```javascript
// ❌ WITHOUT Contact Shadows - Objects feel floaty
const scene = {
  candlestick: { position: [0, 0.5, 0], hasShadow: true },
  rune: { position: [2, 0.1, 0], hasShadow: true }
};
// Regular shadows don't capture the close contact.
// Objects look like they're hovering.
```

**Problems:**
- Objects appear to float above surfaces
- Loss of depth perception at close range
- Small objects lack presence
- Props feel "placed" rather than existing

### The Solution: Contact Shadows

```javascript
// ✅ WITH Contact Shadows - Objects feel grounded
const scene = {
  candlestick: {
    position: [0, 0.5, 0],
    hasShadow: true,
    contactShadow: {
      opacity: 0.5,
      blur: 0.3,
      distance: 0.5
    }
  },
  rune: {
    position: [2, 0.1, 0],
    hasShadow: true,
    contactShadow: {
      opacity: 0.7,
      blur: 0.2,
      distance: 0.3
    }
  }
};
// Objects clearly rest on surfaces.
// Scene feels physical and tangible.
```

**Benefits:**
- Objects feel grounded and present
- Better depth perception at close range
- Enhanced visual quality for small objects
- Reduced need for high-resolution shadow maps
- Complements existing shadow systems

---

## Part 2: How Contact Shadows Work

### The Contact Shadow Algorithm

```
CONTACT SHADOW PIPELINE:

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Render     │ →  │   Ray march  │ →  │   Accumulate │
│   Depth      │    │   downward   │    │   Shadows    │
└──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │
      │                    ▼                    ▼
      │           ┌──────────────┐    ┌──────────────┐
      │           │ Check for    │ →  │  Composite   │
      │           │ occlusion    │    │  with scene  │
      │           └──────────────┘    └──────────────┘
      │                                      │
      ▼                                      ▼
┌──────────────┐                     ┌──────────────┐
│ Read depth   │                     │   Final      │
│ buffer       │                     │   Image      │
└──────────────┘                     └──────────────┘

STEP 1: DEPTH PRE-PASS
└── Render scene depth to off-screen buffer
└── Store distance from camera for each pixel

STEP 2: RAY MARCHING
└── For each pixel, cast rays downward
└── Check if ray hits geometry (using depth buffer)
└── Calculate occlusion based on hit distance

STEP 3: ACCUMULATION
└── Collect all shadow samples
└── Apply blur to soften edges
└── Fade based on distance from contact point

STEP 4: COMPOSITING
└── Multiply contact shadow with scene color
└── Result: soft shadows where objects meet surfaces
```

### Screen-Space Technique

Contact shadows are a **screen-space effect**, meaning they work in 2D after the 3D scene is rendered:

```
SCREEN SPACE REPRESENTATION:

Rendered Scene (what camera sees):
┌─────────────────────────────────────────┐
│                                         │
│         ██████                          │  ← Object
│        ████████                         │
│                                         │
│ ─────────────────────────────────────── │  ← Ground
│                                         │
└─────────────────────────────────────────┘

Depth Buffer (distance values):
┌─────────────────────────────────────────┐
│                                         │
│         3.2                             │  ← Object is close
│        3.2                              │
│                                         │
│ 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0    │  ← Ground is farther
│                                         │
└─────────────────────────────────────────┘

Contact Shadow (computed):
┌─────────────────────────────────────────┐
│                                         │
│         ░░░░░░░                         │  ← Shadow under object
│        ░░░░░░░░░                        │
│                                         │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  ← Fade with distance
│                                         │
└─────────────────────────────────────────┘

Final Result (multiplied):
┌─────────────────────────────────────────┐
│                                         │
│         ██████                          │  ← Object
│        ████████                         │
│         ▓▓▓▓▓▓▓▓                        │  ← Contact shadow
│ ██████████████████████████████████████  │  ← Ground
│                                         │
└─────────────────────────────────────────┘
```

---

## Part 3: Contact Shadow Implementation

### Basic Contact Shadow Pass

```javascript
class ContactShadowPass {
  constructor(camera, scene) {
    this.camera = camera;
    this.scene = scene;

    // Create depth render target
    this.depthTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight,
      {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        depthBuffer: true,
        depthTexture: new THREE.DepthTexture()
      }
    );

    // Create shadow render target
    this.shadowTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight,
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat
      }
    );

    // Create contact shadow material
    this.shadowMaterial = this.createShadowMaterial();
  }

  createShadowMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: this.depthTarget.depthTexture },
        cameraNear: { value: this.camera.near },
        cameraFar: { value: this.camera.far },
        shadowOpacity: { value: 0.5 },
        shadowBlur: { value: 0.3 },
        shadowDistance: { value: 0.5 }
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vWorldPos;

        void main() {
          vUv = uv;
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vWorldPos = worldPos.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;
        uniform float shadowOpacity;
        uniform float shadowBlur;
        uniform float shadowDistance;

        varying vec2 vUv;
        varying vec3 vWorldPos;

        // Convert depth buffer value to linear depth
        float readDepth(sampler2D depthSampler, vec2 coord) {
          float fragCoordZ = texture2D(depthSampler, coord).x;
          float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
          return viewZToOrthographicDepth(viewZ, cameraNear, cameraFar);
        }

        void main() {
          // Read current pixel depth
          float currentDepth = readDepth(tDepth, vUv);

          // Ray march downward to find ground
          float shadow = 0.0;
          float samples = 8.0;
          float stepSize = shadowDistance / samples;

          for (float i = 0.0; i < 8.0; i++) {
            vec2 sampleCoord = vUv + vec2(0.0, i * stepSize * 0.01);
            float sampleDepth = readDepth(tDepth, sampleCoord);

            // If sample depth is greater (farther), we found ground
            if (sampleDepth > currentDepth + 0.01) {
              float distance = i * stepSize;
              shadow = 1.0 - smoothstep(0.0, shadowDistance, distance);
              break;
            }
          }

          gl_FragColor = vec4(0.0, 0.0, 0.0, shadow * shadowOpacity);
        }
      `
    });
  }

  render(renderer) {
    // 1. Render depth to target
    const originalTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(this.depthTarget);
    renderer.render(this.scene, this.camera);

    // 2. Generate contact shadows
    renderer.setRenderTarget(this.shadowTarget);

    const quad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      this.shadowMaterial
    );

    renderer.render(quad, this.camera);

    // 3. Restore original target
    renderer.setRenderTarget(originalTarget);

    return this.shadowTarget.texture;
  }
}
```

### Simplified Distance-Based Approach

```javascript
class SimpleContactShadows {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.shadowObjects = new Map();
  }

  // Create a planar shadow for an object
  addContactShadow(object, config = {}) {
    const {
      opacity = 0.5,
      blur = 0.3,
      scale = 1.0,
      yOffset = 0.01
    } = config;

    // Create shadow plane
    const shadowGeometry = new THREE.PlaneGeometry(scale, scale);
    const shadowMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0,
      depthWrite: false
    });

    const shadow = new THREE.Mesh(shadowGeometry, shadowMaterial);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.copy(object.position);
    shadow.position.y = yOffset;

    this.scene.add(shadow);
    this.shadowObjects.set(object.uuid, {
      shadow,
      targetOpacity: opacity,
      blur
    });

    return shadow;
  }

  updateShadows(deltaTime) {
    for (const [uuid, data] of this.shadowObjects) {
      const { shadow, targetOpacity, blur } = data;

      // Calculate distance-based fade
      const distanceToCamera = this.camera.position.distanceTo(shadow.position);
      const maxDistance = 10.0;
      const fadeFactor = 1.0 - Math.min(distanceToCamera / maxDistance, 1.0);

      shadow.material.opacity = targetOpacity * fadeFactor;
      shadow.scale.setScalar(1.0 + blur * (1.0 - fadeFactor));
    }
  }

  removeContactShadow(object) {
    const data = this.shadowObjects.get(object.uuid);
    if (data) {
      this.scene.remove(data.shadow);
      this.shadowObjects.delete(object.uuid);
    }
  }
}
```

---

## Part 4: Contact Shadow Data Structure

### Shadow Configuration in VFX Data

```javascript
// In vfxData.js
export const vfxEffects = {
  // Simple contact shadow
  candlestickShadow: {
    type: "contactShadow",
    target: "candlestick",
    opacity: 0.5,
    blur: 0.3,
    distance: 0.5,
    fadeRange: 5.0,
    criteria: {
      currentZone: "ritual_room",
      candlestickVisible: true
    }
  },

  // Multiple contact shadows
  propShadows: {
    type: "contactShadow",
    targets: ["phoneBooth", "rune", "amplifierCord"],
    opacity: 0.4,
    blur: 0.2,
    distance: 0.3,
    criteria: {
      currentZone: "office"
    }
  },

  // Dynamic contact shadow (follows object)
  playerContactShadow: {
    type: "contactShadow",
    target: "player",
    opacity: 0.6,
    blur: 0.4,
    distance: 0.8,
    dynamic: true,
    updateRate: "frame",
    criteria: {
      playerExists: true
    }
  },

  // Animated contact shadow
  pulsingRuneShadow: {
    type: "contactShadow",
    target: "runeObject",
    opacity: 0.5,
    blur: 0.2,
    distance: 0.3,
    pulse: {
      min: 0.3,
      max: 0.7,
      speed: 2.0
    },
    criteria: {
      runeActive: true
    }
  }
};
```

### Contact Shadow Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | Must be "contactShadow" |
| `target` | string | Conditional | Single object to shadow |
| `targets` | array | Conditional | Multiple objects to shadow |
| `opacity` | number | Yes | Shadow darkness (0-1) |
| `blur` | number | Yes | Shadow edge softness (0-1) |
| `distance` | number | Yes | Max shadow ray distance |
| `fadeRange` | number | No | Distance to fade out shadow |
| `dynamic` | boolean | No | Update shadow position each frame |
| `updateRate` | string | No | "frame" or "tick" update frequency |
| `pulse` | object | No | Pulsing animation config |
| `criteria` | object | Yes | When shadow triggers |

---

## Part 5: Advanced Contact Shadow Techniques

### Multi-Layer Contact Shadows

```javascript
class MultiLayerContactShadows {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.layers = new Map();
  }

  addLayer(name, config) {
    const {
      objects = [],
      opacity = 0.5,
      blur = 0.3,
      renderOrder = 0
    } = config;

    const target = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    this.layers.set(name, {
      objects,
      target,
      opacity,
      blur,
      renderOrder
    });
  }

  renderLayer(renderer, layerName) {
    const layer = this.layers.get(layerName);
    if (!layer) return null;

    // Render only this layer's objects
    const allObjects = this.scene.children;
    layer.objects.forEach(obj => {
      const originalVisible = obj.visible;
      obj.visible = true;
    });

    allObjects.forEach(obj => {
      if (!layer.objects.includes(obj)) {
        obj.visible = false;
      }
    });

    renderer.render(this.scene, this.camera);

    // Restore visibility
    allObjects.forEach(obj => obj.visible = true);

    return layer.target.texture;
  }

  compositeLayers(renderer) {
    // Composite all layers in render order
    const sortedLayers = Array.from(this.layers.entries())
      .sort((a, b) => a[1].renderOrder - b[1].renderOrder);

    for (const [name, layer] of sortedLayers) {
      const texture = this.renderLayer(renderer, name);
      if (texture) {
        this.blitTexture(texture, layer.opacity);
      }
    }
  }

  blitTexture(texture, opacity) {
    const material = new THREE.ShaderMaterial({
      uniforms: {
        tTexture: { value: texture },
        opacity: { value: opacity }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tTexture;
        uniform float opacity;
        varying vec2 vUv;

        void main() {
          vec4 color = texture2D(tTexture, vUv);
          gl_FragColor = vec4(0.0, 0.0, 0.0, color.a * opacity);
        }
      `
    });

    const quad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      material
    );

    // Blit with additive blending
    quad.material.blending = THREE.MultiplyBlending;
    quad.material.transparent = true;

    // Render quad...
  }
}
```

### Soft Contact Shadows with PCSS

```javascript
// Percentage Closer Soft Shadows for contact shadows
class SoftContactShadows {
  constructor() {
    this.samples = 16;  // Number of PCSS samples
  }

  generatePCSSShader() {
    return {
      uniforms: {
        tDepth: { value: null },
        tNoise: { value: this.createNoiseTexture() },
        sampleRadius: { value: 0.02 },
        cameraNear: { value: 0.1 },
        cameraFar: { value: 100 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDepth;
        uniform sampler2D tNoise;
        uniform float sampleRadius;
        uniform float cameraNear;
        uniform float cameraFar;
        varying vec2 vUv;

        float readDepth(vec2 coord) {
          float fragCoordZ = texture2D(tDepth, coord).x;
          return viewZToOrthographicDepth(
            perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar),
            cameraNear,
            cameraFar
          );
        }

        void main() {
          float centerDepth = readDepth(vUv);
          float shadow = 0.0;

          // PCSS sampling with noise pattern
          for (int i = 0; i < 16; i++) {
            vec2 offset = texture2D(tNoise,
              vUv + float(i) * 0.1
            ).xy * 2.0 - 1.0;

            vec2 sampleCoord = vUv + offset * sampleRadius;
            float sampleDepth = readDepth(sampleCoord);

            if (sampleDepth > centerDepth + 0.001) {
              shadow += 1.0;
            }
          }

          shadow /= 16.0;

          gl_FragColor = vec4(0.0, 0.0, 0.0, shadow * 0.5);
        }
      `
    };
  }

  createNoiseTexture() {
    // Create a simple noise texture for sample distribution
    const size = 64;
    const data = new Float32Array(size * size * 2);

    for (let i = 0; i < size * size * 2; i += 2) {
      data[i] = Math.random();
      data[i + 1] = Math.random();
    }

    const texture = new THREE.DataTexture(
      data,
      size,
      size,
      THREE.RGFormat,
      THREE.FloatType
    );
    texture.needsUpdate = true;

    return texture;
  }
}
```

---

## Part 6: Practical Contact Shadow Examples

### Example 1: Phone Booth Ground Shadow

```javascript
export const vfxEffects = {
  phoneBoothContactShadow: {
    type: "contactShadow",
    target: "phoneBooth",
    opacity: 0.6,
    blur: 0.4,
    distance: 2.0,
    scale: 1.5,
    dynamic: true,
    criteria: {
      currentZone: "office",
      phoneBoothVisible: true
    }
  }
};

// Implementation
class PhoneBoothShadow {
  constructor(phoneBooth, scene) {
    this.phoneBooth = phoneBooth;
    this.scene = scene;

    // Create shadow mesh
    const geometry = new THREE.CircleGeometry(1.5, 32);
    const material = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
      blending: THREE.MultiplyBlending
    });

    this.shadow = new THREE.Mesh(geometry, material);
    this.shadow.rotation.x = -Math.PI / 2;
    this.shadow.position.y = 0.01;  // Slightly above ground

    this.scene.add(this.shadow);
  }

  update() {
    // Follow phone booth
    this.shadow.position.x = this.phoneBooth.position.x;
    this.shadow.position.z = this.phoneBooth.position.z;

    // Scale based on phone booth height from ground
    const height = this.phoneBooth.position.y;
    const scaleFactor = 1.0 + height * 0.1;
    this.shadow.scale.setScalar(scaleFactor);

    // Fade with distance
    const distance = this.phoneBooth.position.y;
    const opacity = Math.max(0, 0.6 - distance * 0.2);
    this.shadow.material.opacity = opacity;
  }

  dispose() {
    this.scene.remove(this.shadow);
    this.shadow.geometry.dispose();
    this.shadow.material.dispose();
  }
}
```

### Example 2: Rune Activation Shadow

```javascript
export const vfxEffects = {
  // Dormant state - faint shadow
  runeDormantShadow: {
    type: "contactShadow",
    target: "runeObject",
    opacity: 0.3,
    blur: 0.3,
    distance: 0.2,
    color: "#000000",
    criteria: {
      runeVisible: true,
      runeState: "dormant"
    }
  },

  // Awakening state - growing shadow
  runeAwakeningShadow: {
    type: "contactShadow",
    target: "runeObject",
    opacity: 0.5,
    blur: 0.2,
    distance: 0.4,
    animate: {
      property: "scale",
      from: 1.0,
      to: 1.5,
      duration: 2000,
      easing: "easeOutCubic"
    },
    criteria: {
      runeVisible: true,
      runeState: "awakening"
    }
  },

  // Active state - pulsing colored shadow
  runeActiveShadow: {
    type: "contactShadow",
    target: "runeObject",
    opacity: 0.7,
    blur: 0.1,
    distance: 0.5,
    color: "#ff0000",
    pulse: {
      min: 0.5,
      max: 1.0,
      speed: 3.0
    },
    criteria: {
      runeVisible: true,
      runeState: "active"
    }
  }
};
```

### Example 3: Dynamic Interactive Shadows

```javascript
class InteractiveContactShadows {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.raycaster = new THREE.Raycaster();
    this.shadows = new Map();
  }

  addInteractiveShadow(object, config) {
    const shadow = this.createContactShadowMesh(config);

    // Store shadow data
    this.shadows.set(object.uuid, {
      mesh: shadow,
      config,
      groundNormal: new THREE.Vector3(0, 1, 0)
    });

    this.scene.add(shadow);
    return shadow;
  }

  createContactShadowMesh(config) {
    const {
      size = 1.0,
      opacity = 0.5,
      color = 0x000000
    } = config;

    // Use soft gradient texture for realistic shadow
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Draw radial gradient
    const gradient = ctx.createRadialGradient(
      128, 128, 0,
      128, 128, 128
    );
    gradient.addColorStop(0, 'rgba(0, 0, 0, 1)');
    gradient.addColorStop(0.5, 'rgba(0, 0, 0, 0.5)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 256);

    const texture = new THREE.CanvasTexture(canvas);

    const geometry = new THREE.PlaneGeometry(size, size);
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: opacity,
      depthWrite: false,
      blending: THREE.MultiplyBlending,
      side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = 0.01;

    return mesh;
  }

  update(object) {
    const data = this.shadows.get(object.uuid);
    if (!data) return;

    const { mesh, config, groundNormal } = data;

    // Cast ray down to find ground
    this.raycaster.set(
      object.position,
      new THREE.Vector3(0, -1, 0)
    );

    const intersects = this.raycaster.intersectObjects(
      this.scene.children,
      true
    );

    if (intersects.length > 0) {
      const hit = intersects[0];
      mesh.position.copy(hit.point);
      mesh.position.y += 0.01;

      // Orient shadow to match ground normal
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        hit.face.normal
      );

      // Scale based on distance to ground
      const distance = object.position.y - hit.point.y;
      const scale = 1.0 + distance * 0.2;
      mesh.scale.setScalar(Math.min(scale, config.maxScale || 2.0));

      // Fade based on distance
      const maxDistance = config.maxDistance || 5.0;
      const opacity = config.opacity * (1.0 - distance / maxDistance);
      mesh.material.opacity = Math.max(0, opacity);
    }
  }
}
```

---

## Part 7: Optimizing Contact Shadows

### Performance Considerations

```
CONTACT SHADOW COST FACTORS:

Resolution:
├── 512px:   Low cost (mobile friendly)
├── 1024px:  Medium cost (good quality)
└── 2048px:  High cost (high quality)

Ray Marching Samples:
├── 4 samples:   Fast (basic quality)
├── 8 samples:   Balanced (recommended)
└── 16+ samples: Expensive (high quality)

Number of Shadowed Objects:
├── 1-3:   Negligible impact
├── 4-10:  Minor impact
└── 10+:   Noticeable performance hit

Update Frequency:
├── Every frame:   Best quality, highest cost
├── Every 2 frames: Good balance
└── Every 4+ frames: Low cost, may jitter

RECOMMENDATION:
- Use 512-1024px resolution for mobile
- Keep samples at 4-8 for performance
- Limit concurrent contact shadows to ~10
- Update every 2 frames on low-end devices
```

### Level-of-Detail Contact Shadows

```javascript
class LODContactShadows {
  constructor() {
    this.qualityProfiles = {
      low: {
        resolution: 512,
        samples: 4,
        blur: 0.2,
        updateRate: 4  // Update every 4 frames
      },
      medium: {
        resolution: 1024,
        samples: 8,
        blur: 0.3,
        updateRate: 2
      },
      high: {
        resolution: 2048,
        samples: 16,
        blur: 0.4,
        updateRate: 1
      }
    };

    this.currentProfile = 'medium';
    this.frameCounter = 0;
  }

  setQualityProfile(profile) {
    this.currentProfile = profile;
    this.applyProfile(this.qualityProfiles[profile]);
  }

  shouldRender() {
    const profile = this.qualityProfiles[this.currentProfile];
    this.frameCounter++;

    if (this.frameCounter >= profile.updateRate) {
      this.frameCounter = 0;
      return true;
    }
    return false;
  }

  autoDetectQuality() {
    // Detect device capabilities
    const gl = document.createElement('canvas').getContext('webgl2');
    if (!gl) {
      this.setQualityProfile('low');
      return;
    }

    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const rendererInfo = gl.getExtension('WEBGL_debug_renderer_info');

    if (maxTextureSize < 4096) {
      this.setQualityProfile('low');
    } else if (maxTextureSize < 8192) {
      this.setQualityProfile('medium');
    } else {
      this.setQualityProfile('high');
    }
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Shadow Too Dark

```javascript
// ❌ WRONG: Unnatural black shadow
{
  opacity: 1.0,  // Too dark!
  blur: 0.1
}
// Looks like a hole in the ground

// ✅ CORRECT: Natural subtle shadow
{
  opacity: 0.4,  // Subtle
  blur: 0.3
}
// Looks like natural contact shadow
```

### 2. Shadow Too Sharp

```javascript
// ❌ WRONG: Hard-edged shadow
{
  opacity: 0.5,
  blur: 0.0  // No blur!
}
// Unnatural cutout look

// ✅ CORRECT: Soft-edged shadow
{
  opacity: 0.5,
  blur: 0.3  // Soft edges
}
// Natural falloff
```

### 3. Shadow Too Large

```javascript
// ❌ WRONG: Giant shadow blob
{
  distance: 5.0,  // Too far!
  scale: 3.0
}
// Shadow extends unrealistically

// ✅ CORRECT: Localized shadow
{
  distance: 0.5,  // Close to object
  scale: 1.0
}
// Natural contact area
```

### 4. Updating Every Frame Unnecessarily

```javascript
// ❌ WRONG: Update static object shadows
function update() {
  staticObjectShadow.update();  // Wastes CPU!
}

// ✅ CORRECT: Cache static shadows
const staticShadows = new Set(['book', 'lamp', 'chair']);

function update() {
  for (const shadow of allShadows) {
    if (!staticShadows.has(shadow.target)) {
      shadow.update();  // Only update dynamic objects
    }
  }
}
```

---

## 🎮 Game Design Perspective

### Contact Shadows for Storytelling

```javascript
// Shadows communicate object state

Ordinary object:
{
  book: { shadow: { opacity: 0.3, blur: 0.3 } }
  // Just sitting there
}

Important object:
{
  ancientTome: {
    shadow: {
      opacity: 0.5,
      blur: 0.2,
      color: "#440000"  // Slightly red tint
    }
  }
  // "This book has presence"
}

Magical object:
{
  glowingOrb: {
    shadow: {
      opacity: 0.4,
      blur: 0.4,
      pulse: true,
      color: "#ff00ff"
    }
  }
  // "This orb is magical"
}
```

### Environmental Storytelling

```javascript
// Contact shadows reveal unseen objects

// A shadow where nothing is visible...
mysteriousShadow: {
  type: "contactShadow",
  position: [5, 0, 3],
  opacity: 0.7,
  blur: 0.15,
  scale: 1.2
  // "Something invisible is here"
  // Player investigates → reveals ghost!
}

// Shadows that move independently...
movingShadow: {
  type: "contactShadow",
  path: "circular",
  speed: 0.5,
  opacity: 0.6
  // "Something is circling..."
  // Creates tension before reveal
}
```

### Pacing and Tension

```javascript
// Shadow intensity tracks with tension

Safe area:
{
  shadowOpacity: 0.2,  // Light, casual
  shadowBlur: 0.4
  // Relaxed atmosphere
}

Tension building:
{
  shadowOpacity: 0.5,  // Darker shadows
  shadowBlur: 0.2,
  pulse: true
  // Something's wrong...
}

Danger imminent:
{
  shadowOpacity: 0.8,  // Heavy shadows
  shadowBlur: 0.1,
  pulse: { speed: 4.0 }
  // Threat is close!
}
```

---

## Next Steps

Now that you understand contact shadows:

- [VFXManager](./vfx-manager.md) - Base visual effects system
- [Selective Bloom](./selective-bloom.md) - Glowing highlights
- [Splat Fractal Effect](./splat-fractal-effect.md) - Procedural effects
- [Audio-Reactive Lighting](./audio-reactive-lighting.md) - Sound-driven visuals

---

## References

- [Three.js Shadow Maps](https://threejs.org/docs/#api/en/renderers/WebGLRenderer.shadowMap) - Shadow system
- [Screen Space Shadows](https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch11.html) - GPU Gems chapter
- [Percentage Closer Soft Shadows](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch08.html) - PCSS technique
- [WebGL Depth Textures](https://www.khronos.org/opengl/wiki/Depth_Texture) - Depth buffer usage

*Documentation last updated: January 12, 2026*
