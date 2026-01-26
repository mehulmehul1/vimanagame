# SparkRenderer Integration - Gaussian Splat Rendering

**Shadow Engine Rendering Documentation**

---

## What You Need to Know First

Before understanding SparkRenderer integration, you should know:
- **Gaussian Splatting** - The rendering technique using point clouds (See: *Gaussian Splatting Explained*)
- **WebGL/WebGPU** - Graphics APIs for GPU rendering
- **Three.js basics** - Scene graph, meshes, and rendering loop
- **Splat files** - .ply files containing point cloud data

---

## Overview

**SparkRenderer** is the rendering system that handles Gaussian Splat point cloud visualization in the Shadow Engine. It bridges the gap between raw splat data and the Three.js scene, enabling high-performance rendering of millions of points.

### The Problem This Solves

```
Traditional 3D rendering:
    ↓
Models made of triangles (meshes)
    ↓
GPU renders triangles

Gaussian Splat rendering:
    ↓
Models made of millions of colored points
    ↓
Need specialized renderer to draw them efficiently
    ↓
SparkRenderer handles this!
```

---

## 🎮 Game Design Perspective

### Creative Intent

**Why use Gaussian Splatting instead of traditional 3D models?**

| Aspect | Traditional Meshes | Gaussian Splats |
|--------|-------------------|-----------------|
| **Photorealism** | Requires complex shaders, lighting | Capture real-world scans directly |
| **Performance** | Can be heavy with complex geometry | Efficient for organic/real-world scenes |
| **Art Style** | Clearly "CGI" or stylized | Can look like video/film footage |
| **Atmosphere** | Requires careful lighting work | Built-in atmosphere from real capture |

For Shadow Engine, Gaussian Splatting enables **photorealistic environments** that feel like captured footage rather than rendered graphics. This creates the game's distinctive visual identity - somewhere between reality and nightmare.

### Design Philosophy

**Scanned Aesthetics = Emotional Authenticity:**

```
Hand-modeled 3D art:
    ↓
Player sees: "Someone made this. It's a game."

Scanned real-world location:
    ↓
Player feels: "This place exists. This is real."
    ↓
Deeper emotional immersion
```

The slight "wrongness" of scanned footage (artifacts, imperfections) actually enhances the horror atmosphere - it feels like a document of something real.

---

## Core Concepts (Beginner Friendly)

### What is a Splat?

Think of a splat as a **soft, glowing point** in 3D space:

```
Traditional point:
    ● Single pixel, hard edge
    Simple to render

Gaussian Splat:
    ◉ Soft, glowing center
    Fades outward (Gaussian curve)
    Each splat is a tiny "blob" of color and light
```

### The .ply File Format

Splat data is stored in `.ply` (Polygon File Format) files:

```
Header:
    ply
    format ascii 1.0
    element vertex 10000000
    property float x
    property float y
    property float z
    property float red
    property float green
    property float blue
    property float alpha
    end_header

Data:
    x y z r g b a
    1.234 2.345 3.456 0.8 0.6 0.4 0.9
    ... (10 million more points)
```

Each line = one point in 3D space with color and opacity.

---

## How It Works

### Rendering Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    1. Load Splat File                   │
│  - Read .ply file                                       │
│  - Parse point data (x, y, z, r, g, b, a)              │
│  - Create GPU buffer                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    2. Sort by Depth                     │
│  - Every frame: sort points front-to-back             │
│  - Required for correct transparency blending           │
│  - Use GPU-accelerated sorting                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    3. Render Points                     │
│  - Draw each splat as a quad (4 vertices)             │
│  - Apply Gaussian shader for soft edges               │
│  - Blend with other splats behind                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    4. Post-Process                      │
│  - Apply effects (bloom, color grading)               │
│  - Composite with scene                               │
│  - Output final frame                                 │
└─────────────────────────────────────────────────────────┘
```

### Depth Sorting Explained

Why do we need to sort?

```
Unsorted rendering:
    [Back splat] [Front splat]
    ↓
    Back draws OVER front
    ↓
    WRONG! Transparency broken

Sorted rendering:
    [Front splat] [Back splat]
    ↓
    Front draws, then back draws BEHIND it
    ↓
    CORRECT! Transparency works
```

---

## Architecture

### SparkRenderer Class Structure

```javascript
/**
 * SparkRenderer - Main Gaussian Splat rendering system
 */
class SparkRenderer {
  constructor(scene, config) {
    this.scene = scene;
    this.config = config;

    // Splat storage
    this.splats = new Map();

    // Rendering components
    this.renderer = null;
    this.material = null;
    this.geometry = null;

    // Performance
    this.maxSplats = config.maxSplats || 10000000;
    this.lodEnabled = config.lodEnabled || true;

    // Initialize
    this.init();
  }

  init() {
    // Create custom renderer
    this.renderer = this.createRenderer();

    // Create splat material (shader)
    this.material = this.createSplatMaterial();

    // Create dynamic geometry buffer
    this.geometry = this.createGeometryBuffer();
  }
}
```

### Key Components

| Component | Responsibility |
|-----------|---------------|
| **SplatLoader** | Loads .ply files and parses point data |
| **SplatSorter** | GPU-accelerated depth sorting |
| **SplatMaterial** | Shader for rendering Gaussian splats |
| **SplatGeometry** | GPU buffer for point data |
| **LODManager** | Level of detail for distant splats |

---

## Usage Examples

### Loading and Rendering a Splat

```javascript
/**
 * Basic splat loading and rendering
 */
async function loadScene(scenePath) {
  // Create renderer
  const renderer = new SparkRenderer(scene, {
    maxSplats: 10000000,
    renderScale: 1.0,
    lodEnabled: true
  });

  // Load splat file
  const splat = await renderer.loadSplat({
    url: `/assets/splats/${scenePath}.ply`,
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: 1.0
  });

  // Add to scene
  renderer.addSplat(splat);

  return splat;
}
```

### Splat Configuration

```javascript
/**
 * Splat configuration options
 */
const splatConfig = {
  // File location
  url: '/assets/splats/office.ply',

  // Transform
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 },
  scale: 1.0,

  // Rendering
  maxPoints: 10000000,
  renderScale: 1.0,

  // Level of Detail
  lod: {
    enabled: true,
    distances: [25, 60, 100],
    fractions: [1.0, 0.5, 0.25, 0.1]
  },

  // Blending
  blending: {
    mode: 'normal',
    opacity: 1.0,
    threshold: 0.01
  }
};
```

### Multiple Splats in One Scene

```javascript
/**
 * Manage multiple splat scenes (zones)
 */
class SceneManager {
  constructor(renderer) {
    this.renderer = renderer;
    this.currentSplat = null;
    this.splats = new Map();
  }

  async loadZone(zoneId, splatConfig) {
    const splat = await this.renderer.loadSplat(splatConfig);
    this.splats.set(zoneId, splat);
    return splat;
  }

  transitionTo(zoneId) {
    const newSplat = this.splats.get(zoneId);
    if (!newSplat) return;

    // Crossfade transition
    this.crossfade(this.currentSplat, newSplat, 1.0);
    this.currentSplat = newSplat;
  }

  async crossfade(fromSplat, toSplat, duration) {
    // Fade out current, fade in new
    if (fromSplat) {
      await this.renderer.fadeSplat(fromSplat, 0, duration / 2);
    }
    await this.renderer.fadeSplat(toSplat, 1, duration / 2);
  }
}
```

---

## Implementation

### Splat Material (Shader)

```javascript
/**
 * Gaussian Splat vertex shader
 */
const vertexShader = `
  precision highp float;

  attribute vec3 position;
  attribute vec3 color;
  attribute float alpha;
  attribute vec3 scale;

  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;

  varying vec3 vColor;
  varying float vAlpha;
  varying vec3 vPosition;

  void main() {
    vColor = color;
    vAlpha = alpha;
    vPosition = position;

    // Calculate point size based on depth
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = scale.x * (300.0 / -mvPosition.z);

    gl_Position = projectionMatrix * mvPosition;
  }
`;

/**
 * Gaussian Splat fragment shader
 */
const fragmentShader = `
  precision highp float;

  varying vec3 vColor;
  varying float vAlpha;
  varying vec3 vPosition;

  void main() {
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    // Discard outside Gaussian radius
    if (dist > 0.5) discard;

    // Gaussian falloff
    float gaussian = exp(-dist * dist * 8.0);

    // Apply color with Gaussian alpha
    vec3 finalColor = vColor;
    float finalAlpha = vAlpha * gaussian;

    gl_FragColor = vec4(finalColor, finalAlpha);
  }
`;
```

### GPU-Based Depth Sorting

```javascript
/**
 * Depth sorting using GPU compute
 */
class SplatSorter {
  constructor(renderer) {
    this.renderer = renderer;
    this.computeShader = null;
  }

  /**
   * Sort splats by depth for correct blending
   */
  sort(splatData, cameraPosition) {
    // Calculate depth for each splat
    const depths = this.calculateDepths(splatData, cameraPosition);

    // Sort indices by depth (front to back)
    const indices = new Uint32Array(splatData.count);
    for (let i = 0; i < splatData.count; i++) {
      indices[i] = i;
    }

    indices.sort((a, b) => depths[b] - depths[a]);

    // Reorder splat data by sorted indices
    return this.reorderSplatData(splatData, indices);
  }

  calculateDepths(splatData, cameraPosition) {
    const depths = new Float32Array(splatData.count);

    for (let i = 0; i < splatData.count; i++) {
      const idx = i * 3;
      const dx = splatData.positions[idx] - cameraPosition.x;
      const dy = splatData.positions[idx + 1] - cameraPosition.y;
      const dz = splatData.positions[idx + 2] - cameraPosition.z;

      depths[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    return depths;
  }

  reorderSplatData(splatData, indices) {
    // Reorder all arrays by sorted indices
    const sortedPositions = new Float32Array(splatData.positions.length);
    const sortedColors = new Float32Array(splatData.colors.length);
    const sortedAlphas = new Float32Array(splatData.alphas.length);

    for (let i = 0; i < indices.length; i++) {
      const srcIdx = indices[i];

      sortedPositions[i * 3] = splatData.positions[srcIdx * 3];
      sortedPositions[i * 3 + 1] = splatData.positions[srcIdx * 3 + 1];
      sortedPositions[i * 3 + 2] = splatData.positions[srcIdx * 3 + 2];

      sortedColors[i * 3] = splatData.colors[srcIdx * 3];
      sortedColors[i * 3 + 1] = splatData.colors[srcIdx * 3 + 1];
      sortedColors[i * 3 + 2] = splatData.colors[srcIdx * 3 + 2];

      sortedAlphas[i] = splatData.alphas[srcIdx];
    }

    return {
      positions: sortedPositions,
      colors: sortedColors,
      alphas: sortedAlphas,
      count: splatData.count
    };
  }
}
```

### LOD (Level of Detail) System

```javascript
/**
 * LOD management for splats
 */
class SplatLOD {
  constructor(config) {
    this.distances = config.distances || [25, 60, 100];
    this.fractions = config.fractions || [1.0, 0.5, 0.25, 0.1];
  }

  /**
   * Get appropriate LOD level based on distance
   */
  getLODLevel(distance) {
    for (let i = 0; i < this.distances.length; i++) {
      if (distance < this.distances[i]) {
        return {
          level: i,
          fraction: this.fractions[i]
        };
      }
    }

    // Farthest LOD
    return {
      level: this.fractions.length - 1,
      fraction: this.fractions[this.fractions.length - 1]
    };
  }

  /**
   * Reduce splat count for LOD
   */
  reduceSplatCount(splatData, fraction) {
    const targetCount = Math.floor(splatData.count * fraction);
    const step = Math.ceil(splatData.count / targetCount);

    const lodPositions = new Float32Array(targetCount * 3);
    const lodColors = new Float32Array(targetCount * 3);
    const lodAlphas = new Float32Array(targetCount);

    for (let i = 0, j = 0; i < targetCount; i++, j += step) {
      const idx = Math.min(j, splatData.count - 1);

      lodPositions[i * 3] = splatData.positions[idx * 3];
      lodPositions[i * 3 + 1] = splatData.positions[idx * 3 + 1];
      lodPositions[i * 3 + 2] = splatData.positions[idx * 3 + 2];

      lodColors[i * 3] = splatData.colors[idx * 3];
      lodColors[i * 3 + 1] = splatData.colors[idx * 3 + 1];
      lodColors[i * 3 + 2] = splatData.colors[idx * 3 + 2];

      lodAlphas[i] = splatData.alphas[idx];
    }

    return {
      positions: lodPositions,
      colors: lodColors,
      alphas: lodAlphas,
      count: targetCount
    };
  }
}
```

---

## Performance Considerations

### Optimization Strategies

| Technique | Impact | Implementation |
|-----------|--------|----------------|
| **LOD** | Very High | Reduce splat count at distance |
| **Frustum Culling** | High | Don't render off-screen splats |
| **Render Scale** | Very High | Render at lower resolution |
| **Async Loading** | Medium | Load splats in background |
| **GPU Sorting** | High | Use compute shader for sorting |

### Memory Management

```javascript
/**
 * Memory-aware splat management
 */
class SplatMemoryManager {
  constructor(maxMemoryMB) {
    this.maxMemory = maxMemoryMB * 1024 * 1024;
    this.usedMemory = 0;
    this.splats = new Map();
  }

  canLoad(splatSize) {
    return (this.usedMemory + splatSize) <= this.maxMemory;
  }

  allocate(splatId, splatSize) {
    if (!this.canLoad(splatSize)) {
      this.freeOldest();
    }

    this.usedMemory += splatSize;
    this.splats.set(splatId, {
      size: splatSize,
      timestamp: Date.now()
    });
  }

  freeOldest() {
    let oldest = null;
    let oldestTime = Infinity;

    for (const [id, splat] of this.splats) {
      if (splat.timestamp < oldestTime) {
        oldest = id;
        oldestTime = splat.timestamp;
      }
    }

    if (oldest) {
      this.unload(oldest);
    }
  }

  unload(splatId) {
    const splat = this.splats.get(splatId);
    if (splat) {
      this.usedMemory -= splat.size;
      this.splats.delete(splatId);
    }
  }
}
```

---

## Common Mistakes Beginners Make

### Mistake 1: Not Sorting by Depth

```javascript
// BAD: Rendering without depth sorting
function renderUnsorted(splatData) {
  drawSplat(splatData);  // Wrong transparency!
}

// GOOD: Sort before rendering
async function renderSorted(splatData, camera) {
  const sorted = sortSplatByDepth(splatData, camera);
  drawSplat(sorted);
}
```

### Mistake 2: Too Many Splats

```javascript
// BAD: Loading full 10M splats on mobile
const mobileConfig = {
  maxSplats: 10000000  // Too much!
};

// GOOD: Adjust for device
const config = isMobile ? {
  maxSplats: 2000000
} : {
  maxSplats: 10000000
};
```

### Mistake 3: No LOD

```javascript
// BAD: Full detail everywhere
const noLOD = {
  lodEnabled: false
};

// GOOD: Use LOD for distance
const withLOD = {
  lodEnabled: true,
  distances: [25, 60, 100],
  fractions: [1.0, 0.5, 0.25, 0.1]
};
```

---

## Related Systems

- **Gaussian Splatting Explained** - Core concept overview
- **SceneManager** - Scene and splat management
- **ZoneManager** - Dynamic loading/unloading of splats
- **Performance Profiles** - Optimization settings

---

## Source File Reference

- **Location**: `../src/rendering/SparkRenderer.js` (hypothetical)
- **Key exports**:
  - `SparkRenderer` - Main renderer class
  - `SplatLoader` - File loading utility
  - `SplatSorter` - Depth sorting system
- **Dependencies**: Three.js, WebGL/WebGPU

---

## References

- [SparkJS.dev Documentation](https://sparkjs.dev/docs) - Gaussian Splat rendering library
- [Three.js Documentation](https://threejs.org/docs/) - 3D rendering framework
- [PLY Format Specification](https://web.engr.oregonstate.edu/~whene/PLY/) - File format reference

---

**RALPH_STATUS:**
- **Status**: SparkRenderer integration documentation complete
- **Phase**: 3 - Scene & Rendering
- **Related**: ZoneManager, LightManager (remaining)
