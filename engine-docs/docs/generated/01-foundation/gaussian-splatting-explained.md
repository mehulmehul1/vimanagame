# Gaussian Splatting - First Principles Guide

## Overview

Gaussian Splatting (often called **3DGS** or just **"splatting"**) is a revolutionary way to render 3D scenes in real-time. Unlike traditional 3D graphics that use triangles, Gaussian Splatting uses millions of tiny, fuzzy "splats" to create photorealistic scenes.

This guide explains Gaussian Splatting from first principles - assuming you've never heard of it before.

## What You Need to Know First

Before understanding Gaussian Splatting, it helps to know:
- **What a "pixel" is** - The tiny dots that make up your screen
- **Basic 3D coordinates** - X, Y, Z positions in 3D space
- **What "rendering" means** - Drawing 3D objects onto a 2D screen
- **Traditional 3D graphics** (briefly) - Using triangles/meshes

If you don't know these, don't worry - we'll explain as we go!

---

## Part 1: The Problem with Traditional 3D

### How Traditional 3D Graphics Work

Traditional 3D graphics (like in most video games) use **polygons** - usually triangles:

```
Traditional 3D Mesh:
    ▲
   ▲ ▲        ← Made of triangles
  ▲ ▲ ▲
 ▲ ▲ ▲ ▲
```

**The Process:**
1. Build 3D objects out of triangles (called a "mesh")
2. Cover triangles with materials/colors/textures
3. Use math to project 3D triangles onto 2D screen
4. Fill in the pixels

**The Problems:**
- **Lots of triangles needed** for realism (millions!)
- **Hard to get photorealism** - looks "computer-generated"
- **Complex to create** - need 3D modeling skills
- **Heavy on memory** - all those triangles take space

---

## Part 2: What Is Gaussian Splatting?

### The Core Idea

Instead of triangles, use **millions of tiny fuzzy dots** called "splats":

```
Gaussian Splatting:
  .  .  .    .  .  .
 .  .  .  .  .  .  .
.  .  .  .  .  .  .  .
```

Each dot (splat) is a **3D Gaussian** - a fancy math term for a "fuzzy blob" that:
- Has a position in 3D space (X, Y, Z)
- Has a color
- Has transparency (alpha)
- Has a size/orientation
- Fades out at the edges (that's the "Gaussian" part)

### Why "Gaussian"?

A **Gaussian** (or "normal distribution") is a bell curve shape:

```
    ╱╲
   ╱  ╲    ← Bell curve shape
  ╱    ╲
 ╱      ╲_╱
```

In 2D, it looks like a fuzzy dot. In 3D, it's a fuzzy blob. This natural fade-out makes splats blend together smoothly.

### Why "Splatting"?

When you throw something and it hits the ground hard, it makes a **"splat"** sound and flattens out. Gaussian splats are flattened onto your screen during rendering - hence the name!

---

## Part 3: How Gaussian Splatting Works

### The Basic Process

```
1. Capture Scene
   ↓
2. Create Millions of Splats
   ↓
3. Sort Splats (back to front)
   ↓
4. "Splat" onto Screen
   ↓
5. Blend Colors Together
   ↓
6. Final Image!
```

### Step 1: Creating the Splats

Splats are created from **photos or videos** of a real scene:

1. Take many photos from different angles
2. Use AI/machine learning to figure out 3D positions
3. Generate millions of splats that reproduce the scene

**Each splat contains:**
- **Position** - Where it is in 3D (X, Y, Z)
- **Color** - RGB values
- **Opacity** - How transparent (0 = invisible, 1 = solid)
- **Covariance** - Size, shape, and orientation (fancy math!)

### Step 2: Sorting (The Tricky Part)

To render correctly, splats must be drawn **back to front** (things in back drawn first, things in front drawn last).

**Why?** Think of it like painting:
```
Paint background first →
Then paint middle →
Then paint foreground →

If you paint foreground first, background will cover it!
```

**The Challenge:** With millions of splats, sorting takes time. SparkJS uses GPU-based sorting for speed.

### Step 3: Splatting (Rendering)

Each splat is "splatted" onto the screen:

1. **Project** 3D position to 2D screen coordinates
2. **Calculate** which pixels this splat affects
3. **Blend** the splat's color with existing pixels
4. **Accumulate** multiple splats overlapping

---

## Part 4: The Math (Simplified)

### What Each Splat Stores

```
struct Splat {
    position: vec3,      // X, Y, Z in 3D space
    color: vec3,         // Red, Green, Blue (0-1)
    opacity: float,      // 0 to 1
    covariance: mat3,    // Size, shape, orientation
    harmonic: vec3       // For view-dependent color (advanced)
}
```

### The Gaussian Function (Simplified)

At any pixel, the splat's contribution is:

```
contribution = color × opacity × gaussian_function(distance)

gaussian_function(d) = e^(-d²/2σ²)

Where:
- d = distance from splat center
- σ = standard deviation (splat size)
```

**Translation:** The further from center, the more the splat fades out.

---

## Part 5: Advantages Over Traditional 3D

| Gaussian Splatting | Traditional 3D |
|---------------------|----------------|
| Photorealistic from photos | Looks "CGI" |
| Millions of points | Millions of triangles |
| Great for organic scenes | Great for hard surfaces |
| New technique (2023+) | Decades of development |
| Smaller files (sometimes) | Proven compression |

---

## Part 6: The SOG File Format

The Shadow Engine uses **.SOG** files (Spark Splat Object) which contain:

```
.sog file structure:
├── Header
│   ├── Version info
│   ├── Splat count
│   └── Scene bounds
├── Splat Data (millions of entries)
│   ├── Position (3 floats)
│   ├── Color (3 floats)
│   ├── Opacity (1 float)
│   ├── Covariance (6 floats)
│   └── Harmonics (optional)
└── Metadata
    ├── Scene info
    └── Compression info
```

**File sizes in Shadow Engine:**
- `club.sog` - ~42 MB (full quality)
- `club-1m.sog` - ~18 MB (1M splats, reduced quality)
- `green-room-1m.sog` - ~11 MB

---

## Part 7: Performance Considerations

### GPU Requirements

Gaussian Splatting is **GPU-intensive** because:
- Millions of splats to process
- Complex sorting operations
- Per-pixel blending calculations

**The Shadow Engine Solution:** Performance profiles

| Profile | Target Device | Splat Count |
|---------|---------------|-------------|
| `mobile` | Phones/tablets | ~2M splats |
| `laptop` | Integrated GPU | ~5M splats |
| `desktop` | Discrete GPU | ~8M splats |
| `max` | High-end GPU | Full quality |

### Memory Management

- **Zone loading** - Load/unload areas as player moves
- **LOD (Level of Detail)** - Use fewer splats when far away
- **Culling** - Don't render off-screen splats

---

## Part 8: How SparkJS Renders Splats

From the [official SparkJS documentation](https://sparkjs.dev/docs/overview/):

> "Spark is a dynamic 3DGS renderer built for THREE.js and WebGL2 that runs in any web browser... With a handful of lines of code, anyone using THREE.js can easily add 3DGS to their scenes."

### Key Features:
- **Integrates with Three.js** - Splats live alongside meshes
- **Cross-platform** - Works on 98%+ of devices
- **Multiple splat objects** - Render many .sog files together
- **Fully dynamic** - Modify splats in real-time
- **Shader graph system** - Programmable splat manipulation

### In the Shadow Engine:

```javascript
// Simplified example from the engine
import { Spark } from '@sparkjsdev/spark';

const spark = new Spark();
await spark.init('./path/to/scene.sog');

// Spark handles:
// - Loading the .sog file
// - Creating SplatMesh objects
// - GPU-based sorting
// - Rendering each frame
```

---

## Part 9: Common Mistakes Beginners Make

### 1. Thinking More Splats = Always Better

More splats = more detail BUT:
- Slower rendering
- More memory
- Diminishing returns

**Solution:** Use appropriate quality for device

### 2. Ignoring View-Dependent Effects

Splats can change color based on viewing angle (spherical harmonics). Forgetting this looks weird!

**Solution:** Include harmonics data when capturing

### 3. Not Sorting Correctly

Wrong order = visual artifacts (things in front appearing behind)

**Solution:** Let SparkJS handle sorting automatically

### 4. Using Wrong File Format

Not all splat formats work with all renderers!

**Solution:** Use .SOG format for SparkJS

---

## Part 10: Glossary

| Term | Meaning |
|------|---------|
| **3DGS** | 3D Gaussian Splatting |
| **Splat** | A single Gaussian blob/point |
| **Covariance** | Math describing size, shape, orientation |
| **Spherical Harmonics** | Math for view-dependent color |
| **SOG** | Spark Splat Object file format |
| **Rasterization** | Drawing 3D onto 2D screen |
| **Photorealism** | Looking like a photo/real life |

---

## Summary

Gaussian Splatting is:
- **A new way** to render 3D scenes using millions of fuzzy dots
- **Photorealistic** - can look indistinguishable from reality
- **Web-ready** - runs in browsers with WebGL2
- **The future** of 3D web graphics (maybe!)

**Key Takeaway:** Instead of building 3D worlds with triangles, Gaussian Splatting captures real-world scenes as millions of tiny, glowing splats that blend together to create incredibly realistic images.

---

## Next Steps

Now that you understand Gaussian Splatting:
- [Spark Renderer Deep Dive](../03-scene-rendering/spark-renderer.md) - How SparkJS renders splats
- [SceneManager Guide](../03-scene-rendering/scene-manager.md) - Loading .sog files
- [Performance Optimization](../11-performance-platform/performance-profiles.md) - Making it run fast

---

## References

- [SparkJS Overview Documentation](https://sparkjs.dev/docs/overview/) - Official SparkJS docs
- [3D Gaussian Splatting Explained](https://www.youtube.com/watch?v=sQcrZHvrEnU) - Video tutorial
- [Beginners Guide to Gaussian Splatting](https://www.creativeailab.be/beginners-guide-to-gaussian-splatting/) - Written guide
- [3D Gaussian Splatting Tutorial](https://papers-100-lines.medium.com/3d-gaussian-splatting-tutorial-from-scratch-in-100-lines-of-pytorch-code-no-cuda-no-c-6ef104dc6419) - Code tutorial
- [A Field Guide To Gaussian Splatting](https://rd.nytimes.com/projects/gaussian-splatting-guide/) - NYT practical guide
- [Introduction to 3D Gaussian Splatting](https://huggingface.co/blog/gaussian-splatting) - HuggingFace overview
- [Original 3DGS Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Academic paper by Kerbl et al.

*Documentation last updated: January 2026*
