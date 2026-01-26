# WebGPU/WebGL Rendering Pipeline - First Principles Guide

## Overview

The **rendering pipeline** is the journey that 3D data takes from your code to the pixels on your screen. Understanding this pipeline is crucial for building performant 3D web experiences.

This document explains how the Shadow Engine uses **WebGPU** (the modern standard) and **WebGL** (the legacy standard) to render Gaussian Splatting scenes.

## What You Need to Know First

Before understanding the rendering pipeline, you should know:

1. **What a GPU is** - Graphics Processing Unit, a specialized processor for graphics
2. **Basic 3D coordinates** - X (left/right), Y (up/down), Z (forward/backward)
3. **What a "frame" is** - One complete image rendered to screen (typically 60fps = 60 frames per second)
4. **JavaScript basics** - Functions, promises, async/await
5. **Canvas elements** - HTML `<canvas>` is where graphics are drawn

### What is a "Pipeline" in Graphics?

Think of a pipeline like an **assembly line in a factory**:

```
Raw Materials (3D Data)
    ↓
[Station 1: Process vertices]
    ↓
[Station 2: Assemble triangles]
    ↓
[Station 3: Calculate colors]
    ↓
[Station 4: Draw pixels]
    ↓
Finished Product (Screen Image)
```

Each "station" is a stage in the rendering pipeline. The GPU runs these stages in parallel for maximum performance.

---

## Part 1: WebGL vs WebGPU - The Evolution

### WebGL (2011) - The First Generation

**WebGL** (Web Graphics Library) brought 3D graphics to web browsers. It's a JavaScript port of **OpenGL ES 2.0** - a graphics standard from the 1990s.

**Key Characteristics:**
- Based on OpenGL ES 2.0 (older technology)
- Designed primarily for drawing graphics
- Uses GLSL (OpenGL Shading Language) for shaders
- Global state machine (complex to manage)
- No longer being updated with new features

**What WebGL Does Well:**
- Rendering 3D graphics to a canvas
- Cross-browser compatibility
- Well-documented with many libraries (Three.js)

**WebGL's Fundamental Problems:**

1. **No More Updates** - OpenGL is no longer being developed, so WebGL won't get new GPU features
2. **Graphics-Only** - Not designed for general-purpose GPU (GPGPU) computations
3. **High CPU Overhead** - Too much work on the CPU, not enough on the GPU
4. **Complex State Management** - Easy to make mistakes

### WebGPU (2023+) - The Modern Solution

**WebGPU** is the successor to WebGL, designed for modern GPUs and modern use cases.

**Key Characteristics:**
- Based on modern native APIs: Vulkan, Metal, Direct3D 12
- First-class support for GPGPU computations
- Uses WGSL (WebGPU Shading Language) - Rust-like shader language
- Lower CPU overhead
- Active development with new features

**What WebGPU Improves:**

| WebGL | WebGPU |
|-------|--------|
| High CPU overhead | Minimal CPU overhead |
| Graphics-focused | Graphics + Compute |
| GLSL shaders | WGSL shaders (Rust-like) |
| No more updates planned | Active development |
| Limited to ~10,000 draw calls | Can handle 100,000+ draw calls |

---

## Part 2: The Graphics Pipeline - Step by Step

Let's trace a single frame through the rendering pipeline.

### The Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RENDERING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. APPLICATION STAGE (Your JavaScript Code)                           │
│     └──> Define what to render (geometry, materials, lights)           │
│                                                                         │
│  2. VERTEX STAGE (Vertex Shader)                                       │
│     └──> Transform 3D positions to 2D screen coordinates               │
│                                                                         │
│  3. RASTERIZATION STAGE (Fixed Function)                               │
│     └──> Convert triangles into pixel fragments                        │
│                                                                         │
│  4. FRAGMENT STAGE (Fragment/Pixel Shader)                             │
│     └──> Calculate the color of each pixel                             │
│                                                                         │
│  5. OUTPUT MERGER STAGE (Fixed Function)                               │
│     └──> Combine fragments, apply depth testing, write to framebuffer  │
│                                                                         │
│  6. DISPLAY STAGE                                                       │
│     └──> Present the final image to the screen                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Application Stage (Your Code)

This is where **your JavaScript code** prepares the scene:

```javascript
// What YOU do in your code:
scene.add(cube);
camera.position.z = 5;
light.intensity = 0.8;
```

**What happens:**
- Scene graph is updated
- Animation is applied
- Visibility culling (don't draw what you can't see)
- Data is prepared for the GPU

### Stage 2: Vertex Stage (Vertex Shader)

**The vertex shader** processes each "vertex" (corner point) of your 3D geometry.

**Input:**
- 3D position (x, y, z)
- Normal vectors (for lighting)
- Texture coordinates
- Other vertex attributes

**What the Vertex Shader Does:**
```wgsl
// Example WebGPU vertex shader (WGSL)
@vertex
fn vertex_main(@location(0) position: vec3f) -> @builtin(position) vec4f
{
    // 1. Apply model matrix (object's position/rotation/scale)
    // 2. Apply view matrix (camera's position/orientation)
    // 3. Apply projection matrix (perspective/orthographic)

    return vec4f(position, 1.0);
}
```

**Output:**
- 2D screen position (after perspective transformation)
- Data passed to fragment shader (interpolated)

### Stage 3: Rasterization

This is a **fixed-function stage** (you can't program it, the GPU does it automatically).

**What Rasterization Does:**

```
Input: Triangle defined by 3 vertices
Output: All pixels covered by that triangle

    V1 ●────────● V2
       │       /
       │     /
       │   /
       │ /
    V3 ●

Becomes:

███████████
███████████
███████████
```

Each triangle becomes a set of "fragments" (potential pixels).

### Stage 4: Fragment Stage (Fragment Shader)

**The fragment shader** calculates the final color for each pixel.

**Input:**
- Interpolated data from vertex shader
- Textures
- Material properties
- Lighting information

**What the Fragment Shader Does:**
```wgsl
// Example WebGPU fragment shader (WGSL)
@fragment
fn fragment_main() -> @location(0) vec4f
{
    // 1. Sample textures
    // 2. Calculate lighting
    // 3. Apply material properties
    // 4. Return final color (RGBA)

    return vec4f(1.0, 0.5, 0.0, 1.0); // Orange
}
```

**Output:**
- Final color for each pixel
- Depth value (for depth testing)

### Stage 5: Output Merger

**Fixed-function stage** that combines everything:

1. **Depth Testing** - Only draw pixels that are closer to the camera
2. **Blending** - Mix semi-transparent pixels
3. **Writing** - Final pixel values written to the framebuffer

### Stage 6: Display

The framebuffer is presented to the screen, and the user sees the final image.

---

## Part 3: WebGPU-Specific Pipeline Architecture

WebGPU introduces some important concepts that differ from WebGL.

### The WebGPU Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Web Application                      │
│                    (JavaScript / WASM)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      WebGPU API                             │
│                   (navigator.gpu)                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Browser WebGPU Implementation            │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│     Metal     │   │    Vulkan     │   │  Direct3D 12  │
│   (macOS/iOS) │   │ (Linux/Win)   │   │  (Windows)    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌─────────────────┐
                   │   Physical GPU  │
                   └─────────────────┘
```

### WebGPU Initialization Pattern

Here's how WebGPU is initialized (verified against MDN documentation):

```javascript
// Step 1: Check for WebGPU support
if (!navigator.gpu) {
  throw new Error("WebGPU not supported on this browser");
}

// Step 2: Request an adapter (represents a physical GPU)
const adapter = await navigator.gpu.requestAdapter({
  powerPreference: "high-performance" // or "low-power"
});

if (!adapter) {
  throw new Error("No appropriate GPUAdapter found");
}

// Step 3: Request a device (logical connection to the GPU)
const device = await adapter.requestDevice({
  requiredFeatures: [],
  requiredLimits: {}
});

// Step 4: Configure the canvas context
const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");

context.configure({
  device: device,
  format: navigator.gpu.getPreferredCanvasFormat(), // bgra8unorm or rgba8unorm
  alphaMode: "premultiplied"
});
```

### WebGPU Pipeline Descriptor

WebGPU uses a **pipeline descriptor** to define the entire rendering pipeline upfront:

```javascript
const renderPipeline = device.createRenderPipeline({
  // Vertex shader stage
  vertex: {
    module: shaderModule,
    entryPoint: "vertex_main",
    buffers: [vertexBufferLayout]
  },

  // Fragment shader stage
  fragment: {
    module: shaderModule,
    entryPoint: "fragment_main",
    targets: [{
      format: navigator.gpu.getPreferredCanvasFormat()
    }]
  },

  // Primitive topology
  primitive: {
    topology: "triangle-list", // or "triangle-strip", "line-list", etc.
    cullMode: "back" // or "front", "none"
  },

  // Pipeline layout (auto or explicit)
  layout: "auto" // or device.createPipelineLayout(...)
});
```

### The Command Queue Pattern

WebGPU uses a **command queue** pattern for efficiency:

```javascript
// 1. Create a command encoder
const commandEncoder = device.createCommandEncoder();

// 2. Begin a render pass
const renderPass = commandEncoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    loadOp: "clear",
    storeOp: "store"
  }]
});

// 3. Issue commands
renderPass.setPipeline(renderPipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.setBindGroup(0, uniformBindGroup);
renderPass.draw(vertexCount);

// 4. End the pass
renderPass.end();

// 5. Submit to the GPU queue
device.queue.submit([commandEncoder.finish()]);
```

---

## Part 4: Gaussian Splatting Rendering Pipeline

The Shadow Engine uses **Gaussian Splatting** via SparkJS, which has a specialized rendering pipeline that differs from traditional polygon rendering.

### Traditional vs Splat Rendering

```
TRADITIONAL POLYGON RENDERING:

CPU: Generate geometry (triangles)
  ↓
GPU Vertex Shader: Transform vertices
  ↓
GPU Rasterizer: Convert to fragments
  ↓
GPU Fragment Shader: Calculate colors
  ↓
Output: Final pixels

GAUSSIAN SPLATTING (Shadow Engine):

CPU: Load .sog file (millions of splat centers)
  ↓
GPU Sort: Sort splats by depth (every frame)
  ↓
GPU Vertex Shader: Transform splat centers
  ↓
GPU Fragment Shader: Render splat as Gaussian blob
  ↓
Output: Blended, photorealistic pixels
```

### Why Splatting Needs a Different Pipeline

1. **Depth Sorting** - Splats must be rendered back-to-front for correct blending
2. **Per-Splat Parameters** - Each splat has position, rotation, scale, opacity, color
3. **Massive Count** - Millions of splats vs. thousands of triangles
4. **Additive Blending** - Splats blend together differently than opaque polygons

### The Splat Rendering Pipeline

```javascript
// Simplified view of how SparkJS renders splats:

// 1. Load splat data
const splatData = await loadSOGFile('scene.sog');
// Contains: position (x,y,z), rotation, scale, color, opacity for each splat

// 2. Every frame:
function renderSplatFrame() {
  // 2a. Sort splats by depth (back to front)
  const sortedSplats = sortSplatsByDepth(splatData, camera);

  // 2b. Upload sorted data to GPU
  device.queue.writeBuffer(splatBuffer, 0, sortedSplats);

  // 2c. Render with specialized splat shader
  renderPass.setPipeline(splatPipeline);
  renderPass.setVertexBuffer(0, splatBuffer);
  renderPass.draw(splatCount); // Draw millions of splats

  // 2d. Each splat is rendered as a Gaussian "blob"
  //     The fragment shader calculates the Gaussian falloff
}
```

---

## Part 5: WebGPU vs WebGL API Comparison

Here's how the two APIs compare for common operations:

### Getting a Context

```javascript
// WebGL
const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");

// WebGPU
const context = canvas.getContext("webgpu");
```

### Setting Up Shaders

```javascript
// WebGL (uses GLSL)
const vsSource = `
  attribute vec4 position;
  void main() {
    gl_Position = position;
  }
`;
const vertexShader = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vertexShader, vsSource);
gl.compileShader(vertexShader);

// WebGPU (uses WGSL)
const shaderCode = `
  @vertex
  fn vertex_main(@location(0) position: vec4f) -> @builtin(position) vec4f {
    return position;
  }
`;
const shaderModule = device.createShaderModule({ code: shaderCode });
```

### Drawing Geometry

```javascript
// WebGL
gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
gl.useProgram(program);
gl.drawArrays(gl.TRIANGLES, 0, vertexCount);

// WebGPU
renderPass.setPipeline(pipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.draw(vertexCount);
```

---

## Performance Considerations

### WebGPU Performance Advantages

1. **Lower CPU Overhead** - Fewer API calls to the GPU
2. **Better Batching** - Can submit more work in fewer commands
3. **Compute Shaders** - Offload calculations to GPU
4. **Indirect Drawing** - GPU can determine what to draw

### Performance Best Practices

```javascript
// ❌ BAD: Draw one object at a time
for (const object of objects) {
  renderPass.setPipeline(object.pipeline);
  renderPass.draw(object.vertexCount);
}

// ✅ GOOD: Batch by pipeline
const byPipeline = groupByPipeline(objects);
for (const [pipeline, objs] of byPipeline) {
  renderPass.setPipeline(pipeline);
  for (const obj of objs) {
    renderPass.setVertexBuffer(0, obj.vertexBuffer);
    renderPass.draw(obj.vertexCount);
  }
}
```

### For Gaussian Splatting Specifically

1. **Use LOD (Level of Detail)** - Fewer splats for distant objects
2. **Culling** - Don't render off-screen splats
3. **Compression** - Use compressed splat formats
4. **Async Loading** - Load splat data progressively

---

## Common Mistakes Beginners Make

### 1. Forgetting to Request a Device

```javascript
// ❌ WRONG: Trying to use WebGPU without initialization
const device = navigator.gpu.device; // undefined!

// ✅ CORRECT: Always request adapter and device
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
```

### 2. Not Handling WebGPU Unavailability

```javascript
// ❌ WRONG: Assumes WebGPU is available
const context = canvas.getContext("webgpu");

// ✅ CORRECT: Check for support
if (!navigator.gpu) {
  // Fallback to WebGL or show error message
  showMessage("Your browser doesn't support WebGPU");
}
```

### 3. Ignoring Canvas Format

```javascript
// ❌ WRONG: Hardcoding format
context.configure({
  device,
  format: "bgra8unorm" // May not work on all devices
});

// ✅ CORRECT: Use preferred format
context.configure({
  device,
  format: navigator.gpu.getPreferredCanvasFormat()
});
```

### 4. Not Sorting Splats Correctly

For Gaussian Splatting, the **back-to-front sort** is critical:

```javascript
// ❌ WRONG: Drawing in any order
renderSplats(splats);

// ✅ CORRECT: Sort by depth first
const sorted = sortSplatsByDepth(splats, cameraPosition);
renderSplats(sorted);
```

### 5. Synchronous Operations in Render Loop

```javascript
// ❌ WRONG: Synchronous operations block the frame
function render() {
  const result = computeOnCPU(); // Blocks!
  renderWithResult(result);
}

// ✅ CORRECT: Use async compute shaders
function render() {
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.dispatchWorkgroups(workgroupCount);
  computePass.end();
}
```

---

## Browser Compatibility

### WebGPU Support (as of January 2026)

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 113+ | Supported |
| Edge | 113+ | Supported |
| Firefox | Nightly/Dev | Experimental |
| Safari | Technology Preview | Experimental |

### WebGL Support (Fallback)

| Browser | WebGL 2.0 | WebGL 1.0 |
|---------|-----------|-----------|
| Chrome | ✅ | ✅ |
| Firefox | ✅ | ✅ |
| Safari | ✅ | ✅ |
| Edge | ✅ | ✅ |

**The Shadow Engine uses WebGPU when available, with WebGL as fallback.**

---

## Related Systems

- [Tech Stack Overview](./tech-stack-overview.md) - Overview of all technologies used
- [Gaussian Splatting Explained](./gaussian-splatting-explained.md) - Deep dive on splatting
- [SceneManager Deep Dive](../03-scene-rendering/scene-manager.md) - How the engine manages scenes
- [GameManager Deep Dive](../02-core-architecture/game-manager-deep-dive.md) - Core engine architecture

---

## References

- [WebGPU API - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) - Accessed: January 2026
- [WebGL API - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API) - Accessed: January 2026
- [WebGPU Specification](https://www.w3.org/TR/webgpu/) - W3C Standard
- [WebGL Specification](https://www.khronos.org/registry/webgl/specs/latest/2.0/) - Khronos Group
- [WGSL Language Specification](https://www.w3.org/TR/WGSL/) - WebGPU Shading Language
- [WebGPU Best Practices](https://googlechrome.github.io/WebGPU-Samples/) - Chrome WebGPU Samples
- [Three.js WebGPU Renderer](https://threejs.org/docs/#api/en/renderers/webgpu/WebGPURenderer) - Three.js WebGPU support

*Documentation last updated: January 12, 2026*
