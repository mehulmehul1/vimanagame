# STORY-002-002: Depth & Thickness Rendering Pipeline

**Epic**: `EPIC-002` - WaterBall Fluid Simulation System
**Story ID**: `STORY-002-002`
**Points**: `5`
**Status**: `Ready for Dev`
**Owner**: `TBD`

---

## User Story

As a **player**, I want **water with realistic depth and thickness**, so that **light interacts with the water volume naturally**.

---

## Overview

Implement the multi-pass rendering pipeline that generates depth maps and thickness maps from particle positions. These textures are critical inputs for the final fluid shader.

**Source:**
- [WaterBall FluidRenderer](https://github.com/matsuoka-601/WaterBall/blob/master/render/fluidRender.ts)
- [depthMap.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/depthMap.wgsl)
- [thicknessMap.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/thicknessMap.wgsl)

---

## Technical Specification

### Rendering Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  DEPTH PASS                                                         │
│  Input: particle positions                                         │
│  Output: depthMapTexture (r32float)                                │
│  - Render particles as stretched quads                              │
│  - Store view-space depth in red channel                            │
│  - 4x bilateral filter passes for smoothing                        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  THICKNESS PASS                                                     │
│  Input: particle positions                                         │
│  Output: thicknessTexture (r16float)                                │
│  - Render particles as quads                                       │
│  - Accumulate thickness (alpha blending: additive)                  │
│  - 1x gaussian filter pass for smoothing                           │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  FLUID PASS (STORY-002-003)                                        │
│  Input: depthMap + thicknessMap + cubemap                          │
│  Output: final water render                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Textures to Create

```typescript
// Depth texture
const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'r32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

// Temporary depth texture (for ping-pong filtering)
const tmpDepthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'r32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

// Thickness texture
const thicknessTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'r16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

// Temporary thickness texture
const tmpThicknessTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'r16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

// Depth texture for depth testing
const depthTestTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

### 2. Depth Map Shader (`depthMap.wgsl`)

**Vertex Shader:**
- Renders each particle as a stretched quad (based on velocity)
- Stretches particles along velocity direction for motion blur effect
- Size scales with particle density: `sphere_size * clamp(density / restDensity * 4.0, 0, 1)`

**Fragment Shader:**
```wgsl
// Render circle, discard outside
var normalxy: vec2f = input.uv * 2.0 - 1.0;
var r2: f32 = dot(normalxy, normalxy);
if (r2 > 1.0) { discard; }
var normalz = sqrt(1.0 - r2);

// Calculate depth at surface of sphere
var radius = uniforms.sphere_size / 2;
var real_view_pos = input.view_position + normal * radius;
var clip_space_pos = uniforms.projection_matrix * vec4(real_view_pos, 1.0);

// Output depth (store in red channel)
out.frag_color = vec4(real_view_pos.z, 0., 0., 1.);
out.frag_depth = clip_space_pos.z / clip_space_pos.w;
```

**Clear Value:** `{ r: 1e6, ... }` (very far depth)

### 3. Bilateral Filter Shader (`bilateral.wgsl`)

Run **4 iterations** (2x horizontal, 2x vertical) for smooth depth:

```wgsl
// Bilateral filter preserves edges while smoothing
fn bilateralFilter(depth: f32, uv: vec2f) -> f32 {
    var sum: f32 = 0.0;
    var weightSum: f32 = 0.0;

    for (var i: i32 = -filterRadius; i <= filterRadius; i++) {
        let sampleDepth = textureLoad(depthTex, uv + vec2(f32(i), 0.0));
        let spatialWeight = exp(-abs(f32(i)) / sigmaSpace);
        let rangeWeight = exp(-abs(sampleDepth - depth) / sigmaRange);
        sum += sampleDepth * spatialWeight * rangeWeight;
        weightSum += spatialWeight * rangeWeight;
    }

    return sum / weightSum;
}
```

**Constants:**
- `depth_threshold`: `radius * 10`
- `max_filter_size`: `100`
- `projected_particle_constant`: `(blurFilterSize * diameter * 0.05 * screenHeight / 2) / tan(fov/2)`

### 4. Thickness Map Shader (`thicknessMap.wgsl`)

**Fragment Shader:**
```wgsl
// Render circle, discard outside
var normalxy: vec2f = input.uv * 2.0 - 1.0;
var r2: f32 = dot(normalxy, normalxy);
if (r2 > 1.0) { discard; }

// Thickness based on circle area
var thickness: f32 = sqrt(1.0 - r2);
let particle_alpha = 0.05; // Accumulation factor

// Additive blending accumulates thickness
return vec4(vec3(particle_alpha * thickness), 1.0);
```

**Blend Mode:** Additive (`srcFactor: 'one', dstFactor: 'one'`)

**Filter:** 1 iteration of Gaussian blur

### 5. File Structure

```
src/systems/fluid/render/
├── DepthThicknessRenderer.ts     # Main renderer class
├── shaders/
│   ├── depthMap.wgsl
│   ├── bilateral.wgsl
│   ├── thicknessMap.wgsl
│   └── gaussian.wgsl
└── types.ts
```

---

## Implementation Tasks

1. **[PIPELINE]** Create depth map render pipeline (depth32float + r32float color)
2. **[PIPELINE]** Create thickness map render pipeline (r16float + additive blend)
3. **[PIPELINE]** Create bilateral filter pipeline (fullscreen quad)
4. **[PIPELINE]** Create gaussian blur pipeline
5. **[BUFFERS]** Create all textures with proper formats and usage flags
6. **[RENDER]** Implement depth pass rendering stretched particle quads
7. **[RENDER]** Implement thickness pass with additive accumulation
8. **[FILTER]** Implement 4x bilateral filter ping-pong passes
9. **[FILTER]** Implement 1x gaussian blur for thickness

---

## Render Pass Descriptors

```typescript
// Depth Map Pass
const depthPass = {
    colorAttachments: [{
        view: depthTextureView,
        clearValue: { r: 1e6 }, // Clear to far depth
        loadOp: 'clear',
        storeOp: 'store',
    }],
    depthStencilAttachment: {
        view: depthTestTextureView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
    },
};

// Thickness Map Pass
const thicknessPass = {
    colorAttachments: [{
        view: thicknessTextureView,
        clearValue: { r: 0 },
        loadOp: 'clear',
        storeOp: 'store',
    }],
    // Additive blend enabled in pipeline
};
```

---

## Acceptance Criteria

- [ ] `DepthThicknessRenderer` class instantiated
- [ ] Depth texture renders with visible particle depths (debug view)
- [ ] Thickness texture accumulates particle density (debug view)
- [ ] Bilateral filter produces smooth edges without blurring particle boundaries
- [ ] All pipelines compile without WebGPU errors
- [ ] Performance: entire pipeline completes in <8ms at 1080p
- [ ] Debug views: `window.debugVimana.fluid.showDepthMap()`, `showThicknessMap()`

---

## Dependencies

- **Requires**: STORY-002-001 (particle positions from `posvelBuffer`)
- **Requires**: Cubemap texture (for STORY-002-003)
- **Blocks**: STORY-002-003 (needs depth+thickness textures for final render)

---

## Optimization Notes

- **Vertex stretching**: Reduces particle count needed for smooth appearance
- **Format selection**: r16float for thickness (enough precision, half bandwidth)
- **Filter iteration count**: 4x for depth (edges critical), 1x for thickness (less critical)
- **Ping-pong buffers**: Avoid read-write hazards during filtering

---

**Sources:**
- [WaterBall depthMap.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/depthMap.wgsl)
- [WaterBall bilateral.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/bilateral.wgsl)
- [WaterBall thicknessMap.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/thicknessMap.wgsl)
