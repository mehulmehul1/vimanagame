# STORY-003: Fluid Surface Shader (Fresnel, Transmittance, Reflections)

**Epic**: `EPIC-001` - WaterBall Fluid Simulation System
**Story ID**: `STORY-003`
**Points**: `3`
**Status**: `Ready for Dev`
**Owner**: `TBD`

---

## User Story

As a **player**, I want **water that looks like real water**, so that **I feel immersed in the contemplative space**.

---

## Overview

Implement the final fluid surface shader that combines depth maps, thickness maps, and environment reflections to produce the WaterBall visual result.

**Source:**
- [WaterBall fluid.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/fluid.wgsl)

---

## Technical Specification

### Shader Inputs (BindGroup 0)

```typescript
BindGroup(0) = [
    { binding: 0, resource: sampler },              // linear sampler
    { binding: 1, resource: depthTextureView },     // r32float
    { binding: 2, resource: renderUniforms },       // view/projection matrices
    { binding: 3, resource: thicknessTextureView }, // r16float
    { binding: 4, resource: cubemapTextureView },   // cube map
]
```

### Fullscreen Pipeline

```typescript
const fluidPipeline = device.createRenderPipeline({
    vertex: { module: fullScreenTriangleModule },
    fragment: {
        module: fluidShaderModule,
        targets: [{ format: presentationFormat }],
    },
    primitive: { topology: 'triangle-list' },
});
```

---

## Fragment Shader Implementation

### 1. Surface Reconstruction from Depth

```wgsl
fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    // Reconstruct NDC from depth
    var ndc: vec4f = vec4f(
        tex_coord.x * 2.0 - 1.0,
        1.0 - 2.0 * tex_coord.y,
        0.0, 1.0
    );
    ndc.z = -uniforms.projection_matrix[2].z +
             uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;
    return eye_pos.xyz / eye_pos.w;
}
```

### 2. Normal Reconstruction (Sobel-style)

```wgsl
// Central difference with edge-aware selection
var ddx = getViewPosFromTexCoord(uv + texel_size.x) - viewPos;
var ddy = getViewPosFromTexCoord(uv + texel_size.y) - viewPos;

// Check backward difference for edge cases
var ddx2 = viewPos - getViewPosFromTexCoord(uv - texel_size.x);
var ddy2 = viewPos - getViewPosFromTexCoord(uv - texel_size.y);

if (abs(ddx.z) > abs(ddx2.z)) { ddx = ddx2; }
if (abs(ddy.z) > abs(ddy2.z)) { ddy = ddy2; }

var normal: vec3f = -normalize(cross(ddx, ddy));
```

### 3. Fresnel (Schlick's Approximation)

```wgsl
let F0 = 0.02; // Water reflectance at normal incidence
var fresnel: f32 = F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0);
fresnel = clamp(fresnel, 0.0, 1.0);
```

### 4. Transmittance (Beer's Law)

```wgsl
var density = 0.7;
var thickness = textureLoad(thickness_texture, uv).r;
var diffuseColor = vec3f(0.0, 0.7375, 0.95); // Cyan water
var transmittance: vec3f = exp(-density * thickness * (1.0 - diffuseColor));

// Refraction color (background seen through water)
var refractionColor = bgColor * transmittance;
```

### 5. Environment Reflection

```wgsl
var reflectionDir = reflect(rayDir, normal);
var reflectionDirWorld = (uniforms.inv_view_matrix * vec4(reflectionDir, 0.0)).xyz;
var reflectionColor = textureSampleLevel(envmap_texture, sampler, reflectionDirWorld, 0).rgb;
```

### 6. Final Color Composition

```wgsl
// Combine refraction and reflection based on fresnel
var finalColor = mix(refractionColor, reflectionColor, fresnel);

// Edge smoothing (hide gaps between particles)
let maxDeltaZ = max(max(abs(ddx.z), abs(ddy.z)),
                   max(abs(ddx2.z), abs(ddy2.z)));
if (maxDeltaZ > 1.5 * uniforms.sphere_size) {
    finalColor = mix(finalColor, vec3f(0.9), 0.4); // Blend to fog color
}

return vec4(finalColor, 1.0);
```

---

## Lighting Model

```wgsl
// Directional light (sunlight from upper-left)
var lightDir = normalize((uniforms.view_matrix * vec4f(-1, 1, -1, 0)).xyz);
var H = normalize(lightDir - rayDir); // Halfway vector
var specular = pow(max(0.0, dot(H, normal)), 250.0);
var diffuse = max(0.0, dot(lightDir, normal)) * 1.0;

// Note: specular is at 0.0 weight in current WaterBall
// Can enable for more dramatic highlights
```

---

## Color Constants

| Use | Value |
|-----|-------|
| Background fog | `vec3(0.7, 0.7, 0.75)` |
| Water diffuse | `vec3(0.0, 0.7375, 0.95)` |
| Density | `0.7` |
| F0 (fresnel) | `0.02` |
| Edge blend | `vec3(0.9)` |

---

## Implementation Tasks

1. **[SHADER]** Create `fluid.wgsl` with full fragment shader
2. **[SHADER]** Create `fullScreen.wgsl` vertex shader (draws triangle)
3. **[PIPELINE]** Create fluid render pipeline
4. **[BINDGROUP]** Set up bind group with all textures
5. **[UNIFORMS]** Create render uniforms buffer with matrices
6. **[RENDER]** Implement final fluid pass rendering
7. **[CUBEMAP]** Load or generate cubemap for reflections

---

## File Structure

```
src/systems/fluid/render/
├── FluidSurfaceRenderer.ts        # Final fluid shader renderer
├── shaders/
│   ├── fluid.wgsl                  # Main fragment shader
│   └── fullScreen.wgsl             # Fullscreen triangle vertex
└── types.ts
```

---

## Acceptance Criteria

- [ ] `FluidSurfaceRenderer` class instantiated
- [ ] Final render matches WaterBall visual (side-by-side comparison)
- [ ] Fresnel effect visible at glancing angles
- [ ] Depth-based color absorption (deeper = darker/more saturated)
- [ ] Environment reflections visible on water surface
- [ ] Edge smoothing hides particle gaps
- [ ] Performance: <3ms per frame at 1080p
- [ ] Debug view: `window.debugVimana.fluid.toggleNormals()`

---

## Dependencies

- **Requires**: STORY-001 (posvelBuffer with particle data)
- **Requires**: STORY-002 (depthTexture, thicknessTexture)
- **Requires**: Cubemap texture (6 faces: posx, negx, posy, negy, posz, negz)

---

## Notes

- **Background**: When depth >= 1e4, render background fog color (no water present)
- **Performance**: `textureLoad` is fast; avoid `textureSample` in tight loops
- **Quality**: 4 bilateral filter iterations should eliminate most artifacts
- **Cubemap**: Can procedurally generate or load from `/public/cubemap/*.png`

---

**Sources:**
- [WaterBall fluid.wgsl](https://github.com/matsuoka-601/WaterBall/blob/master/render/fluid.wgsl)
- [Schlick's Fresnel Approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation)
- [Beer-Lambert Law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law)
