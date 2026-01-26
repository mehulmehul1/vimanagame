/**
 * fluid.wgsl - Fluid Surface Fragment Shader
 * =========================================
 *
 * Final fluid surface shader combining depth maps, thickness maps,
 * and environment reflections to produce the WaterBall visual result.
 *
 * Features:
 * - Surface reconstruction from depth map
 * - Normal reconstruction using Sobel-style differences
 * - Fresnel (Schlick's approximation)
 * - Transmittance (Beer's Law)
 * - Environment reflection from cubemap
 * - Edge smoothing to hide particle gaps
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/render/fluid.wgsl
 */

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) iuv: vec2f,
}

struct FragmentInput {
    @location(0) uv: vec2f,
    @location(1) iuv: vec2f,
}

struct RenderUniforms {
    texel_size: vec2f,
    sphere_size: f32,
    inv_projection_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    view_matrix: mat4x4f,
    inv_view_matrix: mat4x4f,
}

// Bind group 0: Textures and uniforms
@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var depth_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(3) var thickness_texture: texture_2d<f32>;
@group(0) @binding(4) var envmap_texture: texture_cube<f32>;

override screenWidth: f32;
override screenHeight: f32;

// Vertex shader - fullscreen triangle
@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var pos = array(
        vec2( 1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
    );

    var uv = array(
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
    );

    out.position = vec4(pos[vertex_index], 0.0, 1.0);
    out.uv = uv[vertex_index];
    out.iuv = out.uv * vec2f(screenWidth, screenHeight);

    return out;
}

// Reconstruct view-space position from UV and depth
fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(
        tex_coord.x * 2.0 - 1.0,
        1.0 - 2.0 * tex_coord.y,
        0.0,
        1.0
    );

    // Reconstruct z from depth
    ndc.z = -uniforms.projection_matrix[2].z + uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;
    return eye_pos.xyz / eye_pos.w;
}

// Get view position from texture coordinate (using depth texture)
fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(depth_texture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

// Fragment shader - main fluid rendering
@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    // Read depth from depth texture
    var depth: f32 = abs(textureLoad(depth_texture, vec2u(input.iuv), 0).r);

    // Background fog color (when no water is present)
    let bgColor: vec3f = vec3f(0.7, 0.7, 0.75);

    // If depth is very far, render background (no water)
    if (depth >= 1e4) {
        return vec4f(bgColor, 1.0);
    }

    // Reconstruct view-space position from depth
    var viewPos: vec3f = computeViewPosFromUVDepth(input.uv, depth);

    // === NORMAL RECONSTRUCTION (Sobel-style with edge awareness) ===
    // Central difference for normal computation
    var ddx: vec3f = getViewPosFromTexCoord(input.uv + vec2f(uniforms.texel_size.x, 0.0), input.iuv + vec2f(1.0, 0.0)) - viewPos;
    var ddy: vec3f = getViewPosFromTexCoord(input.uv + vec2f(0.0, uniforms.texel_size.y), input.iuv + vec2f(0.0, 1.0)) - viewPos;

    // Check backward difference for edge cases
    var ddx2: vec3f = viewPos - getViewPosFromTexCoord(input.uv + vec2f(-uniforms.texel_size.x, 0.0), input.iuv + vec2f(-1.0, 0.0));
    var ddy2: vec3f = viewPos - getViewPosFromTexCoord(input.uv + vec2f(0.0, -uniforms.texel_size.y), input.iuv + vec2f(0.0, -1.0));

    // Use backward difference if it gives better result (edge-aware)
    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2;
    }
    if (abs(ddy.z) > abs(ddy2.z)) {
        ddy = ddy2;
    }

    // Compute normal from cross product (negated for correct orientation)
    var normal: vec3f = -normalize(cross(ddx, ddy));

    // === LIGHTING ===
    // View direction (from camera to surface)
    var rayDir = normalize(viewPos);

    // Directional light (sunlight from upper-left)
    var lightDir = normalize((uniforms.view_matrix * vec4f(-1, 1, -1, 0)).xyz);

    // Halfway vector for specular
    var H: vec3f = normalize(lightDir - rayDir);

    // Specular highlight (currently at 0 weight in WaterBall)
    var specular: f32 = pow(max(0.0, dot(H, normal)), 250.0);

    // Diffuse lighting
    var diffuse: f32 = max(0.0, dot(lightDir, normal)) * 1.0;

    // === TRANSMITTANCE (Beer's Law) ===
    // Light absorption through water depth
    let density = 0.7;
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;

    // Cyan-blue diffuse color for water
    let diffuseColor = vec3f(0.0, 0.7375, 0.95);

    // Transmittance: how much light passes through the water
    var transmittance: vec3f = exp(-density * thickness * (1.0 - diffuseColor));

    // Refraction color: background seen through water
    var refractionColor: vec3f = bgColor * transmittance;

    // === FRESNEL (Schlick's Approximation) ===
    // F0 = water reflectance at normal incidence
    let F0 = 0.02;
    var fresnel: f32 = F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0);
    fresnel = clamp(fresnel, 0.0, 1.0);

    // === ENVIRONMENT REFLECTION ===
    // Sample cubemap for reflections
    var reflectionDir: vec3f = reflect(rayDir, normal);
    var reflectionDirWorld: vec3f = (uniforms.inv_view_matrix * vec4f(reflectionDir, 0.0)).xyz;
    var reflectionColor: vec3f = textureSampleLevel(envmap_texture, texture_sampler, reflectionDirWorld, 0.0).rgb;

    // === FINAL COLOR COMPOSITION ===
    // Mix refraction and reflection based on fresnel
    // Specular is at 0.0 weight in current WaterBall implementation
    var finalColor = 0.0 * specular + mix(refractionColor, reflectionColor, fresnel);

    // === EDGE SMOOTHING ===
    // Hide gaps between particles by blending to fog color at edges
    let maxDeltaZ = max(max(abs(ddx.z), abs(ddy.z)), max(abs(ddx2.z), abs(ddy2.z)));
    if (maxDeltaZ > 1.5 * uniforms.sphere_size) {
        finalColor = mix(finalColor, vec3f(0.9), 0.4);
    }

    return vec4f(finalColor, 1.0);
}
