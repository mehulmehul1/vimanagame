/**
 * bilateral.wgsl - Bilateral Filter for Depth Smoothing
 * =====================================================
 *
 * Edge-preserving bilateral filter for smoothing the depth map.
 * Preserves sharp particle boundaries while smoothing interior areas.
 *
 * Uses both spatial (distance) and range (depth difference) weights
 * to preserve edges during filtering.
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/render/bilateral.wgsl
 */

@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) uv : vec2f,
    @location(1) iuv : vec2f,
}

struct FragmentInput {
    @location(0) uv: vec2f,
    @location(1) iuv: vec2f
}

override depth_threshold: f32;
override projected_particle_constant: f32;
override max_filter_size: f32;

struct FilterUniforms {
    blur_dir: vec2f,
}

// Fullscreen vertex shader embedded here
override screenWidth: f32;
override screenHeight: f32;

@vertex
fn vs(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
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

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);

    // Skip filtering for background pixels
    if (depth >= 1e4) {
        return vec4f(vec3f(depth), 1.);
    }

    // Adaptive filter size based on depth
    var filter_size: i32 = min(i32(max_filter_size), i32(ceil(projected_particle_constant / depth)));

    // Spatial sigma for Gaussian weight
    var sigma: f32 = f32(filter_size) / 3.0;
    var two_sigma: f32 = 2.0 * sigma * sigma;

    // Range sigma for depth difference weight
    var sigma_depth: f32 = depth_threshold / 3.0;
    var two_sigma_depth: f32 = 2.0 * sigma_depth * sigma_depth;

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_depth: f32 = abs(textureLoad(texture, vec2u(input.iuv + coords * uniforms.blur_dir), 0).r);

        // Spatial weight (distance from center)
        var rr: f32 = dot(coords, coords);
        var w: f32 = exp(-rr / two_sigma);

        // Range weight (depth difference)
        var r_depth: f32 = sampled_depth - depth;
        var wd: f32 = exp(-r_depth * r_depth / two_sigma_depth);

        sum += sampled_depth * w * wd;
        wsum += w * wd;
    }

    sum /= wsum;

    return vec4f(sum, 0., 0., 1.);
}
