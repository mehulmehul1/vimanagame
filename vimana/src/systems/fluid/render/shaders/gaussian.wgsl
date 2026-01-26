/**
 * gaussian.wgsl - Gaussian Blur Filter for Thickness Smoothing
 * ============================================================
 *
 * Simple Gaussian blur for smoothing the thickness map.
 * Unlike bilateral filter, this doesn't preserve edges - it's a pure blur.
 *
 * Applied once (X then Y) to the thickness map after accumulation.
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/render/gaussian.wgsl
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
    var thickness: f32 = textureLoad(texture, vec2u(input.iuv), 0).r;

    // Skip filtering for zero-thickness areas
    if (thickness == 0.) {
        return vec4f(0., 0., 0., 1.);
    }

    // Fixed filter size for thickness blur
    var filter_size: i32 = 30;
    var sigma: f32 = f32(filter_size) / 3.0;
    var two_sigma: f32 = 2.0 * sigma * sigma;

    var sum = 0.;
    var wsum = 0.;

    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_thickness: f32 = textureLoad(texture, vec2u(input.iuv + uniforms.blur_dir * coords), 0).r;

        var w: f32 = exp(-coords.x * coords.x / two_sigma);

        sum += sampled_thickness * w;
        wsum += w;
    }

    sum /= wsum;

    return vec4f(sum, 0., 0., 1.);
}
