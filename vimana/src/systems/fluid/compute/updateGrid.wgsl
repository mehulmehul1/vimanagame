// updateGrid.wgsl - Grid Update with Boundary Constraints
// =====================================================
// Updates grid velocities, applies boundary constraints, and handles mouse interaction.
// Based on: https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/updateGrid.wgsl

struct Cell {
    vx: i32,
    vy: i32,
    vz: i32,
    mass: i32,
}

struct RenderUniforms {
    texel_size: vec2f,
    sphere_size: f32,
    padding0: f32,
    inv_projection_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    view_matrix: mat4x4f,
    inv_view_matrix: mat4x4f,
}

struct MouseInfo {
    screenSize: vec2f,
    padding0: vec2f,
    mouseCoord: vec2f,
    padding1: vec2f,
    mouseVel: vec2f,
    padding2: vec2f,
    mouseRadius: f32,
    padding3: f32,
}

override fixed_point_multiplier: f32;
override dt: f32;

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(4) var depthTexture: texture_2d<f32>;
@group(0) @binding(5) var<uniform> mouseInfo: MouseInfo;

fn encodeFixedPoint(floating_point: f32) -> i32 {
    return i32(floating_point * fixed_point_multiplier);
}

fn decodeFixedPoint(fixed_point: i32) -> f32 {
    return f32(fixed_point) / fixed_point_multiplier;
}

fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(tex_coord.x * 2.0 - 1.0, 1.0 - 2.0 * tex_coord.y, 0.0, 1.0);
    ndc.z = -uniforms.projection_matrix[2].z + uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;

    return eye_pos.xyz / eye_pos.w;
}

fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(depthTexture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        let uv: vec2f = mouseInfo.mouseCoord;
        let iuv = uv * mouseInfo.screenSize;
        let depth: f32 = abs(textureLoad(depthTexture, vec2u(iuv), 0).x);
        var mouseCellIndex: u32 = 1000000000; // Invalid value
        var cellSquareDistToMouse: f32 = 1e9;
        var forceDir = vec3f(0.);

        if (depth < 1e5) {
            let mouseViewPos = getViewPosFromTexCoord(uv, iuv);
            let mouseWorldPos = uniforms.inv_view_matrix * vec4f(mouseViewPos, 1.);
            let mouseCellPos: vec3i = vec3i(floor(mouseWorldPos).xyz);
            mouseCellIndex = u32(mouseCellPos.x) * u32(init_box_size.y) * u32(init_box_size.z) +
                              u32(mouseCellPos.y) * u32(init_box_size.z) +
                              u32(mouseCellPos.z);

            // Only apply force if mouse is moving
            if (dot(mouseInfo.mouseVel, mouseInfo.mouseVel) > 0.) {
                forceDir = (uniforms.inv_view_matrix * vec4f(mouseInfo.mouseVel, 0.0, 0)).xyz;
            } else {
                forceDir = vec3f(0.);
            }

            var x: f32 = f32(i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y));
            var y: f32 = f32((i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y));
            var z: f32 = f32(i32(id.x) % i32(init_box_size.z));
            let cellPos = vec3f(x, y, z);
            let diff = floor(mouseWorldPos).xyz - cellPos;
            cellSquareDistToMouse = dot(diff, diff);
        }

        let r = mouseInfo.mouseRadius;

        if (cells[id.x].mass > 0) {
            var float_v: vec3f = vec3f(
                decodeFixedPoint(cells[id.x].vx),
                decodeFixedPoint(cells[id.x].vy),
                decodeFixedPoint(cells[id.x].vz)
            );
            float_v /= decodeFixedPoint(cells[id.x].mass);

            // Apply mouse interaction force
            if (cellSquareDistToMouse < r * r) {
                let strength = (r * r - cellSquareDistToMouse) / (r * r) * 0.15;
                cells[id.x].vx = encodeFixedPoint(float_v.x + strength * forceDir.x);
                cells[id.x].vy = encodeFixedPoint(float_v.y + strength * forceDir.y);
                cells[id.x].vz = encodeFixedPoint(float_v.z + strength * forceDir.z);
            } else {
                cells[id.x].vx = encodeFixedPoint(float_v.x);
                cells[id.x].vy = encodeFixedPoint(float_v.y);
                cells[id.x].vz = encodeFixedPoint(float_v.z);
            }

            // Boundary constraints
            var x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
            var y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
            var z: i32 = i32(id.x) % i32(init_box_size.z);

            if (x < 2 || x > i32(ceil(real_box_size.x) - 3)) { cells[id.x].vx = 0; }
            if (y < 2 || y > i32(ceil(real_box_size.y) - 3)) { cells[id.x].vy = 0; }
            if (z < 2 || z > i32(ceil(real_box_size.z) - 3)) { cells[id.x].vz = 0; }
        }
    }
}
