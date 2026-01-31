// spawnParticles.wgsl - Spawn Particles in Dam-Break Pattern
// ==========================================================
// Spawns particles in a dam-break pattern for initial fluid setup.
// Based on: https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/spawnParticles.wgsl

struct Particle {
    position: vec3f,
    v: vec3f,
    C: mat3x3f,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> init_box_size: vec3f;
@group(0) @binding(2) var<uniform> numSpawnParticles: u32;

// Simple hash function for deterministic randomness
fn hash(n: u32) -> u32 {
    var h = n;
    h = h * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) * 277803737u);
    return h;
}

fn rand(n: u32) -> f32 {
    return f32(hash(n)) / 4294967296.0;
}

@compute @workgroup_size(64)
fn spawnParticles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;

    // Only spawn requested number of particles
    if (id >= numSpawnParticles) {
        return;
    }

    // Only spawn within the max particles array size
    if (id >= arrayLength(&particles)) {
        return;
    }

    // Dam-break pattern: spawn in a block from one corner
    let spacing = 0.55;
    let particlesPerRow = u32(ceil(52.0 / spacing)); // Based on init box size
    let particlesPerLayer = particlesPerRow * particlesPerRow;

    let layer = id / particlesPerLayer;
    let row = (id % particlesPerLayer) / particlesPerRow;
    let col = id % particlesPerRow;

    // Position within the dam-break block
    var pos: vec3f = vec3f(
        3.0 + f32(col) * spacing,
        3.0 + f32(row) * spacing,
        3.0 + f32(layer) * spacing
    );

    // Apply jitter for more natural distribution
    pos += vec3f(rand(id) * 2.0, rand(id + 1u) * 2.0, rand(id + 2u) * 2.0);

    // Clamp to box bounds
    pos.x = clamp(pos.x, 1.0, 49.0);
    pos.y = clamp(pos.y, 1.0, 41.0);  // 80% of box height
    pos.z = clamp(pos.z, 1.0, 26.0);  // Half of box depth

    // Initialize particle
    particles[id].position = pos;
    particles[id].v = vec3f(0.0, 0.0, 0.0);
    particles[id].C = mat3x3f(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0)
    );
}
