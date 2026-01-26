/**
 * types.ts - MLS-MPM Fluid Simulation Types
 * =====================================================
 *
 * Type definitions for the WaterBall fluid simulation system.
 * Based on: https://github.com/matsuoka-601/WaterBall
 */

// Particle struct (80 bytes per particle)
// WGSL struct definition matches the compute shader layout
export interface Particle {
    position: [number, number, number];  // 12 bytes - world position (offset 0)
    padding0?: number;                   // 4 bytes - padding to 16-byte boundary
    v: [number, number, number];         // 12 bytes - velocity (offset 16)
    padding1?: number;                   // 4 bytes - padding
    C: number[];                         // 36 bytes - affine momentum matrix (3x3) (offset 32)
    density?: number;                    // 4 bytes - particle density (offset 68)
    padding2?: number;                   // 4 bytes - padding (offset 72)
}

// Cell struct (16 bytes per cell)
export interface Cell {
    vx: number;  // velocity X (fixed point i32)
    vy: number;  // velocity Y (fixed point i32)
    vz: number;  // velocity Z (fixed point i32)
    mass: number; // cell mass (fixed point i32)
}

// Position-velocity-density struct for rendering
export interface PosVel {
    position: [number, number, number];
    v: [number, number, number];
    density: number;
}

// Fluid simulation configuration constants
export const FLUID_CONFIG = {
    // Simulation
    stiffness: 3.0,              // Material stiffness
    restDensity: 4.0,            // Rest density
    dynamicViscosity: 0.1,       // Viscosity
    dt: 0.016,                   // Time step (~60fps)
    fixedPointMultiplier: 1e7,    // For velocity encoding

    // Grid
    gridResolution: 80,
    cellSize: 1.0,
    cellStructSize: 16,          // bytes per cell

    // Particles
    maxParticles: 10000,
    spawnRate: 100,              // Particles per spawn
    particleSpacing: 0.55,
    particleStructSize: 80,      // bytes per particle

    // Sphere constraint (for tunnel effect)
    sphereRadius: 15.0,          // Will be dynamic
    sphereStiffness: 3.0,
    sphereAttraction: 0.1,
} as const;

// Mouse/interaction info uniform buffer layout
export interface MouseInfo {
    screenSize: [number, number];    // Canvas dimensions
    padding0: [number, number];      // Padding to 16 bytes
    mouseCoord: [number, number];    // Mouse position in screen coords
    padding1: [number, number];      // Padding
    mouseVel: [number, number];      // Mouse velocity
    padding2: [number, number];      // Padding
    mouseRadius: number;             // Interaction radius
    padding3: number;                // Padding
}

// Render uniforms for depth-based interaction
export interface RenderUniforms {
    texelSize: [number, number];     // 1/textureSize
    sphereSize: number;              // Particle render size
    padding0: number;                // Padding
    invProjectionMatrix: number[];   // 4x4 matrix (64 bytes)
    projectionMatrix: number[];      // 4x4 matrix (64 bytes)
    viewMatrix: number[];            // 4x4 matrix (64 bytes)
    invViewMatrix: number[];         // 4x4 matrix (64 bytes)
}

// Simulation state
export interface SimulationState {
    numParticles: number;
    boxWidthRatio: number;           // For plane→sphere animation
    isTunnelOpen: boolean;
}

export type vec3 = [number, number, number];
export type vec2 = [number, number, number];
