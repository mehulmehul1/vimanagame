/**
 * ShellShader.ts - TSL implementation of Shell SDF shader
 * =========================================================
 *
 * Converts the GLSL procedural SDF shell shader to Three.js Shading Language (TSL).
 * Creates a beautiful iridescent nautilus shell with spiral pattern, dissolve effect,
 * and 3-second appear animation with easing.
 *
 * Features:
 * - Nautilus spiral SDF pattern on surface
 * - 5-color iridescence effect based on view angle
 * - Simplex noise dissolve effect
 * - 3-second appear animation with smoothstep easing
 * - Bobbing idle animation
 * - Fresnel-based edge glow
 *
 * Story: 4.5 - Shell SDF TSL Migration
 * =====================================================================
 */

import * as THREE from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
    Fn,
    positionLocal,
    normalLocal,
    uniform,
    mix,
    distance,
    max,
    min,
    sin,
    cos,
    pow,
    float,
    vec3,
    vec2,
    normalize,
    dot,
    cameraPosition,
    positionWorld,
    normalWorld,
    time,
    If,
    abs,
    atan,
    length,
    add,
    sub,
    mul,
    div,
    floor,
    fract,
    smoothstep,
    transformDirection,
    modelWorldMatrix,
} from 'three/tsl';

// ============================================================================
// COLOR CONSTANTS - 5 Iridescent Colors
// ============================================================================

const IRIDESCENT_COLOR_0 = vec3(1.0, 0.8, 0.6);  // Peach
const IRIDESCENT_COLOR_1 = vec3(0.8, 0.6, 1.0);  // Lavender
const IRIDESCENT_COLOR_2 = vec3(0.6, 0.9, 1.0);  // Sky blue
const IRIDESCENT_COLOR_3 = vec3(1.0, 0.7, 0.5);  // Orange
const IRIDESCENT_COLOR_4 = vec3(0.7, 1.0, 0.8);  // Mint

const BASE_SHELL_COLOR = vec3(0.95, 0.9, 0.85);  // Cream base

// ============================================================================
// SIMPLEX NOISE 3D (for dissolve effect)
// ============================================================================

/**
 * 3D Simplex noise function for TSL
 * Based on Perlin's improved noise algorithm
 * GLSL equivalent: float snoise(vec3 v) from original shader
 */
const snoise3 = Fn(({ p }) => {
    // Simplex noise constants
    const C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const D = vec4(0.0, 0.5, 1.0, 2.0);

    // Skew the input space to determine which simplex cell we're in
    const s = p.mul(3).add(p.dot(p).mul(p.dot(p))).mul(C.x).floor();
    const i = p.add(s).add(s.dot(s).mul(C.y)).floor();

    const x0 = p.sub(i).add(i.dot(C.xxx));

    // Gradients
    const g = x0.yzx().step(x0);
    const l = float(1).sub(g);
    const i1 = min(g.xyz(), l.zxy());
    const i2 = max(g.xyz(), l.zxy());

    const x1 = x0.sub(i1).add(C.xxx);
    const x2 = x0.sub(i2).add(C.yyy);
    const x3 = x0.sub(D.yyy);

    // Permutations (simplified for TSL)
    const iMod = i.mul(34.0).add(1.0).mul(i).floor();
    const p0Hash = sin(iMod.x.mul(12.9898)).mul(43758.5453).fract();
    const p1Hash = sin(iMod.y.mul(12.9898)).mul(43758.5453).fract();
    const p2Hash = sin(iMod.z.mul(12.9898)).mul(43758.5453).fract();

    // Calculate noise contribution from each corner
    const n0 = x0.dot(x0).mul(-0.5).exp().mul(p0Hash.mul(2).sub(1));
    const n1 = x1.dot(x1).mul(-0.5).exp().mul(p1Hash.mul(2).sub(1));
    const n2 = x2.dot(x2).mul(-0.5).exp().mul(p2Hash.mul(2).sub(1));
    const n3 = x3.dot(x3).mul(-0.5).exp().mul(sin(iMod.z.mul(17.0)).mul(2).sub(1));

    // Sum contributions and scale
    const noise = n0.add(n1).add(n2).add(n3).mul(42.0);

    return noise.mul(0.5).add(0.5);
});

// ============================================================================
// IRIDESCENCE EFFECT
// ============================================================================

/**
 * Calculate iridescent color based on view angle and time
 * Cycles through 5 colors using smooth interpolation
 *
 * GLSL equivalent:
 * float viewAngle = dot(viewDir, normal);
 * float t = sin(viewAngle * 5.0 + time) * 0.5 + 0.5;
 * return mix(colors[index], colors[index + 1], localT);
 */
const shellIridescence = Fn(({ viewDir, normal, timeParam }) => {
    // Calculate view angle
    const viewAngle = dot(viewDir, normal);
    const t = sin(viewAngle.mul(5).add(timeParam)).mul(0.5).add(0.5);

    // Map t to color gradient (0-1 across 4 color transitions)
    const t4 = t.mul(4);
    const index = floor(t4);
    const localT = fract(t4);

    // Manual color interpolation (5 colors = 4 transitions)
    const c01 = mix(IRIDESCENT_COLOR_0, IRIDESCENT_COLOR_1, localT);
    const c12 = mix(IRIDESCENT_COLOR_1, IRIDESCENT_COLOR_2, localT);
    const c23 = mix(IRIDESCENT_COLOR_2, IRIDESCENT_COLOR_3, localT);
    const c34 = mix(IRIDESCENT_COLOR_3, IRIDESCENT_COLOR_4, localT);

    // Select based on index using If statements
    const result1 = If(index.lessThan(1), c01, c12);
    const result2 = If(index.lessThan(2), result1, c23);
    return If(index.lessThan(3), result2, c34);
});

// ============================================================================
// SPIRAL PATTERN
// ============================================================================

/**
 * Create nautilus spiral pattern on shell surface
 * Uses angle and radius from Y-axis to create spiral
 *
 * GLSL equivalent:
 * float angle = atan(vLocalPosition.z, vLocalPosition.x);
 * float radius = length(vLocalPosition.xz);
 * float spiral = sin(angle * 3.0 + radius * 5.0) * 0.5 + 0.5;
 */
const shellSpiralPattern = Fn(({ position }) => {
    const angle = atan(position.z, position.x);
    const radius = length(position.xz);

    // Create spiral pattern
    const spiral = sin(angle.mul(3).add(radius.mul(5))).mul(0.5).add(0.5);

    return spiral;
});

// ============================================================================
// FRESNEL EFFECT
// ============================================================================

/**
 * Calculate Fresnel edge glow for the shell
 * Uses view direction and surface normal
 *
 * GLSL equivalent:
 * float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);
 */
const shellFresnel = Fn(({ normal, viewDir }) => {
    const cosTheta = dot(normal, viewDir);
    return pow(float(1).sub(cosTheta.abs()), float(3));
});

// ============================================================================
// DISSOLVE EFFECT
// ============================================================================

/**
 * Create dissolve effect using simplex noise
 * Pixels are discarded based on dissolve threshold
 *
 * GLSL equivalent:
 * float dissolve = snoise(vLocalPosition * 3.0 + uTime) * 0.5 + 0.5;
 * float alpha = smoothstep(uDissolveAmount - 0.1, uDissolveAmount + 0.1, dissolve);
 */
const shellDissolve = Fn(({ position, timeParam, dissolveAmount }) => {
    // Generate noise for dissolve pattern
    const noise = snoise3({ p: position.mul(3).add(timeParam) });
    const noiseValue = noise.mul(0.5).add(0.5);

    // Apply dissolve threshold with smooth edges
    const alpha = smoothstep(
        dissolveAmount.sub(0.1),
        dissolveAmount.add(0.1),
        noiseValue
    );

    return alpha;
});

// ============================================================================
// APPEAR ANIMATION WITH EASING
// ============================================================================

/**
 * Calculate appear progress with smoothstep easing
 * Includes bobbing motion for idle animation
 *
 * GLSL equivalent:
 * float eased = smoothstep(0.0, 1.0, appearProgress);
 * float bob = sin(time * 2.0) * 0.05;
 */
const shellAppear = Fn(({ appearProgress, timeParam }) => {
    // Smoothstep easing for appear animation
    const easedProgress = smoothstep(0, 1, appearProgress);

    // Bobbing motion: sine wave on Y axis
    const bobOffset = sin(timeParam.mul(2)).mul(0.05);

    // Return scale with bobbing (bob only applies during idle/appearing)
    return vec3(
        easedProgress,  // Scale X
        easedProgress,  // Scale Y
        easedProgress   // Scale Z
    );
});

// ============================================================================
// VERTEX DISPLACEMENT (for bobbing animation)
// ============================================================================

/**
 * Apply bobbing animation to vertex position
 * Only affects Y position with sine wave
 */
const shellDisplacement = Fn(({ position, timeParam, appearProgress }) => {
    // Calculate bobbing offset
    const bobOffset = sin(timeParam.mul(2)).mul(0.05);

    // Apply scale from appear progress
    const scale = smoothstep(0, 1, appearProgress);

    return vec3(
        position.x.mul(scale),
        position.y.mul(scale).add(bobOffset.mul(scale)),
        position.z.mul(scale)
    );
});

// ============================================================================
// MATERIAL CLASS
// ============================================================================

/**
 * ShellMaterialTSL - TSL-based Shell Material
 *
 * Extends MeshStandardNodeMaterial to provide TSL shader functionality
 * while maintaining API compatibility with the original GLSL shell shader.
 *
 * Features:
 * - Iridescence with 5 colors based on view angle
 * - Spiral pattern on shell surface
 * - Fresnel-based edge glow
 * - Dissolve effect with simplex noise
 * - 3-second appear animation with easing
 * - Bobbing idle animation
 */
export class ShellMaterialTSL extends MeshStandardNodeMaterial {
    private timeUniform = uniform(0);
    private appearProgressUniform = uniform(0);
    private dissolveAmountUniform = uniform(1.0);
    private cameraPositionUniform = uniform(new THREE.Vector3(0, 0, 5));

    constructor() {
        super();

        // Set up material properties
        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.depthWrite = false;
        this.opacity = 1.0;

        // Get world position and normal
        const worldPos = positionWorld;
        const normalWorldNode = normalLocal.transformDirection(modelWorldMatrix);

        // View direction from camera to surface
        const viewDir = this.cameraPositionUniform.sub(worldPos).normalize();

        // Calculate shader effects
        const iridescence = shellIridescence({
            viewDir,
            normal: normalWorldNode,
            timeParam: this.timeUniform.mul(0.5)  // Slower iridescence animation
        });

        const spiral = shellSpiralPattern({ position: positionLocal });
        const fresnel = shellFresnel({
            normal: normalWorldNode,
            viewDir
        });

        const dissolve = shellDissolve({
            position: positionLocal,
            timeParam: this.timeUniform,
            dissolveAmount: this.dissolveAmountUniform
        });

        // Base color with spiral pattern
        // Mix between cream base and iridescence based on spiral pattern
        const spiralColor = mix(BASE_SHELL_COLOR, iridescence, spiral.mul(0.6).add(fresnel.mul(0.4)));

        // Final color with fresnel glow
        const finalColor = spiralColor.add(iridescence.mul(fresnel.mul(0.3)));

        // Set color node
        this.colorNode = finalColor;

        // Set alpha with dissolve and appear progress
        const alphaNode = dissolve.mul(this.appearProgressUniform);
        this.opacityNode = alphaNode;

        // Apply vertex displacement for bobbing animation
        this.positionNode = shellDisplacement({
            position: positionLocal,
            timeParam: this.timeUniform,
            appearProgress: this.appearProgressUniform
        });

        // Material properties
        this.roughness = 0.3;
        this.metalness = 0.1;
    }

    // ========================================================================
    // PUBLIC API (maintains compatibility with original ShellMaterial)
    // ========================================================================

    /**
     * Update time for animation (iridescence, bobbing)
     * @param t - Current time in seconds
     */
    public setTime(t: number): void {
        this.timeUniform.value = t;
    }

    /**
     * Update appear progress (0-1)
     * Controls the 3-second appear animation with easing
     * @param progress - Progress value from 0 to 1
     */
    public setAppearProgress(progress: number): void {
        this.appearProgressUniform.value = Math.max(0, Math.min(1, progress));
    }

    /**
     * Update dissolve amount (0-1)
     * Controls the dissolve threshold (1 = fully dissolved, 0 = fully visible)
     * @param amount - Dissolve amount from 0 (visible) to 1 (invisible)
     */
    public setDissolveAmount(amount: number): void {
        this.dissolveAmountUniform.value = Math.max(0, Math.min(1, amount));
    }

    /**
     * Update camera position for fresnel calculations
     * @param position - Current camera position
     */
    public setCameraPosition(position: THREE.Vector3): void {
        this.cameraPositionUniform.value.copy(position);
    }

    /**
     * Get current appear progress
     */
    public getAppearProgress(): number {
        return this.appearProgressUniform.value;
    }

    /**
     * Get current dissolve amount
     */
    public getDissolveAmount(): number {
        return this.dissolveAmountUniform.value;
    }

    /**
     * Cleanup method for memory management
     */
    public destroy(): void {
        this.dispose();
    }
}

// ============================================================================
// EXPORT INDIVIDUAL SHADER FUNCTIONS FOR REUSE
// ============================================================================

export {
    snoise3,
    shellIridescence,
    shellSpiralPattern,
    shellFresnel,
    shellDissolve,
    shellAppear,
    shellDisplacement,
    IRIDESCENT_COLOR_0,
    IRIDESCENT_COLOR_1,
    IRIDESCENT_COLOR_2,
    IRIDESCENT_COLOR_3,
    IRIDESCENT_COLOR_4,
    BASE_SHELL_COLOR,
};

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default ShellMaterialTSL;
