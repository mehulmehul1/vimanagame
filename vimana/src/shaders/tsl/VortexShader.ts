/**
 * VortexShader.ts - TSL implementation of Vortex shader
 * =====================================================
 *
 * Converts the GLSL vortex shader to Three.js Shading Language (TSL).
 * Creates a glowing vortex portal that intensifies based on duet progress.
 * Features spin animation, breathing displacement, and edge glow.
 *
 * Story: 4.3 - Vortex Shader TSL Migration
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
    floor,
    mod,
} from 'three/tsl';

// ============================================================================
// COLOR CONSTANTS
// ============================================================================

const INNER_COLOR = vec3(0, 1, 1);      // Cyan: 0x00ffff
const OUTER_COLOR = vec3(0.53, 0, 1);   // Purple: 0x8800ff
const CORE_COLOR = vec3(1, 1, 1);       // White: 0xffffff

// ============================================================================
// VERTEX DISPLACEMENT FUNCTION
// ============================================================================

/**
 * Vortex vertex displacement - creates spin and breathing animation
 *
 * GLSL equivalent:
 * - spinSpeed = 1.0 + uVortexActivation * 3.0
 * - angle = uTime * spinSpeed
 * - breathe = sin(uTime * 2.0) * 0.05 * (0.5 + uVortexActivation * 0.5)
 * - rotation matrix applied to XZ plane
 * - newPosition += normal * breathe
 * - newPosition += normal * uVortexActivation * 0.1
 */
const vortexDisplacement = Fn(({ timeParam, activation }) => {
    // Calculate spin speed based on activation
    const spinSpeed = float(1).add(activation.mul(3));
    const angle = timeParam.mul(spinSpeed);

    // Breathing animation - intensifies with activation
    const breathe = sin(timeParam.mul(2)).mul(0.05).mul(float(0.5).add(activation.mul(0.5)));

    // Distance from center for swirl angle calculation
    const dist = distance(positionLocal.xz, vec2(0, 0));
    const swirlAngle = angle.mul(float(1).sub(dist.mul(0.3)));

    // Apply rotation to XZ plane using manual rotation matrix
    const cosAngle = cos(swirlAngle);
    const sinAngle = sin(swirlAngle);

    const newX = positionLocal.x.mul(cosAngle).sub(positionLocal.z.mul(sinAngle));
    const newZ = positionLocal.x.mul(sinAngle).add(positionLocal.z.mul(cosAngle));

    // Add breathing and activation displacement along normal
    const displacement = normalLocal.mul(breathe).add(normalLocal.mul(activation.mul(0.1)));

    return vec3(newX, positionLocal.y.add(displacement.y), newZ);
});

// ============================================================================
// SIMPLEX NOISE FUNCTION (simplified for TSL)
// ============================================================================

/**
 * Simplified noise function for swirl effect
 * Uses sin/cos based approximation that works in TSL
 */
const snoise = Fn(({ p }) => {
    // Simplified pseudo-noise using sin
    return sin(p.x.mul(17)).add(sin(p.y.mul(17))).add(sin(p.z.mul(17))).mul(0.5);
});

/**
 * Swirl noise for fragment shader effect
 *
 * GLSL equivalent:
 * - angle = atan(p.z, p.x)
 * - radius = length(p.xz)
 * - spiral = angle + radius * 3.0 + time
 * - return snoise(spiralPos * 2.0) * 0.5 + 0.5
 */
const swirlNoise = Fn(({ p, timeParam, activation }) => {
    const angle = atan(p.z, p.x);
    const radius = distance(p.xz, vec2(0, 0));
    const spiral = angle.add(radius.mul(3)).add(timeParam.mul(float(1).add(activation.mul(2))));

    const spiralPos = vec3(cos(spiral).mul(radius), p.y, sin(spiral).mul(radius));
    const noise = snoise({ p: spiralPos.mul(2) });

    return noise.mul(0.5).add(0.5);
});

// ============================================================================
// FRESNEL CALCULATION
// ============================================================================

/**
 * Fresnel effect for edge glow
 *
 * GLSL equivalent:
 * fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0)
 */
const vortexFresnel = Fn(({ viewDir, normal }) => {
    const dotProd = max(dot(normal.normalize(), viewDir), 0);
    return float(1).sub(dotProd).pow(2);
});

// ============================================================================
// COLOR MIXING FUNCTION
// ============================================================================

/**
 * Mix colors based on activation level
 *
 * GLSL equivalent:
 * if (uVortexActivation < 0.5) {
 *     baseColor = mix(uInnerColor, uOuterColor, uVortexActivation * 2.0);
 * } else {
 *     baseColor = mix(uOuterColor, uCoreColor, (uVortexActivation - 0.5) * 2.0);
 * }
 */
const vortexColor = Fn(({ activation }) => {
    // When activation < 0.5: mix inner → outer
    const t1 = activation.mul(2);
    const color1 = mix(INNER_COLOR, OUTER_COLOR, t1);

    // When activation >= 0.5: mix outer → core
    const t2 = activation.sub(0.5).mul(2);
    const color2 = mix(OUTER_COLOR, CORE_COLOR, t2);

    // Select based on activation threshold using select
    return select(activation.greaterThan(0.5), color2, color1);
});

// ============================================================================
// EMISSIVE COLOR FUNCTION
// ============================================================================

/**
 * Calculate final emissive color with glow and swirl
 */
const vortexEmissive = Fn(({ activation, timeParam, localPosition, cameraPos }) => {
    // Get world position
    const worldPos = positionWorld;

    // View direction from camera to world position
    const viewDir = cameraPos.sub(worldPos).normalize();
    const normal = normalWorld;

    // Calculate fresnel for edge glow
    const fresnel = vortexFresnel({ viewDir, normal });

    // Calculate swirl noise
    const swirl = swirlNoise({
        p: localPosition,
        timeParam: timeParam,
        activation: activation
    });

    // Get base color based on activation
    const baseColor = vortexColor({ activation });

    // Add swirl noise to base color (need to use a temp variable)
    const baseWithSwirl = vec3(baseColor).add(swirl.mul(0.2));

    // Edge intensity increases with activation
    const edgeIntensity = float(0.5).add(activation.mul(1.5));

    // Create glow from fresnel and base color
    const glow = baseWithSwirl.mul(fresnel).mul(edgeIntensity);

    // Final color: base + glow
    const finalColor = baseWithSwirl.mul(0.6).add(glow);

    // Emissive intensity increases with activation
    const emissive = float(0.2).add(activation.mul(2.8));
    finalColor.mulAssign(emissive);

    // Core white blend at high activation (> 0.7)
    const coreIntensity = activation.sub(0.7).div(0.3);
    const coreMix = coreIntensity.max(0).min(1);

    const finalWithCore = mix(finalColor, CORE_COLOR, coreMix.mul(0.5));

    return finalWithCore;
});

// ============================================================================
// MATERIAL CLASS
// ============================================================================

/**
 * VortexMaterialTSL - TSL-based Vortex Material
 *
 * Extends MeshStandardNodeMaterial to provide TSL shader functionality
 * while maintaining API compatibility with the original GLSL VortexMaterial.
 */
export class VortexMaterialTSL extends MeshStandardNodeMaterial {
    private timeUniform = uniform(0);
    private activationUniform = uniform(0);
    private duetProgressUniform = uniform(0);

    // Camera position uniform for fresnel calculations
    private cameraPositionUniform = uniform(new THREE.Vector3(0, 0, 5));

    constructor() {
        super();

        // Set up material properties for TSL
        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.depthWrite = false;
        this.opacity = 0.9;

        // Set up time node from global time
        const timeNode = time;

        // Apply vertex displacement
        this.positionNode = vortexDisplacement({
            timeParam: this.timeUniform,
            activation: this.activationUniform
        });

        // Apply emissive color
        this.emissiveNode = vortexEmissive({
            activation: this.activationUniform,
            timeParam: this.timeUniform,
            localPosition: positionLocal,
            cameraPos: this.cameraPositionUniform
        });

        // Enable transmission for glass-like effect
        this.transmission = 0;
        this.roughness = 0.5;
        this.metalness = 0.1;
    }

    // ========================================================================
    // PUBLIC API (maintains compatibility with original VortexMaterial)
    // ========================================================================

    /**
     * Update time for animation
     */
    public setTime(time: number): void {
        this.timeUniform.value = time;
    }

    /**
     * Update vortex activation level (0-1)
     * Controls intensity, spin speed, and color transition
     */
    public setActivation(activation: number): void {
        this.activationUniform.value = Math.max(0, Math.min(1, activation));
    }

    /**
     * Update duet progress (0-1)
     * Maintained for API compatibility
     */
    public setDuetProgress(progress: number): void {
        this.duetProgressUniform.value = Math.max(0, Math.min(1, progress));
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        this.cameraPositionUniform.value.copy(position);
    }

    /**
     * Get current activation level
     */
    public getActivation(): number {
        return this.activationUniform.value;
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
    vortexDisplacement,
    swirlNoise,
    vortexFresnel,
    vortexColor,
    vortexEmissive,
    INNER_COLOR,
    OUTER_COLOR,
    CORE_COLOR,
};

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default VortexMaterialTSL;
