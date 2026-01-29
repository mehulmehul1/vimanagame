/**
 * Jelly Shader - TSL (Three.js Shading Language) Implementation
 * ================================================================
 *
 * Bioluminescent jelly creature shader with organic pulsing animation,
 * teaching state enhancement, and fresnel-based rim lighting.
 *
 * Story: 4.6 - Jelly Shader TSL Migration
 *
 * Converted from GLSL vertex/fragment shaders to TSL nodes for WebGPU.
 * All animation and visual effects are implemented using TSL Fn() functions.
 */

import * as THREE from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
    Fn,
    positionLocal,
    normalLocal,
    uniform,
    timerLocal,
    mix,
    max,
    min,
    sin,
    cos,
    pow,
    float,
    vec3,
    vec2,
    vec4,
    normalize,
    dot,
    cameraPosition,
    positionWorld,
    modelWorldMatrix,
    smoothstep,
    abs,
    add,
    sub,
    mul,
    div,
    floor,
    If,
    select,
    transformDirection
} from 'three/tsl';

// ============================================================================
// COLOR CONSTANTS
// ============================================================================

export const BIOLUMINESCENT_COLOR = new THREE.Color(0x88ccff); // Soft cyan-blue
export const BASE_JELLY_COLOR = vec3(0.4, 0.8, 0.7);
export const INTERNAL_COLOR = vec3(0.8, 0.9, 0.85);
export const TEACHING_GLOW_COLOR = vec3(0.2, 0.1, 0.0);

// ============================================================================
// SIMPLEX NOISE (Simplified for TSL)
// ============================================================================

/**
 * Simplified 3D simplex noise function for TSL
 * Uses a simplified permutation approach for better TSL compatibility
 */
export const snoise3 = Fn(({ p }) => {
    // Simplified noise calculation using sin/cos combination
    // This provides organic variation without the full simplex noise complexity
    const x = p.x.mul(10.0);
    const y = p.y.mul(10.0);
    const z = p.z.mul(10.0);

    // Create pseudo-random noise using sin/cos
    const noise = sin(x.add(y).add(z))
        .mul(sin(x.sub(y).add(z)))
        .mul(sin(x.add(y).sub(z)))
        .mul(0.5)
        .add(0.5);

    return noise.sub(0.5); // Range: [-0.5, 0.5]
});

/**
 * Fractional Brownian Motion for more organic variation
 */
export const fbm3 = Fn(({ p, octaves }) => {
    let value = float(0);
    let amplitude = float(0.5);
    let frequency = float(1);
    let pos = p;

    // Manual loop unrolling for TSL (3 octaves)
    for (let i = 0; i < 3; i++) {
        const octaveNoise = snoise3({ p: pos.mul(frequency) });
        value = value.add(octaveNoise.mul(amplitude));
        amplitude = amplitude.mul(0.5);
        frequency = frequency.mul(2);
    }

    return value;
});

// ============================================================================
// JELLY PULSE ANIMATION
// ============================================================================

/**
 * Organic pulse animation: sin(time * rate) * 0.5 + 0.5
 * Creates a smooth pulsing effect for the jelly creature
 */
export const jellyPulse = Fn(({ time, pulseRate }) => {
    return sin(time.mul(pulseRate)).mul(0.5).add(0.5);
});

// ============================================================================
// FRESNEL EFFECT
// ============================================================================

/**
 * Fresnel effect for rim lighting
 * Uses view direction and normal to create edge glow
 */
export const jellyFresnel = Fn(({ normal, viewDir }) => {
    const cosTheta = dot(normal, viewDir);
    // Fresnel: (1 - |dot|)^2 for soft rim lighting
    return float(1).sub(cosTheta.abs()).pow(2);
});

// ============================================================================
// BELL-SHAPED ENHANCEMENT
// ============================================================================

/**
 * Bell-shaped enhancement based on position.y
 * Creates more pronounced pulse at the top of the jelly
 */
export const bellFactor = Fn(({ position }) => {
    // smoothstep(-0.5, 0.5, position.y) creates bell curve along Y axis
    return smoothstep(float(-0.5), float(0.5), position.y);
});

// ============================================================================
// JELLY DISPLACEMENT (Vertex)
// ============================================================================

/**
 * Vertex displacement with organic noise and pulse animation
 */
export const jellyDisplacement = Fn(({ time, pulseRate, isTeaching }) => {
    // Calculate pulse
    const pulse = jellyPulse({ time, pulseRate });

    // Organic noise displacement
    const noise = snoise3({
        p: positionLocal.mul(2.0).add(time.mul(0.5))
    }).mul(0.1);

    // Teaching enhancement (more movement when teaching)
    const teachingEnhancement = float(1).add(isTeaching.mul(0.5));

    // Base displacement
    const displacement = normalLocal
        .mul(pulse.mul(0.15).add(noise))
        .mul(teachingEnhancement);

    // Bell factor for top enhancement (more pulse at top)
    const bell = bellFactor({ position: positionLocal });
    const bellDisplacement = normalLocal.mul(bell.mul(pulse).mul(0.1));

    return displacement.add(bellDisplacement);
});

// ============================================================================
// JELLY EMISSIVE COLOR (Fragment)
// ============================================================================

/**
 * Calculate the emissive color with bioluminescent glow,
 * teaching enhancement, and rim lighting
 */
export const jellyEmissive = Fn(({
    time,
    pulseRate,
    isTeaching,
    teachingIntensity,
    bioColor
}) => {
    const pulse = jellyPulse({ time, pulseRate });

    // Glow intensity combines pulse and teaching state
    const glowIntensity = float(0.3)
        .add(pulse.mul(0.4))
        .add(teachingIntensity.mul(0.3));

    const glow = bioColor.mul(glowIntensity);

    // Teaching enhancement adds warm color
    const teachingGlow = TEACHING_GLOW_COLOR
        .mul(teachingIntensity)
        .mul(isTeaching);

    // Fresnel calculation for rim lighting and internal visibility
    const normalWorld = normalLocal.transformDirection(modelWorldMatrix);
    const worldPos = positionWorld;
    const viewDir = cameraPosition.sub(worldPos).normalize();
    const fresnel = jellyFresnel({ normal: normalWorld, viewDir });

    // Internal color visibility (more visible at center)
    const internalVisibility = float(1).sub(fresnel.mul(0.5));
    const internalColor = INTERNAL_COLOR.mul(internalVisibility).mul(0.3);

    // Base color contribution
    const baseContribution = BASE_JELLY_COLOR.mul(0.4);

    // Combine all color components
    const mainColor = baseContribution.add(glow).add(internalColor).add(teachingGlow);

    // Rim lighting with bioluminescent color
    const rimColor = bioColor.mul(2);
    const rimLight = rimColor.mul(fresnel).mul(0.5);

    return mainColor.add(rimLight);
});

/**
 * Calculate transparency based on fresnel and pulse
 */
export const jellyAlpha = Fn(({ time, pulseRate }) => {
    const pulse = jellyPulse({ time, pulseRate });

    // Recalculate fresnel for alpha (similar to emissive)
    const normalWorld = normalLocal.transformDirection(modelWorldMatrix);
    const worldPos = positionWorld;
    const viewDir = cameraPosition.sub(worldPos).normalize();
    const fresnel = float(1).sub(dot(normalWorld, viewDir).abs()).pow(2);

    // Alpha: more transparent at edges, varies with pulse
    return float(0.5).add(fresnel.mul(0.3)).add(pulse.mul(0.1));
});

// ============================================================================
// MAIN MATERIAL CLASS
// ============================================================================

/**
 * JellyMaterialTSL - TSL implementation of bioluminescent jelly shader
 *
 * Features:
 * - Organic pulsing animation
 * - Simplex noise displacement
 * - Fresnel-based rim lighting
 * - Teaching state enhancement
 * - Per-jelly frequency variations
 */
export class JellyMaterialTSL extends MeshStandardNodeMaterial {
    // Uniforms
    private timeUniform = uniform(0);
    private pulseRateUniform = uniform(2.0);
    private isTeachingUniform = uniform(0);
    private teachingIntensityUniform = uniform(0);
    private bioluminescentColorUniform = uniform(new THREE.Color(BIOLUMINESCENT_COLOR));
    private cameraPositionUniform = uniform(new THREE.Vector3());

    constructor() {
        super();

        this.setupTSLNodes();
        this.configureMaterial();
    }

    private setupTSLNodes(): void {
        const time = this.timeUniform;
        const pulseRate = this.pulseRateUniform;
        const isTeaching = this.isTeachingUniform;
        const teachingIntensity = this.teachingIntensityUniform;
        const bioColor = this.bioluminescentColorUniform;

        // Vertex displacement
        const displacement = jellyDisplacement({
            time,
            pulseRate,
            isTeaching
        });
        this.positionNode = displacement;

        // Emissive color with glow and rim lighting
        const emissive = jellyEmissive({
            time,
            pulseRate,
            isTeaching,
            teachingIntensity,
            bioColor
        });
        this.emissiveNode = emissive;

        // Alpha transparency
        const alpha = jellyAlpha({ time, pulseRate });
        this.transparentNode = alpha;
    }

    private configureMaterial(): void {
        // Material properties
        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.depthWrite = false;

        // Disable standard lighting features (we use emissive only)
        this.color.setHex(0x000000);
        this.roughness = 1.0;
        this.metalness = 0.0;
    }

    // ========================================================================
    // PUBLIC API
    // ========================================================================

    /**
     * Update animation time
     */
    setTime(t: number): void {
        this.timeUniform.value = t;
    }

    /**
     * Set pulse rate (frequency of pulsing animation)
     * Default: 2.0
     */
    setPulseRate(rate: number): void {
        this.pulseRateUniform.value = rate;
    }

    /**
     * Set teaching state
     * When true, enhances glow and adds warm color
     */
    setTeaching(isTeaching: boolean): void {
        this.isTeachingUniform.value = isTeaching ? 1.0 : 0.0;
    }

    /**
     * Set teaching intensity (0-1)
     * Controls how much the teaching state affects appearance
     */
    setTeachingIntensity(intensity: number): void {
        this.teachingIntensityUniform.value = Math.max(0, Math.min(1, intensity));
    }

    /**
     * Set bioluminescent color
     * Default: #88ccff (soft cyan-blue)
     */
    setColor(color: THREE.Color): void {
        this.bioluminescentColorUniform.value.copy(color);
    }

    /**
     * Update camera position for fresnel calculations
     * Should be called each frame with current camera position
     */
    setCameraPosition(position: THREE.Vector3): void {
        this.cameraPositionUniform.value.copy(position);
    }

    /**
     * Get current pulse rate
     */
    getPulseRate(): number {
        return this.pulseRateUniform.value;
    }

    /**
     * Get current teaching state
     */
    getTeaching(): boolean {
        return this.isTeachingUniform.value > 0.5;
    }

    /**
     * Get current teaching intensity
     */
    getTeachingIntensity(): number {
        return this.teachingIntensityUniform.value;
    }
}

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default JellyMaterialTSL;
