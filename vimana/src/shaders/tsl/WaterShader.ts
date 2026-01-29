/**
 * WaterShader.ts - TSL implementation of Water shader
 * =====================================================
 *
 * Converts the GLSL water shader to Three.js Shading Language (TSL).
 * Creates a bioluminescent water surface that responds to 6 harp strings,
 * jelly creature ripples, and sphere constraint animation for duet progress.
 * Features Fresnel effects, depth-based absorption, caustics, and transmission.
 *
 * Story: 4.4 - Water Material TSL Migration
 * =====================================================================
 */

import * as THREE from 'three';
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import {
    Fn,
    positionLocal,
    normalLocal,
    uniform,
    uv,
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
    exp,
    smoothstep,
    add,
    sub,
    mul,
    div,
    floor,
    fract,
} from 'three/tsl';

// ============================================================================
// COLOR CONSTANTS
// ============================================================================

const BIOLUMINESCENT_COLOR = vec3(0, 1, 0.53);     // Cyan-green: 0x00ff88
const DEEP_COLOR = vec3(0, 0.08, 0.16);            // Dark blue: depth color
const SHALLOW_COLOR = vec3(0, 0.24, 0.32);         // Bright teal: shallow color
const DIFFUSE_COLOR = vec3(0, 0.7375, 0.95);       // Cyan-blue tint for transmittance

// ============================================================================
// NOISE FUNCTIONS
// ============================================================================

/**
 * Simple pseudo-random noise for TSL
 * GLSL equivalent: fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453)
 */
const simplexNoise = Fn(({ p }) => {
    return sin(p.x.mul(12.9898).add(p.y.mul(78.233))).mul(43758.5453).fract();
});

/**
 * Perlin noise for wave generation
 * GLSL equivalent: mix(mix(a, b, f.x), mix(c, d, f.x), f.y) with smooth interpolation
 */
const perlin = Fn(({ p }) => {
    const i = p.floor();
    const f = p.fract();
    const smoothF = f.mul(f).mul(float(3).sub(f.mul(2)));

    const a = simplexNoise({ p: i });
    const b = simplexNoise({ p: i.add(vec2(1, 0)) });
    const c = simplexNoise({ p: i.add(vec2(0, 1)) });
    const d = simplexNoise({ p: i.add(vec2(1, 1)) });

    const ab = mix(a, b, smoothF.x);
    const cd = mix(c, d, smoothF.x);
    return mix(ab, cd, smoothF.y);
});

/**
 * Fractal Brownian Motion for detailed waves
 * GLSL equivalent: 4 octaves of perlin noise
 */
const fbm = Fn(({ coord, octaves }) => {
    // Manual loop unrolling for TSL (4 octaves)
    const octave1 = perlin({ p: coord.mul(1).add(time.mul(0.1)) }).mul(0.5);
    const octave2 = perlin({ p: coord.mul(2).add(time.mul(0.05)) }).mul(0.25);
    const octave3 = perlin({ p: coord.mul(4).add(time.mul(0.025)) }).mul(0.125);
    const octave4 = perlin({ p: coord.mul(8).add(time.mul(0.0125)) }).mul(0.0625);

    return octave1.add(octave2).add(octave3).add(octave4);
});

// ============================================================================
// STRING WAVE FUNCTIONS
// ============================================================================

/**
 * Calculate wave influence from a single harp string
 * GLSL equivalent:
 * float stringWave = sin(distTo * 3.0 - uTime * 2.0 + uHarpFrequencies[i]) * 0.05;
 * stringWave *= smoothstep(3.0, 0.0, distTo);
 */
const calculateStringWave = Fn(({ position, stringOrigin, frequency, velocity, timeParam }) => {
    const distTo = distance(position.xz, stringOrigin);
    const stringWave = sin(distTo.mul(3).sub(timeParam.mul(2)).add(frequency)).mul(0.05);
    const distanceFalloff = smoothstep(float(3), float(0), distTo);
    return stringWave.mul(distanceFalloff).mul(velocity).mul(0.5);
});

/**
 * Calculate string resonance from all 6 harp strings
 * Each string creates a wave that decays with distance
 */
const stringResonanceEffect = Fn(({ position, frequencies, velocities, timeParam }) => {
    // Process all 6 strings (manual loop unrolling for TSL)
    const string0 = calculateStringWave({
        position,
        stringOrigin: vec2(-5, 0),
        frequency: frequencies.element(0),
        velocity: velocities.element(0),
        timeParam
    });

    const string1 = calculateStringWave({
        position,
        stringOrigin: vec2(-3, 0),
        frequency: frequencies.element(1),
        velocity: velocities.element(1),
        timeParam
    });

    const string2 = calculateStringWave({
        position,
        stringOrigin: vec2(-1, 0),
        frequency: frequencies.element(2),
        velocity: velocities.element(2),
        timeParam
    });

    const string3 = calculateStringWave({
        position,
        stringOrigin: vec2(1, 0),
        frequency: frequencies.element(3),
        velocity: velocities.element(3),
        timeParam
    });

    const string4 = calculateStringWave({
        position,
        stringOrigin: vec2(3, 0),
        frequency: frequencies.element(4),
        velocity: velocities.element(4),
        timeParam
    });

    const string5 = calculateStringWave({
        position,
        stringOrigin: vec2(5, 0),
        frequency: frequencies.element(5),
        velocity: velocities.element(5),
        timeParam
    });

    return string0.add(string1).add(string2).add(string3).add(string4).add(string5);
});

/**
 * Calculate bioluminescent glow from string vibrations
 */
const stringBioluminescence = Fn(({ position, frequencies, velocities }) => {
    let totalGlow = float(0);

    // String 0
    const dist0 = distance(position.xz, vec2(-5, 0));
    const glow0 = float(1).sub(dist0.div(4)).max(0).mul(velocities.element(0));
    totalGlow = totalGlow.add(glow0);

    // String 1
    const dist1 = distance(position.xz, vec2(-3, 0));
    const glow1 = float(1).sub(dist1.div(4)).max(0).mul(velocities.element(1));
    totalGlow = totalGlow.add(glow1);

    // String 2
    const dist2 = distance(position.xz, vec2(-1, 0));
    const glow2 = float(1).sub(dist2.div(4)).max(0).mul(velocities.element(2));
    totalGlow = totalGlow.add(glow2);

    // String 3
    const dist3 = distance(position.xz, vec2(1, 0));
    const glow3 = float(1).sub(dist3.div(4)).max(0).mul(velocities.element(3));
    totalGlow = totalGlow.add(glow3);

    // String 4
    const dist4 = distance(position.xz, vec2(3, 0));
    const glow4 = float(1).sub(dist4.div(4)).max(0).mul(velocities.element(4));
    totalGlow = totalGlow.add(glow4);

    // String 5
    const dist5 = distance(position.xz, vec2(5, 0));
    const glow5 = float(1).sub(dist5.div(4)).max(0).mul(velocities.element(5));
    totalGlow = totalGlow.add(glow5);

    return BIOLUMINESCENT_COLOR.mul(totalGlow.mul(0.5));
});

// ============================================================================
// JELLY RIPPLE FUNCTIONS
// ============================================================================

/**
 * Calculate ripple effect from a single jelly creature
 * GLSL equivalent:
 * float rippleInfluence = exp(-dist * 0.8);
 * float ripple = sin(dist * 8.0 - uTime * 5.0) * uJellyVelocities[i];
 */
const calculateJellyRipple = Fn(({ position, jellyPos, velocity, timeParam }) => {
    const jellyXZ = vec2(jellyPos.x, jellyPos.z);
    const dist = distance(position.xz, jellyXZ);
    const rippleInfluence = exp(dist.mul(-0.8));
    const ripple = sin(dist.mul(8).sub(timeParam.mul(5))).mul(velocity);
    return ripple.mul(rippleInfluence);
});

/**
 * Calculate total ripple effect from all 6 jelly creatures
 */
const jellyRippleEffect = Fn(({ position, jellyPositions, jellyVelocities, jellyActive, timeParam }) => {
    let totalRipple = float(0);

    // Jelly 0
    const jelly0Pos = vec3(
        jellyPositions.element(0),
        jellyPositions.element(1),
        jellyPositions.element(2)
    );
    const ripple0 = calculateJellyRipple({
        position,
        jellyPos: jelly0Pos,
        velocity: jellyVelocities.element(0),
        timeParam
    });
    totalRipple = totalRlow.add(If(jellyActive.element(0).greaterThan(0.5), ripple0, float(0)));

    // Jelly 1
    const jelly1Pos = vec3(
        jellyPositions.element(3),
        jellyPositions.element(4),
        jellyPositions.element(5)
    );
    const ripple1 = calculateJellyRipple({
        position,
        jellyPos: jelly1Pos,
        velocity: jellyVelocities.element(1),
        timeParam
    });
    totalRipple = totalRlow.add(If(jellyActive.element(1).greaterThan(0.5), ripple1, float(0)));

    // Jelly 2
    const jelly2Pos = vec3(
        jellyPositions.element(6),
        jellyPositions.element(7),
        jellyPositions.element(8)
    );
    const ripple2 = calculateJellyRipple({
        position,
        jellyPos: jelly2Pos,
        velocity: jellyVelocities.element(2),
        timeParam
    });
    totalRipple = totalRlow.add(If(jellyActive.element(2).greaterThan(0.5), ripple2, float(0)));

    // Jelly 3
    const jelly3Pos = vec3(
        jellyPositions.element(9),
        jellyPositions.element(10),
        jellyPositions.element(11)
    );
    const ripple3 = calculateJellyRipple({
        position,
        jellyPos: jelly3Pos,
        velocity: jellyVelocities.element(3),
        timeParam
    });
    totalRipple = totalRlow.add(If(jellyActive.element(3).greaterThan(0.5), ripple3, float(0)));

    // Jelly 4
    const jelly4Pos = vec3(
        jellyPositions.element(12),
        jellyPositions.element(13),
        jellyPositions.element(14)
    );
    const ripple4 = calculateJellyRipple({
        position,
        jellyPos: jelly4Pos,
        velocity: jellyVelocities.element(4),
        timeParam
    });
    totalRipple = totalRlow.add(If(jellyActive.element(4).greaterThan(0.5), ripple4, float(0)));

    // Jelly 5
    const jelly5Pos = vec3(
        jellyPositions.element(15),
        jellyPositions.element(16),
        jellyPositions.element(17)
    );
    const ripple5 = calculateJellyRipple({
        position,
        jellyPos: jelly5Pos,
        velocity: jellyVelocities.element(5),
        timeParam
    });
    totalRipple = totalRlow.add(If(jellyActive.element(5).greaterThan(0.5), ripple5, float(0)));

    return totalRipple.mul(0.15);
});

// ============================================================================
// SPHERE CONSTRAINT FUNCTION (for duet progress animation)
// ============================================================================

/**
 * Transform flat plane to sphere shape based on duet progress
 * GLSL equivalent: Sphere constraint with hollow center tunnel
 */
const sphereConstraint = Fn(({ position, duetProgress, radius, center }) => {
    const centeredXZ = position.xz.sub(center.xz);
    const distFromCenter = distance(centeredXZ, vec2(0, 0));

    // Pull strength increases with duet progress
    const pullStrength = smoothstep(float(0.3), float(1), duetProgress);

    // Calculate sphere position
    let spherePos = position;
    const dir = centeredXZ.normalize();
    const sphereX = center.x.add(dir.x.mul(radius));
    const sphereZ = center.z.add(dir.y.mul(radius));
    spherePos = vec3(sphereX, position.y, sphereZ);

    // Hollow center tunnel effect
    const hollowRadius = radius.mul(float(0.2).add(float(0.5).mul(float(1).sub(duetProgress))));
    const hollowFade = smoothstep(hollowRadius.mul(0.5), hollowRadius, distFromCenter);

    // Dissolve vertices in hollow center
    const dissolvedY = position.y.sub(float(10).mul(float(1).sub(hollowFade)).mul(duetProgress));

    const finalY = If(
        duetProgress.greaterThan(0.5).and(distFromCenter.lessThan(hollowRadius)),
        dissolvedY,
        position.y
    );

    // Edge lift for bowl/tunnel shape
    const edgeLift = smoothstep(radius.mul(0.5), radius.mul(2), distFromCenter);
    const liftedY = finalY.add(edgeLift.mul(duetProgress).mul(2));

    // Blend flat position to sphere position
    const blendedX = mix(position.x, spherePos.x, pullStrength);
    const blendedZ = mix(position.z, spherePos.z, pullStrength);

    return vec3(blendedX, liftedY, blendedZ);
});

// ============================================================================
// VERTEX DISPLACEMENT FUNCTION
// ============================================================================

/**
 * Water vertex displacement - combines waves, strings, and jellies
 * GLSL equivalent: Base FBM + string resonance + jelly ripples
 */
const waterDisplacement = Fn(({ timeParam, frequencies, velocities, duetProgress, sphereRadius, sphereCenter, jellyPositions, jellyVelocities, jellyActive }) => {
    // Start with original position
    let pos = positionLocal;

    // Apply sphere constraint for duet progress animation
    pos = sphereConstraint({
        position: pos,
        duetProgress,
        radius: sphereRadius,
        center: sphereCenter
    });

    // Base wave animation
    const waveCoord = pos.xz.mul(0.5);
    const baseWave = fbm({ coord: waveCoord.add(timeParam.mul(0.1)), octaves: 3 }).mul(0.15);

    // String resonance from harp
    const stringResonance = stringResonanceEffect({
        position: pos,
        frequencies,
        velocities,
        timeParam
    });

    // Jelly creature ripples
    const jellyRipple = jellyRippleEffect({
        position: pos,
        jellyPositions,
        jellyVelocities,
        jellyActive,
        timeParam
    });

    // Combine all wave effects
    const totalWave = baseWave.add(stringResonance).add(jellyRipple);
    const scaledWave = totalWave.mul(float(1).add(duetProgress.mul(0.5)));

    // Apply displacement to Y
    return pos.add(vec3(0, scaledWave, 0));
});

// ============================================================================
// FRESNEL CALCULATION (Schlick's approximation)
// ============================================================================

/**
 * Fresnel effect for edge glow and transparency
 * GLSL equivalent: F0 + (1-F0) * pow(1-cosTheta, 5)
 */
const waterFresnel = Fn(({ normal, viewDir, f0 }) => {
    const cosTheta = max(dot(normal, viewDir.negate().normalize()), 0);
    const x = float(1).sub(cosTheta);
    return f0.add(float(1).sub(f0).mul(x.pow(5)));
});

/**
 * Simplified fresnel using pow(1 - dot, 3)
 * GLSL equivalent: pow(1.0 - max(dot(viewDir, normal), 0.0), 3.0)
 */
const simpleFresnel = Fn(({ normal, viewDir }) => {
    const dotProd = max(dot(normal.normalize(), viewDir.normalize()), 0);
    return float(1).sub(dotProd).pow(3);
});

// ============================================================================
// COLOR CALCULATION FUNCTIONS
// ============================================================================

/**
 * Calculate water thickness based on wave height
 * GLSL equivalent: vThickness = 0.5 + vWaveHeight * 2.0 + uDuetProgress * 0.5
 */
const calculateThickness = Fn(({ waveHeight, duetProgress }) => {
    return float(0.5).add(waveHeight.mul(2)).add(duetProgress.mul(0.5));
});

/**
 * Calculate transmittance (light absorption through water)
 * GLSL equivalent: exp(-density * vThickness * (1.0 - diffuseColor))
 */
const calculateTransmittance = Fn(({ thickness, density }) => {
    return thickness.mul(density).mul(float(1).sub(DIFFUSE_COLOR)).neg().exp();
});

/**
 * Calculate caustics pattern
 * GLSL equivalent: Multi-octave noise with animated offset
 */
const calculateCaustics = Fn(({ worldPos, timeParam }) => {
    const causticUv = worldPos.xz.mul(0.2);

    // Octave 1
    const p1 = causticUv.mul(8).add(timeParam.mul(0.5).mul(vec2(0.3, 0.3)));
    const n1 = perlin({ p: p1 }).abs();

    // Octave 2
    const p2 = causticUv.mul(12).add(timeParam.mul(0.5).mul(vec2(0.4, 0.35)).add(vec2(100, 0)));
    const n2 = perlin({ p: p2 }).abs();

    // Octave 3
    const p3 = causticUv.mul(16).add(timeParam.mul(0.5).mul(vec2(0.5, 0.4)).add(vec2(0, 100)));
    const n3 = perlin({ p: p3 }).abs();

    return n1.mul(1).add(n2.mul(0.5)).add(n3.mul(0.333)).mul(0.5);
});

// ============================================================================
// FRAGMENT COLOR FUNCTION
// ============================================================================

/**
 * Calculate final fragment color with all water effects
 * GLSL equivalent: Combines depth color, transmittance, caustics, bioluminescence, fresnel
 */
const waterColor = Fn(({ timeParam, duetProgress, harmonicResonance, cameraPos, frequencies, velocities, jellyPositions, jellyVelocities, jellyActive, thicknessValue }) => {
    const worldPos = positionWorld;
    const normal = normalWorld;
    const viewDir = cameraPos.sub(worldPos).normalize();

    // Calculate fresnel
    const fresnel = simpleFresnel({ normal, viewDir });

    // Calculate caustics
    const caustic = calculateCaustics({ worldPos, timeParam });

    // Depth-based colors
    const baseColor = mix(DEEP_COLOR, SHALLOW_COLOR, float(0.3)); // Default shallow contribution

    // Transmittance (water absorbs light with depth)
    const transmittance = calculateTransmittance({ thickness: thicknessValue, density: float(0.7) });
    const refractionColor = vec3(0.7, 0.7, 0.75).mul(transmittance);

    // Bioluminescent glow
    const stringGlow = stringBioluminescence({
        position: worldPos,
        frequencies,
        velocities
    });
    const bioluminescence = BIOLUMINESCENT_COLOR.mul(float(0.3).add(caustic.mul(0.2)))
        .add(stringGlow.mul(0.5))
        .mul(float(1).add(harmonicResonance.mul(2)))
        .mul(float(0.5).add(duetProgress.mul(1.5)));

    // Reflection (sky gradient)
    const skyReflection = mix(vec3(0.3, 0.8, 1), vec3(0.1, 0.4, 0.6), fresnel).mul(0.3);

    // Mix refraction with reflection based on fresnel
    const finalColor = mix(refractionColor, skyReflection, fresnel).add(bioluminescence);

    // Add caustic diffuse contribution
    finalColor.addAssign(DIFFUSE_COLOR.mul(caustic.mul(0.1)));

    return finalColor;
});

// ============================================================================
// MATERIAL CLASS
// ============================================================================

/**
 * WaterMaterialTSL - TSL-based Water Material
 *
 * Extends MeshPhysicalNodeMaterial to provide TSL shader functionality
 * while maintaining API compatibility with the original GLSL WaterMaterial.
 *
 * Features:
 * - Fresnel-based transparency and edge glow
 * - 6-string harp frequency response
 * - Jelly creature ripple interactions
 * - Sphere constraint for duet progress animation
 * - Physical material properties (transmission, IOR, roughness)
 */
export class WaterMaterialTSL extends MeshPhysicalNodeMaterial {
    // Uniform declarations
    private timeUniform = uniform(0);
    private stringFreqUniform = uniform(new Float32Array(6));
    private stringVelUniform = uniform(new Float32Array(6));
    private duetProgressUniform = uniform(0);
    private shipPatienceUniform = uniform(1.0);
    private harmonicResonanceUniform = uniform(0);
    private cameraPositionUniform = uniform(new THREE.Vector3(0, 5, 10));

    // WaterBall-inspired uniforms
    private sphereRadiusUniform = uniform(8.0);
    private sphereCenterUniform = uniform(new THREE.Vector3(0, 0, -5));
    private thicknessUniform = uniform(1.0);
    private densityUniform = uniform(0.7);
    private f0Uniform = uniform(0.02);

    // Jelly creature uniforms
    private jellyPositionsUniform = uniform(new Float32Array(18));  // 6 jellies × 3 (xyz)
    private jellyVelocitiesUniform = uniform(new Float32Array(6));
    private jellyActiveUniform = uniform(new Float32Array(6));

    // Bioluminescent color
    private bioluminescentColorUniform = uniform(new THREE.Color(0x00ff88));

    constructor() {
        super();

        // Physical water properties
        this.transmission = 0.8;      // Transparent
        this.ior = 1.33;               // Water IOR
        this.roughness = 0.1;          // Smooth surface
        this.thickness = 1.0;          // Water depth
        this.attenuationColor = new THREE.Color(0x004488);
        this.attenuationDistance = 10;

        // Material settings
        this.transparent = true;
        this.side = THREE.DoubleSide;
        this.depthWrite = false;

        // Apply vertex displacement
        this.positionNode = waterDisplacement({
            timeParam: this.timeUniform,
            frequencies: this.stringFreqUniform,
            velocities: this.stringVelUniform,
            duetProgress: this.duetProgressUniform,
            sphereRadius: this.sphereRadiusUniform,
            sphereCenter: this.sphereCenterUniform,
            jellyPositions: this.jellyPositionsUniform,
            jellyVelocities: this.jellyVelocitiesUniform,
            jellyActive: this.jellyActiveUniform
        });

        // Apply fragment color
        this.colorNode = waterColor({
            timeParam: this.timeUniform,
            duetProgress: this.duetProgressUniform,
            harmonicResonance: this.harmonicResonanceUniform,
            cameraPos: this.cameraPositionUniform,
            frequencies: this.stringFreqUniform,
            velocities: this.stringVelUniform,
            jellyPositions: this.jellyPositionsUniform,
            jellyVelocities: this.jellyVelocitiesUniform,
            jellyActive: this.jellyActiveUniform,
            thicknessValue: this.thicknessUniform
        });
    }

    // ========================================================================
    // PUBLIC API (maintains compatibility with original WaterMaterial)
    // ========================================================================

    /**
     * Update time for animation
     */
    public setTime(time: number): void {
        this.timeUniform.value = time;
    }

    /**
     * Update harp string frequency (0-5 for strings C-A)
     */
    public setStringFrequency(index: number, frequency: number): void {
        if (index >= 0 && index < 6) {
            const arr = this.stringFreqUniform.value as Float32Array;
            arr[index] = frequency;
        }
    }

    /**
     * Update harp string velocity for intensity modulation
     */
    public setStringVelocity(index: number, velocity: number): void {
        if (index >= 0 && index < 6) {
            const arr = this.stringVelUniform.value as Float32Array;
            arr[index] = velocity;
        }
    }

    /**
     * Update duet progress (0-1)
     */
    public setDuetProgress(progress: number): void {
        this.duetProgressUniform.value = Math.max(0, Math.min(1, progress));
    }

    /**
     * Update harmonic resonance intensity (0-1)
     */
    public setHarmonicResonance(resonance: number): void {
        this.harmonicResonanceUniform.value = Math.max(0, Math.min(1, resonance));
    }

    /**
     * Update ship patience teaching state
     */
    public setShipPatience(patience: number): void {
        this.shipPatienceUniform.value = Math.max(0, Math.min(1, patience));
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        this.cameraPositionUniform.value.copy(position);
    }

    /**
     * Trigger ripple effect on a specific string
     */
    public triggerStringRipple(stringIndex: number, intensity: number = 1.0): void {
        if (stringIndex >= 0 && stringIndex < 6) {
            const arr = this.stringVelUniform.value as Float32Array;
            arr[stringIndex] = intensity;
        }
    }

    /**
     * Decay all velocities toward zero (call each frame)
     */
    public decayVelocities(deltaTime: number, decayRate: number = 2.0): void {
        const velocities = this.stringVelUniform.value as Float32Array;
        for (let i = 0; i < 6; i++) {
            if (velocities[i] > 0) {
                velocities[i] = Math.max(0, velocities[i] - decayRate * deltaTime);
            }
        }
    }

    // ========== WaterBall-inspired Control Methods ==========

    /**
     * Set sphere radius for plane-to-sphere transformation
     */
    public setSphereRadius(radius: number): void {
        this.sphereRadiusUniform.value = Math.max(0.1, radius);
    }

    /**
     * Set center point for sphere transformation
     */
    public setSphereCenter(center: THREE.Vector3): void {
        this.sphereCenterUniform.value.copy(center);
    }

    /**
     * Set water thickness for transmittance calculation
     */
    public setThickness(thickness: number): void {
        this.thicknessUniform.value = Math.max(0.01, thickness);
        this.thickness = thickness;
    }

    /**
     * Set absorption density
     */
    public setDensity(density: number): void {
        this.densityUniform.value = Math.max(0, density);
    }

    /**
     * Set Fresnel F0 (reflectance at normal incidence)
     */
    public setF0(f0: number): void {
        this.f0Uniform.value = Math.max(0, Math.min(1, f0));
    }

    /**
     * Set bioluminescent color
     */
    public setBioluminescentColor(color: THREE.Color): void {
        this.bioluminescentColorUniform.value.copy(color);
    }

    /**
     * Animate sphere transformation
     */
    public setSphereTransformation(progress: number, radius: number): void {
        this.duetProgressUniform.value = Math.max(0, Math.min(1, progress));
        this.sphereRadiusUniform.value = Math.max(0.1, radius);
    }

    // ========== Jelly Creature Interaction Methods ==========

    /**
     * Update jelly position for water interaction
     */
    public setJellyPosition(index: number, position: THREE.Vector3): void {
        if (index >= 0 && index < 6) {
            const arr = this.jellyPositionsUniform.value as Float32Array;
            arr[index * 3 + 0] = position.x;
            arr[index * 3 + 1] = position.y;
            arr[index * 3 + 2] = position.z;
        }
    }

    /**
     * Update jelly velocity for ripple intensity
     */
    public setJellyVelocity(index: number, velocity: number): void {
        if (index >= 0 && index < 6) {
            const arr = this.jellyVelocitiesUniform.value as Float32Array;
            arr[index] = velocity;
        }
    }

    /**
     * Set jelly active state (visible/hidden)
     */
    public setJellyActive(index: number, active: boolean): void {
        if (index >= 0 && index < 6) {
            const arr = this.jellyActiveUniform.value as Float32Array;
            arr[index] = active ? 1.0 : 0.0;
        }
    }

    /**
     * Update all jelly positions from JellyManager
     */
    public updateJellyPositions(jellyManager: any): void {
        if (!jellyManager) return;

        for (let i = 0; i < 6; i++) {
            const jelly = jellyManager.getJelly?.(i);
            if (jelly && !jelly.isHidden?.()) {
                this.setJellyPosition(i, jelly.position);
                const state = jelly.getState?.();
                const isMoving = state === 'spawning' || state === 'submerging';
                this.setJellyVelocity(i, isMoving ? 0.5 : 0.1);
                this.setJellyActive(i, true);
            } else {
                this.setJellyActive(i, false);
            }
        }
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
    // Noise functions
    simplexNoise,
    perlin,
    fbm,
    // String effects
    calculateStringWave,
    stringResonanceEffect,
    stringBioluminescence,
    // Jelly effects
    calculateJellyRipple,
    jellyRippleEffect,
    // Sphere constraint
    sphereConstraint,
    // Vertex displacement
    waterDisplacement,
    // Fresnel
    waterFresnel,
    simpleFresnel,
    // Color calculations
    calculateThickness,
    calculateTransmittance,
    calculateCaustics,
    waterColor,
    // Color constants
    BIOLUMINESCENT_COLOR,
    DEEP_COLOR,
    SHALLOW_COLOR,
    DIFFUSE_COLOR,
};

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default WaterMaterialTSL;
