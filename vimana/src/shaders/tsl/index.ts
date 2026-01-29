/**
 * TSL Shaders Index
 * ================
 *
 * Exports all TSL (Three.js Shading Language) shader implementations.
 * These shaders output WGSL for WebGPU rendering.
 *
 * Story: 4.3 - Vortex Shader TSL Migration
 * Story: 4.4 - Water Material TSL Migration
 * =====================================================================
 */

export {
    VortexMaterialTSL,
    vortexDisplacement,
    swirlNoise,
    vortexFresnel,
    vortexColor,
    vortexEmissive,
    INNER_COLOR,
    OUTER_COLOR,
    CORE_COLOR,
    default,
} from './VortexShader';

export {
    WaterMaterialTSL,
    simplexNoise,
    perlin,
    fbm,
    calculateStringWave,
    stringResonanceEffect,
    stringBioluminescence,
    calculateJellyRipple,
    jellyRippleEffect,
    sphereConstraint,
    waterDisplacement,
    waterFresnel,
    simpleFresnel,
    calculateThickness,
    calculateTransmittance,
    calculateCaustics,
    waterColor,
    BIOLUMINESCENT_COLOR,
    DEEP_COLOR,
    SHALLOW_COLOR,
    DIFFUSE_COLOR,
    default as WaterDefault,
} from './WaterShader';

// Re-export for convenience
export { VortexMaterialTSL as VortexMaterialTSL };
export { WaterMaterialTSL as WaterMaterialTSL };

// Type export for TypeScript
export type { default as VortexMaterialTSLType } from './VortexShader';
export type { default as WaterMaterialTSLType } from './WaterShader';
