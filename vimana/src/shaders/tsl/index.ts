/**
 * TSL Shaders Index
 * ================
 *
 * Exports all TSL (Three.js Shading Language) shader implementations.
 * These shaders output WGSL for WebGPU rendering.
 *
 * Story: 4.3 - Vortex Shader TSL Migration
 * Story: 4.4 - Water Material TSL Migration
 * Story: 4.5 - Shell SDF TSL Migration
 * Story: 4.6 - Jelly Shader TSL Migration
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

export {
    ShellMaterialTSL,
    snoise3 as shellSnoise3,
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
    default as ShellDefault,
} from './ShellShader';

export {
    JellyMaterialTSL,
    snoise3 as jellySnoise3,
    fbm3,
    jellyPulse,
    jellyFresnel,
    bellFactor,
    jellyDisplacement,
    jellyEmissive,
    jellyAlpha,
    BIOLUMINESCENT_COLOR as JELLY_BIOLUMINESCENT_COLOR,
    BASE_JELLY_COLOR,
    INTERNAL_COLOR,
    TEACHING_GLOW_COLOR,
    default as JellyDefault,
} from './JellyShader';

// Re-export for convenience
export { VortexMaterialTSL as VortexMaterialTSL };
export { WaterMaterialTSL as WaterMaterialTSL };
export { ShellMaterialTSL as ShellMaterialTSL };
export { JellyMaterialTSL as JellyMaterialTSL };

// Type export for TypeScript
export type { default as VortexMaterialTSLType } from './VortexShader';
export type { default as WaterMaterialTSLType } from './WaterShader';
export type { default as ShellMaterialTSLType } from './ShellShader';
export type { default as JellyMaterialTSLType } from './JellyShader';
