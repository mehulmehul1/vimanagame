/**
 * TSL Shaders Index
 * ================
 *
 * Exports all TSL (Three.js Shading Language) shader implementations.
 * These shaders output WGSL for WebGPU rendering.
 *
 * Story: 4.3 - Vortex Shader TSL Migration
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

// Re-export for convenience
export { VortexMaterialTSL as VortexMaterialTSL };

// Type export for TypeScript
export type { default as VortexMaterialTSLType } from './VortexShader';
