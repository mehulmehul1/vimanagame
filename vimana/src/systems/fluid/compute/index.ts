/**
 * src/systems/fluid/compute/index.ts
 * =================================
 *
 * TSL Compute shader system for WebGPU fluid simulation.
 * Exports main FluidComputeNode class and related utilities.
 *
 * Story 4.7: Fluid System Activation
 */

export { FluidComputeNode, createFluidCompute, createDesktopFluidCompute, createMobileFluidCompute } from './FluidCompute';
export { ParticleBuffers, createParticleBuffers } from './ParticleBuffers';
export {
  createMLSMPMUniforms,
  createSimpleFluidCompute,
  createParticleUpdateStage,
  createP2GStage,
  createGridUpdateStage,
  createG2PStage,
  updateUniforms,
  getUniformValues,
  DEFAULT_MLSPM_UNIFORMS,
} from './MLSMPMStages';

export type { FluidComputeOptions, SphereConstraintState } from './FluidCompute';
export type { BufferResult, ParticleBufferData } from './ParticleBuffers';
export type { MLSMPMUniforms } from './MLSMPMStages';
