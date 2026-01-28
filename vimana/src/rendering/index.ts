/**
 * rendering/index.ts - Rendering module exports
 * =============================================================================
 */

export {
  createWebGPURenderer,
  detectPlatform,
  logPlatformInfo,
  calculatePixelRatio,
  isWebGPUSupported,
  getRendererInfo,
  logRendererInfo,
  setupAnimationLoop,
  showLoadingIndicator,
  hideLoadingIndicator,
  updateLoadingIndicator,
  showLoading,
  hideLoading,
  updateLoading,
} from './createWebGPURenderer';

export {
  VisionarySplatRenderer,
  createSplatRenderer,
  isVisionaryAvailable,
  getVisionaryInstallInstructions,
} from './VisionarySplatRenderer';

export type {
  PlatformInfo,
  RendererResult,
  WebGPURendererOptions,
} from './createWebGPURenderer';

export type {
  SplatModel,
  SplatRendererOptions,
} from './VisionarySplatRenderer';
