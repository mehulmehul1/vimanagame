/**
 * VisionarySplatRenderer.ts - Gaussian Splat Renderer Wrapper
 * =============================================================================
 *
 * This module provides a Gaussian splat rendering interface for the Vimana project.
 *
 * **IMPORTANT NOTE - Visionary Integration Status:**
 *
 * The Visionary package (visionary-core) from GitHub has installation issues
 * related to ONNX runtime WASM file handling in the postinstall script. This
 * prevents direct npm installation at this time.
 *
 * **Current Implementation:**
 * - Uses the existing @sparkjsdev/spark package for Gaussian splat rendering
 * - SparkRenderer is WebGL2-based (not WebGPU)
 * - Does NOT provide proper depth occlusion between splats and scene meshes
 *
 * **Path Forward for Visionary:**
 * 1. Clone Visionary directly: git clone https://github.com/Visionary-Laboratory/visionary
 * 2. Build locally: npm install && npm run build
 * 3. Install from local build: npm install ../path/to/visionary
 * 4. Or wait for Visionary to be published to npm with fixed build
 *
 * **Visionary Architecture (when available):**
 * - Uses WebGPU for rendering
 * - Provides proper depth occlusion via shared depth buffer
 * - Supports PLY and SPLAT file formats
 * - Integrates with Three.js scene graph
 *
 * ==============================================================================
 */

import * as THREE from 'three';
import { SparkRenderer } from '@sparkjsdev/spark';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[VisionarySplatRenderer] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[VisionarySplatRenderer] ${msg}`, ...args),
  error: (msg: string, ...args: unknown[]) => console.error(`[VisionarySplatRenderer] ${msg}`, ...args),
};

/**
 * Splat model interface for loading Gaussian splat files
 */
export interface SplatModel {
  position: THREE.Vector3;
  rotation: THREE.Euler;
  scale: THREE.Vector3;
  url: string;
}

/**
 * Renderer configuration options
 */
export interface SplatRendererOptions {
  apertureAngle?: number;
  focalDistance?: number;
  maxStdDev?: number;
  minAlpha?: number;
  renderOrder?: number;
}

/**
 * Visionary Splat Renderer
 *
 * Currently wraps SparkRenderer as a fallback while Visionary integration
 * is pending resolution of installation issues.
 *
 * When Visionary is available, this class will use:
 * - GaussianThreeJSRenderer from visionary-core
 * - Proper WebGPU rendering with depth occlusion
 * - PLY/SPLAT file format support
 */
export class VisionarySplatRenderer {
  private spark: SparkRenderer | null = null;
  private renderer: THREE.WebGLRenderer | THREE.WebGPURenderer | null = null;
  private scene: THREE.Scene | null = null;
  private isInitialized = false;
  private usingVisionary = false; // Set to true when Visionary is integrated

  /**
   * Initialize the splat renderer
   *
   * @param renderer - Three.js renderer (WebGL2 or WebGPU)
   * @param scene - Three.js scene to add splats to
   * @param options - Optional configuration
   */
  async initialize(
    renderer: THREE.WebGLRenderer | THREE.WebGPURenderer,
    scene: THREE.Scene,
    options: SplatRendererOptions = {}
  ): Promise<void> {
    this.renderer = renderer;
    this.scene = scene;

    const {
      apertureAngle = 2 * Math.atan(0.005),
      focalDistance = 6.0,
      maxStdDev = Math.sqrt(6),
      minAlpha = 0.00033,
      renderOrder = 9998,
    } = options;

    // Detect renderer type
    const isWebGPU = (renderer as any).isWebGPURenderer === true;

    if (isWebGPU) {
      logger.log('🎨 WebGPU renderer detected');
      // TODO: Initialize Visionary when available
      logger.warn('⚠️ Visionary not available - falling back to SparkRenderer (WebGL2)');
    } else {
      logger.log('🎨 WebGL2 renderer detected');
    }

    // Initialize SparkRenderer (current fallback)
    try {
      this.spark = new SparkRenderer({
        renderer,
        apertureAngle,
        focalDistance,
        maxStdDev,
        minAlpha,
      });
      this.spark.renderOrder = renderOrder;
      scene.add(this.spark);
      this.isInitialized = true;

      logger.log('✅ SparkRenderer initialized (fallback mode)');
      logger.warn('⚠️ Note: SparkRenderer does not provide depth occlusion for splats');
      logger.warn('⚠️ Splits will render on top of all geometry regardless of depth');
    } catch (e) {
      logger.error(`Failed to initialize SparkRenderer: ${e instanceof Error ? e.message : String(e)}`);
      throw e;
    }
  }

  /**
   * Load a Gaussian splat model
   *
   * Currently this is a placeholder. The actual implementation
   * will depend on the splat file format and how SparkRenderer
   * handles file loading.
   *
   * @param url - URL to the splat file (.ply or .splat)
   * @param position - World position for the splat
   * @param rotation - Rotation for the splat
   * @param scale - Scale for the splat
   */
  async loadSplat(
    url: string,
    position: THREE.Vector3 = new THREE.Vector3(0, 0, 0),
    rotation: THREE.Euler = new THREE.Euler(0, 0, 0),
    scale: THREE.Vector3 = new THREE.Vector3(1, 1, 1)
  ): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('VisionarySplatRenderer not initialized. Call initialize() first.');
    }

    logger.log(`📦 Loading splat: ${url}`);

    // TODO: Implement actual splat file loading
    // This depends on the API of the underlying renderer
    //
    // For Visionary (when available):
    //   const model = await GaussianModel.load(url);
    //   model.position.copy(position);
    //   model.rotation.copy(rotation);
    //   model.scale.copy(scale);
    //   this.models.push(model);
    //
    // For SparkRenderer:
    //   Check if SparkRenderer has a load method
    //   or if splats need to be pre-loaded differently

    logger.warn(`⚠️ Splat loading not yet implemented for SparkRenderer`);
    logger.warn(`⚠️ URL: ${url}`);
  }

  /**
   * Load multiple splat models (e.g., for 6 jelly creatures)
   *
   * @param models - Array of splat model descriptors
   */
  async loadSplats(models: SplatModel[]): Promise<void> {
    logger.log(`📦 Loading ${models.length} splat models...`);

    for (const model of models) {
      await this.loadSplat(
        model.url,
        model.position,
        model.rotation,
        model.scale
      );
    }
  }

  /**
   * Render the splats
   *
   * This should be called in the render loop before the main render.
   *
   * @param camera - Active camera for rendering
   */
  render(camera: THREE.Camera): void {
    if (!this.isInitialized) {
      return;
    }

    // SparkRenderer handles rendering automatically when added to scene
    // Just sync position if needed
    if (this.spark && camera instanceof THREE.PerspectiveCamera) {
      // SparkRenderer position should sync with camera
      // This is typically handled by the scene graph
    }
  }

  /**
   * Update the renderer's position (for camera-following splats)
   *
   * @param position - World position to sync to
   */
  setPosition(position: THREE.Vector3): void {
    if (this.spark) {
      this.spark.position.copy(position);
    }
  }

  /**
   * Check if the renderer is using Visionary (true) or fallback (false)
   */
  isUsingVisionary(): boolean {
    return this.usingVisionary;
  }

  /**
   * Get renderer info for debugging
   */
  getInfo(): {
    initialized: boolean;
    usingVisionary: boolean;
    hasSpark: boolean;
    rendererType: string;
  } {
    return {
      initialized: this.isInitialized,
      usingVisionary: this.usingVisionary,
      hasSpark: this.spark !== null,
      rendererType: this.renderer ? (this.renderer as any).type : 'none',
    };
  }

  /**
   * Dispose of renderer resources
   */
  dispose(): void {
    logger.log('🧹 Disposing VisionarySplatRenderer...');

    if (this.spark && this.scene) {
      this.scene.remove(this.spark);
      this.spark = null;
    }

    this.isInitialized = false;
    this.usingVisionary = false;

    logger.log('✅ VisionarySplatRenderer disposed');
  }
}

/**
 * Factory function to create and initialize a VisionarySplatRenderer
 *
 * @param renderer - Three.js renderer
 * @param scene - Three.js scene
 * @param options - Optional configuration
 * @returns Initialized VisionarySplatRenderer instance
 */
export async function createSplatRenderer(
  renderer: THREE.WebGLRenderer | THREE.WebGPURenderer,
  scene: THREE.Scene,
  options: SplatRendererOptions = {}
): Promise<VisionarySplatRenderer> {
  const splatRenderer = new VisionarySplatRenderer();
  await splatRenderer.initialize(renderer, scene, options);
  return splatRenderer;
}

/**
 * Check if Visionary is available (for future use)
 *
 * This function will check if the visionary-core package is properly
 * installed and can be imported.
 */
export async function isVisionaryAvailable(): Promise<boolean> {
  // TODO: Implement actual check when Visionary is integrated
  // For now, always return false since we're using SparkRenderer fallback
  return false;
}

/**
 * Get Visionary installation instructions
 */
export function getVisionaryInstallInstructions(): string {
  return `
Visionary Integration Instructions:
==================================

The Visionary package (visionary-core) is currently not published to npm
due to build issues with ONNX runtime WASM file handling.

To install Visionary manually:

1. Clone the repository:
   git clone https://github.com/Visionary-Laboratory/visionary

2. Navigate to the directory:
   cd visionary

3. Install dependencies (may require fixes):
   npm install

4. Build the package:
   npm run build

5. Install from local build:
   npm install ../path/to/visionary

Alternative: Use @mkkellogg/gaussian-splats-3d
npm install @mkkellogg/gaussian-splats-3d

Note: The project currently uses @sparkjsdev/spark as a fallback.
For proper depth occlusion, Visionary or a similar WebGPU-based solution
is required.
`.trim();
}
