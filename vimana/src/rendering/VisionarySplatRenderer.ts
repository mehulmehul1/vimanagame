/**
 * VisionarySplatRenderer.ts - Gaussian Splat Renderer
 * =============================================================================
 *
 * This module provides a Gaussian splat rendering interface for Vimana project
 * using Visionary-Core package for WebGPU-powered rendering with proper depth occlusion.
 */

import * as THREE from 'three';
import {
  GaussianThreeJSRenderer,
  GaussianModel
} from 'visionary-core';
import { SparkRenderer } from '@sparkjsdev/spark';

/**
 * Splat model entry for loading
 */
export interface SplatEntry {
  url: string;
  position?: THREE.Vector3;
  rotation?: THREE.Euler;
  scale?: THREE.Vector3;
}

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[VisionarySplatRenderer] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[VisionarySplatRenderer] ${msg}`, ...args),
  error: (msg: string, ...args: unknown[]) => console.error(`[VisionarySplatRenderer] ${msg}`, ...args),
};

/**
 * Renderer configuration options
 */
export interface SplatRendererOptions {
  apertureAngle?: number;
  focalDistance?: number;
  maxStdDev?: number;
  minAlpha?: number;
  minOpacity?: number;  // Visionary uses minOpacity instead of minAlpha
  renderOrder?: number;
}

/**
 * Visionary Splat Renderer
 * 
 * Uses WebGPU for rendering Gaussian splats correctly composited with 3D scene.
 */
export class VisionarySplatRenderer {
  private visionary: GaussianThreeJSRenderer | null = null;
  private spark: SparkRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private renderer: any = null;  // Use any to avoid TypeScript errors with WebGPURenderer
  private isInitialized = false;
  private usingVisionary = false;
  private gaussianModels: any[] = [];  // Store loaded Gaussian models

  // Compatibility shims for Spark JS properties used by CharacterController
  private _apertureAngle: number = 0;
  private _focalDistance: number = 0;

  get apertureAngle(): number { return this._apertureAngle; }
  set apertureAngle(v: number) { this._apertureAngle = v; }

  get focalDistance(): number { return this._focalDistance; }
  set focalDistance(v: number) { this._focalDistance = v; }

  /**
   * Initialize the splat renderer
   */
  async initialize(
    renderer: THREE.WebGLRenderer | any,
    scene: THREE.Scene,
    options: SplatRendererOptions = {}
  ): Promise<void> {
    this.scene = scene;
    this.renderer = renderer;

    // Detect if this is a WebGPURenderer
    const isWebGPU = !!(renderer as any).isWebGPURenderer;

    if (isWebGPU) {
      try {
        logger.log('✨ Initializing Visionary renderer (WebGPU)...');

        // Create GaussianThreeJSRenderer with model list
        // Note: We pass empty array initially, models will be added via loadSplat()
        this.visionary = new GaussianThreeJSRenderer(renderer, scene, this.gaussianModels);
        await this.visionary.init();

        // Add to scene (it's a THREE.Mesh)
        scene.add(this.visionary);

        this.usingVisionary = true;
        logger.log('✅ Visionary renderer initialized');
      } catch (e) {
        logger.error(`Failed to initialize Visionary: ${e instanceof Error ? e.message : String(e)}`);
        logger.warn('Falling back to SparkRenderer... (Warning: WebGPU might conflict with Spark)');
        this.initSpark(renderer, scene, options);
      }
    } else {
      logger.log('🎨 WebGL2 detected - using SparkRenderer fallback');
      this.initSpark(renderer, scene, options);
    }

    this.isInitialized = true;
  }

  private initSpark(renderer: THREE.WebGLRenderer | any, scene: THREE.Scene, options: SplatRendererOptions) {
    this.spark = new SparkRenderer({
      renderer,
      apertureAngle: options.apertureAngle ?? 2 * Math.atan(0.005),
      focalDistance: options.focalDistance ?? 6.0,
      maxStdDev: options.maxStdDev ?? Math.sqrt(6),
      minAlpha: options.minAlpha ?? 0.00033,
    });
    this.spark.renderOrder = options.renderOrder ?? 9998;
    scene.add(this.spark);
    this.usingVisionary = false;
  }

  /**
   * Load a Gaussian splat model
   * For Visionary: Creates GaussianModel and adds to renderer
   * For Spark: Creates SplatMesh and adds to spark
   */
  async loadSplat(
    url: string,
    position: THREE.Vector3 = new THREE.Vector3(0, 0, 0),
    rotation: THREE.Euler = new THREE.Euler(0, 0, 0),
    scale: THREE.Vector3 = new THREE.Vector3(1, 1, 1)
  ): Promise<THREE.Object3D | null> {
    if (!this.isInitialized) throw new Error('Not initialized');

    if (this.spark) {
      // Legacy Spark fallback
      // @ts-ignore - SparkRenderer might have SplatMesh static property or need import
      const splatMesh = new SparkRenderer.SplatMesh({ url });
      this.spark.add(splatMesh);
      splatMesh.position.copy(position);
      splatMesh.rotation.copy(rotation);
      splatMesh.scale.copy(scale);
      return splatMesh;
    }

    // Visionary path: Use GaussianThreeJSRenderer with GaussianModel
    if (this.visionary && this.scene) {
      try {
        logger.log(`Loading Visionary model from: ${url}`);

        // Create entry for GaussianModel
        const entry: SplatEntry = {
          url,
          position,
          rotation,
          scale
        };

        // Create GaussianModel instance
        const model = new GaussianModel(entry);

        // Add to scene and model list
        this.scene.add(model);
        this.gaussianModels.push(model);

        logger.log(`✅ Visionary model loaded: ${url}`);
        return model;
      } catch (e) {
        logger.error(`Failed to load Visionary model: ${e instanceof Error ? e.message : String(e)}`);
        return null;
      }
    }

    return null;
  }

  /**
   * Load multiple splat models
   */
  async loadSplats(models: SplatEntry[]): Promise<void> {
    for (const model of models) {
      await this.loadSplat(model.url, model.position, model.rotation, model.scale);
    }
  }

  /**
   * Render loop update
   * Visionary: Renders Three.js scene and draws splats with depth occlusion
   * Spark: Renders normally (renderer handles it)
   */
  render(camera: THREE.Camera): void {
    if (this.usingVisionary && this.visionary && this.renderer) {
      // Visionary dual-pass rendering per documentation
      // 1. Render Three.js scene (captures depth)
      this.visionary.renderThreeScene(camera);
      // 2. Draw splats with depth occlusion
      this.visionary.drawSplats(this.renderer, this.scene!, camera);
    }
    // Spark renderer handles its own rendering automatically
  }

  /**
   * Render environment map
   * Shims to SparkRenderer if available for EnvMap generation
   */
  async renderEnvMap(options: any): Promise<THREE.Texture | null> {
    if (this.spark) {
      // Fallback to Spark's environment map generation if available
      return this.spark.renderEnvMap(options);
    }

    // If we're fully WebGPU/Visionary-only and no Spark instance:
    // We might need to implement a WebGPU-compatible env map capture or return null
    // For now, logging warning and returning null to prevent crash
    logger.warn('renderEnvMap called but no legacy Spark instance available. EnvMap generation skipped.');
    return null;
  }

  dispose(): void {
    if (this.visionary) {
      this.visionary.disposeDepthResources();
      this.scene?.remove(this.visionary);
      this.visionary = null;
    }
    if (this.spark) {
      this.scene?.remove(this.spark);
      this.spark = null;
    }
    this.isInitialized = false;
  }
}

export async function createSplatRenderer(
  renderer: THREE.WebGLRenderer | any,
  scene: THREE.Scene,
  options: SplatRendererOptions = {}
): Promise<VisionarySplatRenderer> {
  const splatRenderer = new VisionarySplatRenderer();
  await splatRenderer.initialize(renderer, scene, options);
  return splatRenderer;
}

export async function isVisionaryAvailable(): Promise<boolean> {
  return true;
}
