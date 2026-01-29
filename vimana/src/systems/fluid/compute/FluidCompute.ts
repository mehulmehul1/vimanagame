/**
 * FluidCompute.ts - TSL Compute System for WebGPU Fluid Simulation
 * ==================================================================
 *
 * Main compute node class for executing fluid simulation via TSL compute shaders.
 * Integrates with Three.js WebGPU renderer's compute() API.
 *
 * Based on Story 4.7 implementation requirements:
 * - WebGPURenderer context passed to fluid system
 * - Compute shaders execute via renderer.compute(computeNode)
 * - Compute executed BEFORE render in loop
 * - TSL compute shaders for particle physics
 * - 10,000 particles at 60 FPS on desktop
 * - Storage buffers: position, velocity, force
 * - Boundary collision with If() conditions
 * - Sphere constraint animation integrated
 */

import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import {
  Fn,
  instanceIndex,
  compute,
  uniform,
} from 'three/tsl';
import { ParticleBuffers, createParticleBuffers } from './ParticleBuffers';
import {
  createSimpleFluidCompute,
  createMLSMPMUniforms,
  updateUniforms,
  getUniformValues,
  DEFAULT_MLSPM_UNIFORMS,
  MLSMPMUniforms,
} from './MLSMPMStages';
import { vec3, FLUID_CONFIG } from '../types';

const logger = {
  log: (msg: string, ...args: unknown[]) => console.log(`[FluidCompute] ${msg}`, ...args),
  warn: (msg: string, ...args: unknown[]) => console.warn(`[FluidCompute] ${msg}`, ...args),
  error: (msg: string, ...args: unknown[]) => console.error(`[FluidCompute] ${msg}`, ...args),
};

/**
 * Configuration for TSL fluid compute system
 */
export interface FluidComputeOptions {
  particleCount?: number;
  boxSize?: vec3;
  sphereRadius?: number;
  sphereCenter?: vec3;
  gravity?: number;
  deltaTime?: number;
  initPattern?: 'grid' | 'sphere';
}

/**
 * State for sphere constraint animation (tunnel effect)
 */
export interface SphereConstraintState {
  radius: number;
  targetRadius: number;
  centerX: number;
  centerY: number;
  centerZ: number;
  isAnimating: boolean;
  animationSpeed: number;
}

/**
 * FluidComputeNode - Main TSL compute system for fluid simulation
 *
 * Uses Three.js WebGPU compute shader API to execute fluid physics
 * on the GPU. Particles are simulated using TSL compute functions.
 */
export class FluidComputeNode {
  // Particle storage buffers
  public buffers: ParticleBuffers;

  // TSL uniforms
  public uniforms: MLSMPMUniforms;

  // Compute function (TSL Fn)
  private computeFn: ReturnType<typeof compute> | null = null;

  // Simulation state
  public readonly particleCount: number;
  private enabled = true;
  private lastUpdateTime = 0;

  // Sphere constraint for tunnel animation
  public sphereState: SphereConstraintState = {
    radius: 15.0,
    targetRadius: 15.0,
    centerX: 0,
    centerY: 0,
    centerZ: 0,
    isAnimating: false,
    animationSpeed: 0.5,
  };

  // Performance tracking
  private computeStartTime = 0;
  public computeTimeMs = 0;
  public lastFrameComputeTime = 0;

  constructor(options: FluidComputeOptions = {}) {
    const {
      particleCount = FLUID_CONFIG.maxParticles,
      boxSize = [52, 52, 52],
      sphereRadius = 15.0,
      sphereCenter = [0, 0, 0],
      gravity = 9.8,
      deltaTime = 0.016,
      initPattern = 'grid',
    } = options;

    this.particleCount = Math.min(particleCount, FLUID_CONFIG.maxParticles);

    // Create TSL uniforms
    this.uniforms = createMLSMPMUniforms({
      deltaTime,
      gravity,
      sphereRadius,
      sphereCenter: new THREE.Vector3(...sphereCenter),
      boxSize: new THREE.Vector3(...boxSize),
    });

    // Create particle buffers with grid initialization
    this.buffers = createParticleBuffers(
      this.particleCount,
      initPattern,
      { boxSize, sphereRadius, sphereCenter }
    );

    // Initialize sphere state
    this.sphereState = {
      radius: sphereRadius,
      targetRadius: sphereRadius,
      centerX: sphereCenter[0],
      centerY: sphereCenter[1],
      centerZ: sphereCenter[2],
      isAnimating: false,
      animationSpeed: 0.5,
    };

    // Create compute function
    this.createComputeFunction();

    logger.log(`FluidComputeNode initialized with ${this.particleCount} particles`);
  }

  /**
   * Create TSL compute function for particle simulation
   */
  private createComputeFunction(): void {
    // Create simplified fluid compute using TSL
    const fn = createSimpleFluidCompute(
      this.buffers.positionBuffer,
      this.buffers.velocityBuffer,
      this.buffers.forceBuffer,
      this.uniforms
    );

    // @ts-ignore - TSL compute() returns a compute node
    this.computeFn = fn().compute(this.particleCount);

    logger.log('TSL compute function created');
  }

  /**
   * Dispatch compute shader on WebGPU renderer
   * This must be called BEFORE renderer.render() in the animation loop
   *
   * @param renderer - Three.js WebGPU renderer
   * @param deltaTime - Frame delta time in seconds
   */
  public dispatch(renderer: WebGPURenderer, deltaTime: number): void {
    if (!this.enabled || !this.computeFn) {
      return;
    }

    this.computeStartTime = performance.now();

    // Update delta time uniform
    this.uniforms.deltaTime.value = deltaTime;
    this.uniforms.dt.value = deltaTime;

    // Update sphere constraint animation
    this.updateSphereConstraint(deltaTime);

    // Execute compute shader via Three.js WebGPU renderer
    // @ts-ignore - renderer.compute() is available in WebGPURenderer
    renderer.compute(this.computeFn);

    // Track compute time
    this.lastFrameComputeTime = performance.now() - this.computeStartTime;
    this.computeTimeMs = this.computeTimeMs * 0.9 + this.lastFrameComputeTime * 0.1; // Smooth average

    this.lastUpdateTime = performance.now();
  }

  /**
   * Update sphere constraint for tunnel animation
   * Called automatically during dispatch()
   */
  private updateSphereConstraint(deltaTime: number): void {
    if (!this.sphereState.isAnimating) {
      return;
    }

    // Animate radius toward target
    const diff = this.sphereState.targetRadius - this.sphereState.radius;
    if (Math.abs(diff) < 0.01) {
      this.sphereState.radius = this.sphereState.targetRadius;
      this.sphereState.isAnimating = false;
    } else {
      this.sphereState.radius += diff * this.sphereState.animationSpeed * deltaTime;
    }

    // Update uniform
    this.uniforms.sphereRadius.value = this.sphereState.radius;
    (this.uniforms.sphereCenter.value as THREE.Vector3).set(
      this.sphereState.centerX,
      this.sphereState.centerY,
      this.sphereState.centerZ
    );
  }

  /**
   * Set sphere constraint radius
   * Used for tunnel animation (box width ratio changes)
   */
  public setSphereRadius(radius: number, animate: boolean = false): void {
    if (animate) {
      this.sphereState.targetRadius = radius;
      this.sphereState.isAnimating = true;
    } else {
      this.sphereState.radius = radius;
      this.sphereState.targetRadius = radius;
      this.uniforms.sphereRadius.value = radius;
    }
  }

  /**
   * Set sphere constraint center position
   */
  public setSphereCenter(center: vec3): void {
    this.sphereState.centerX = center[0];
    this.sphereState.centerY = center[1];
    this.sphereState.centerZ = center[2];

    (this.uniforms.sphereCenter.value as THREE.Vector3).set(center[0], center[1], center[2]);
  }

  /**
   * Get current sphere radius
   */
  public getSphereRadius(): number {
    return this.sphereState.radius;
  }

  /**
   * Animate sphere radius to target value
   */
  public animateSphereToRadius(
    targetRadius: number,
    speed: number = 0.5,
    onComplete?: () => void
  ): void {
    this.sphereState.targetRadius = targetRadius;
    this.sphereState.animationSpeed = speed;
    this.sphereState.isAnimating = true;

    if (onComplete) {
      const checkAnimation = () => {
        if (!this.sphereState.isAnimating) {
          onComplete();
        } else {
          requestAnimationFrame(checkAnimation);
        }
      };
      requestAnimationFrame(checkAnimation);
    }
  }

  /**
   * Change simulation box size
   * Updates boundary constraints
   */
  public setBoxSize(size: vec3): void {
    (this.uniforms.boxSize.value as THREE.Vector3).set(size[0], size[1], size[2]);
  }

  /**
   * Set gravity strength
   */
  public setGravity(gravity: number): void {
    this.uniforms.gravity.value = gravity;
  }

  /**
   * Enable/disable compute simulation
   */
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Check if compute is enabled
   */
  public isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Reset particles to initial grid formation
   */
  public reset(initPattern?: 'grid' | 'sphere'): void {
    const pattern = initPattern || 'grid';
    const boxSize = this.uniforms.boxSize.value as THREE.Vector3;
    const sphereRadius = this.uniforms.sphereRadius.value as number;
    const sphereCenter = this.uniforms.sphereCenter.value as THREE.Vector3;

    // Re-initialize buffers
    if (pattern === 'grid') {
      this.buffers.initializeGrid(
        [boxSize.x, boxSize.y, boxSize.z],
        sphereRadius
      );
    } else {
      this.buffers.initializeSphere(
        [sphereCenter.x, sphereCenter.y, sphereCenter.z],
        sphereRadius
      );
    }

    logger.log(`Particles reset with ${pattern} pattern`);
  }

  /**
   * Spawn additional particles
   * Returns actual count spawned (may be less if at capacity)
   */
  public spawnParticles(count: number): number {
    const currentCount = this.buffers.particleCount;
    const maxCount = this.buffers.maxParticles;
    const spawnCount = Math.min(count, maxCount - currentCount);

    if (spawnCount <= 0) {
      return 0;
    }

    // For TSL compute, we need to expand the buffer
    // This is a limitation - particle count is fixed at creation
    logger.warn(`Cannot spawn particles in TSL compute - count is fixed at ${maxCount}`);

    return 0;
  }

  /**
   * Get particle data for debugging/inspection
   */
  public getParticleData(index: number) {
    return this.buffers.getParticle(index);
  }

  /**
   * Get all particle data (useful for debugging)
   */
  public getAllParticleData() {
    return this.buffers.getData();
  }

  /**
   * Get performance statistics
   */
  public getPerformanceStats() {
    return {
      computeTimeMs: this.computeTimeMs,
      lastFrameComputeTime: this.lastFrameComputeTime,
      particleCount: this.particleCount,
      enabled: this.enabled,
      sphereRadius: this.sphereState.radius,
      targetRadius: this.sphereState.targetRadius,
      isAnimating: this.sphereState.isAnimating,
    };
  }

  /**
   * Get simulation state
   */
  public getState() {
    return {
      numParticles: this.particleCount,
      sphereRadius: this.sphereState.radius,
      boxSize: this.uniforms.boxSize.value as THREE.Vector3,
      gravity: this.uniforms.gravity.value as number,
    };
  }

  /**
   * Dispose of compute resources
   */
  public dispose(): void {
    this.buffers.dispose();
    this.computeFn = null;
    logger.log('FluidComputeNode disposed');
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create a TSL fluid compute system with default settings
 */
export function createFluidCompute(options?: FluidComputeOptions): FluidComputeNode {
  return new FluidComputeNode(options);
}

/**
 * Create a TSL fluid compute system optimized for desktop
 */
export function createDesktopFluidCompute(): FluidComputeNode {
  return new FluidComputeNode({
    particleCount: FLUID_CONFIG.maxParticles, // 10,000 particles
    boxSize: [52, 52, 52],
    sphereRadius: 15.0,
    gravity: 9.8,
  });
}

/**
 * Create a TSL fluid compute system optimized for mobile
 */
export function createMobileFluidCompute(): FluidComputeNode {
  return new FluidComputeNode({
    particleCount: 3000, // Reduced particle count for mobile
    boxSize: [52, 52, 52],
    sphereRadius: 15.0,
    gravity: 9.8,
  });
}

// ============================================================================
// RENDER LOOP INTEGRATION EXAMPLE
// ============================================================================

/**
 * Example: How to integrate FluidComputeNode with render loop
 *
 * ```typescript
 * import { createWebGPURenderer } from './rendering/createWebGPURenderer';
 * import { createFluidCompute } from './systems/fluid/compute/FluidCompute';
 *
 * // Initialize
 * const rendererResult = await createWebGPURenderer();
 * const renderer = rendererResult.renderer as WebGPURenderer;
 *
 * const fluidCompute = createFluidCompute({
 *   particleCount: 10000,
 *   boxSize: [52, 52, 52],
 *   sphereRadius: 15.0,
 * });
 *
 * // In render loop
 * renderer.setAnimationLoop((time) => {
 *   const t = time * 0.001;
 *   const delta = 0.016; // or calculate from frame time
 *
 *   // Step 1: Execute compute shader (BEFORE render)
 *   fluidCompute.dispatch(renderer, delta);
 *
 *   // Step 2: Render scene with updated particle data
 *   renderer.render(scene, camera);
 * });
 *
 * // Cleanup
 * fluidCompute.dispose();
 * ```
 */

export default FluidComputeNode;
